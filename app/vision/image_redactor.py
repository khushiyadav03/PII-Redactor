"""
Hinglish: Ye module ek image ke andar ke saare visual PII (text-in-image,
faces, ID-specific fields, QR codes) ko detect karke ACTUAL PIXELS par
solid black rectangles draw karta hai - overlay nahi, pixel-level
modification (assignment requirement #16: irreversible redaction).

FLOW (per image):
  1. OCR chalao -> words + bounding boxes
  2. OCR text par generic PII detector chalao (email/phone/PAN/Aadhaar/
     DOB/etc.) -> matched words ke boxes collect karo
  3. ID document classify karo (PAN/Aadhaar/Passport/unknown)
  4. Agar ID document mila, to label-based heuristic se "name" aur
     "signature" fields bhi mask karo (NER yahan unreliable hai - see
     id_detector.py docstring)
  5. Face detection chalao -> face boxes bhi mask list mein add karo
  6. QR detect karo -> agar ID document ke andar mila to mask karo
  7. Saare boxes par solid black rectangle draw karo, naya image encode
     karke return karo
"""
from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np

from app.config import RedactionPolicy
from app.synthetic.generator import ConsistentReplacer
from app.text.detector import detect_pii_in_text
from app.vision.face_detector import detect_faces
from app.vision.id_detector import classify_id_document, find_id_field_boxes
from app.vision.ocr import OcrWord, run_ocr
from app.vision.qr_detector import detect_qr_codes
from app.text.patterns import AADHAAR_RE, PASSPORT_RE


@dataclass
class ImageRedactionReport:
    """Hinglish: Kya kya redact hua - COUNTS only, raw PII values kabhi store nahi karte."""
    doc_type: str
    doc_confidence: str
    text_pii_boxes_masked: int = 0
    faces_masked: int = 0
    id_fields_masked: int = 0
    photo_regions_masked: int = 0
    signatures_masked: int = 0
    qr_codes_masked: int = 0
    modified: bool = False


def _build_ocr_logical_text(words: List[OcrWord]) -> Tuple[str, List[Tuple[int, int, OcrWord]]]:
    """
    Hinglish: Line boundaries preserve karte hue logical text build karte hain,
    taaki multi-line address detector individual lines ko separate analyze kar sake.
    """
    from app.vision.id_detector import _group_words_into_lines
    lines = _group_words_into_lines(words)
    
    parts = []
    spans = []
    cursor = 0
    
    for i, line in enumerate(lines):
        for j, w in enumerate(line):
            start = cursor
            end = cursor + len(w.text)
            spans.append((start, end, w))
            parts.append(w.text)
            cursor = end + 1
            if j < len(line) - 1:
                parts.append(" ")
            else:
                if i < len(lines) - 1:
                    parts.append("\n")
                    
    return "".join(parts), spans


def _boxes_overlapping_span(spans, start, end) -> List[Tuple[int, int, int, int]]:
    boxes = []
    for s_start, s_end, word in spans:
        if s_end > start and s_start < end:
            boxes.append(word.box)
    return boxes


def _rect_intersect(r1, r2):
    x1 = max(r1[0], r2[0])
    y1 = max(r1[1], r2[1])
    x2 = min(r1[2], r2[2])
    y2 = min(r1[3], r2[3])
    if x2 > x1 and y2 > y1:
        return (x1, y1, x2, y2)
    return None


def _rect_area(r):
    return (r[2] - r[0]) * (r[3] - r[1])


def normalize_and_merge_boxes(boxes: List[Tuple[int, int, int, int]], width: int, height: int) -> List[Tuple[int, int, int, int]]:
    """
    Hinglish: overlapping aur invalid boxes filter aur merge karta hai target coordinates par.
    """
    # 1. Clip and validate
    clipped = []
    for (x1, y1, x2, y2) in boxes:
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        if x2 > x1 and y2 > y1:
            clipped.append((x1, y1, x2, y2))
            
    # 2. Merge overlapping boxes
    merged = []
    while clipped:
        curr = clipped.pop(0)
        has_merged = False
        for i, other in enumerate(merged):
            inter = _rect_intersect(curr, other)
            if inter:
                inter_area = _rect_area(inter)
                min_area = min(_rect_area(curr), _rect_area(other))
                if min_area > 0 and (inter_area / min_area) > 0.4:
                    merged[i] = (
                        min(curr[0], other[0]),
                        min(curr[1], other[1]),
                        max(curr[2], other[2]),
                        max(curr[3], other[3])
                    )
                    has_merged = True
                    break
        if not has_merged:
            merged.append(curr)
            
    return merged


def redact_image_bytes(image_bytes: bytes, policy: RedactionPolicy, format_ext: str = ".png") -> Tuple[bytes, ImageRedactionReport]:
    """
    Hinglish: Entry point - raw image bytes leke, redacted image bytes +
    ek report (sirf counts, raw PII nahi) return karta hai.

    Agar image decode nahi ho payi (corrupted/unsupported format), to
    processing fail ho jayegi (ValueError) taaki unredacted visual PII
    silently leak na ho (Privacy Fail-Closed Requirement).
    """
    np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Image could not be decoded or is corrupted. Failing document processing to prevent PII leak.")

    boxes_to_mask: List[Tuple[int, int, int, int]] = []
    report = ImageRedactionReport(doc_type="unknown", doc_confidence="low")
    
    # Hinglish: Face detection early chalate hain taaki fallback photo check ke liye use ho sake
    faces = []
    if policy.redact_faces:
        faces = detect_faces(image)

    # ---- Step 1: OCR ----
    words = run_ocr(image)

    # ---- Step 2: generic text-PII detection on OCR text ----
    if words:
        logical_text, spans = _build_ocr_logical_text(words)
        replacer = ConsistentReplacer()  # Hinglish: replacement value use nahi hota images mein, sirf span chahiye
        detections = detect_pii_in_text(logical_text, replacer, policy)
        text_pii_boxes = set()
        for d in detections:
            for box in _boxes_overlapping_span(spans, d.start, d.end):
                text_pii_boxes.add(box)
        boxes_to_mask.extend(text_pii_boxes)
        report.text_pii_boxes_masked = len(text_pii_boxes)

    # ---- Step 3: ID document classification ----
    if words and policy.redact_id_documents:
        classification = classify_id_document(words)
        report.doc_type = classification.doc_type
        report.doc_confidence = classification.confidence

        if classification.doc_type != "unknown":
            # ---- Step 4: label-based name/signature heuristic & fallbacks ----
            id_field_boxes = find_id_field_boxes(words, classification.doc_type)
            
            has_signature_label = any(f_name == "signature" for f_name, _ in id_field_boxes)
            
            photo_regions_count = 0
            signatures_count = 0
            
            for field_name, box in id_field_boxes:
                if field_name == "signature":
                    if not policy.redact_signatures_on_id:
                        continue
                    boxes_to_mask.append(box)
                    signatures_count += 1
                elif field_name == "fallback_photo":
                    if policy.redact_faces:
                        boxes_to_mask.append(box)
                        photo_regions_count += 1
                elif field_name == "fallback_signature":
                    if not has_signature_label and policy.redact_signatures_on_id:
                        boxes_to_mask.append(box)
                        signatures_count += 1
                elif field_name == "fallback_qr":
                    if policy.redact_qr_on_id:
                        boxes_to_mask.append(box)
                        report.qr_codes_masked += 1
                else:
                    boxes_to_mask.append(box)
            report.id_fields_masked = len([f for f, _ in id_field_boxes if not f.startswith("fallback_")])
            report.photo_regions_masked = photo_regions_count
            report.signatures_masked = signatures_count

            # Hinglish: DOC-LEVEL CONTEXT OVERRIDE.
            # Generic text detector ko Aadhaar/Passport number redact
            # karne ke liye LOCAL context keyword chahiye (CONTEXT_WINDOW
            # ke andar) - lekin ID card ke andar keyword (jaise "Unique
            # Identification Authority") aur actual number spatially door
            # ho sakte hain (OCR reading-order mein beech mein address
            # jaisa text aa jaata hai), isliye local-window check fail ho
            # jaata hai. Yahan hum WHOLE-IMAGE classification (jo already
            # confidently "aadhaar"/"passport" bata chuka hai) ko hi
            # sufficient context maan kar, poore OCR text mein number
            # pattern dhoondh kar mask karte hain - local context zaroorat
            # nahi.
            if classification.doc_type == "aadhaar" and policy.redact_aadhaar:
                for m in AADHAAR_RE.finditer(logical_text):
                    boxes_to_mask.extend(_boxes_overlapping_span(spans, m.start(), m.end()))
            if classification.doc_type == "passport" and policy.redact_passport:
                for m in PASSPORT_RE.finditer(logical_text):
                    boxes_to_mask.extend(_boxes_overlapping_span(spans, m.start(), m.end()))

    # ---- Step 5: face detection (mask appending) ----
    if policy.redact_faces:
        for f in faces:
            boxes_to_mask.append(f.box)
        report.faces_masked = len(faces)

    # ---- Step 6: QR code detection (only mask if inside a recognized ID doc) ----
    if policy.redact_qr_on_id and report.doc_type != "unknown":
        qrs = detect_qr_codes(image)
        for q in qrs:
            boxes_to_mask.append(q.bounding_box)
        report.qr_codes_masked = len(qrs)

    if not boxes_to_mask:
        # Hinglish: kuch bhi redact-worthy nahi mila, original bytes hi return karo
        return image_bytes, report

    # ---- Step 7: actual pixel modification (normalized and merged) ----
    h, w = image.shape[:2]
    normalized_boxes = normalize_and_merge_boxes(boxes_to_mask, w, h)
    for (x1, y1, x2, y2) in normalized_boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)

    report.modified = True

    # Hinglish: original format preserve karte hain (PNG/JPEG) taaki
    # document ka structure/size expectations na tootein.
    ext = format_ext if format_ext in (".png", ".jpg", ".jpeg", ".bmp") else ".png"
    if ext == ".jpeg":
        ext = ".jpg"
    success, encoded = cv2.imencode(ext, image)
    if not success:
        return image_bytes, ImageRedactionReport(doc_type=report.doc_type, doc_confidence=report.doc_confidence)
    return encoded.tobytes(), report
