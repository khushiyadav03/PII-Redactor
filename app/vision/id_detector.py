"""
Hinglish: Ye module OCR se mile text ke aadhar par decide karta hai ki
image kisi PEHCHAN-DOCUMENT (ID) ka scan hai ya nahi, aur agar hai to
kaunsa type (PAN / Aadhaar / Passport). Ye ek simple KEYWORD + PATTERN
based classifier hai - koi heavy deep-learning classifier nahi (assignment
ka core-engineering-principle: simple, explainable rakho).

Jab document confidently identify ho jaata hai, to hum ek
DOCUMENT-SPECIFIC redaction policy apply karte hain (config.py section 13
jaisa: name, DOB, ID-number, face, QR, signature sab mask karo).
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

from app.config import (
    PAN_CONTEXT_KEYWORDS, AADHAAR_CONTEXT_KEYWORDS, PASSPORT_CONTEXT_KEYWORDS,
)
from app.text.patterns import PAN_RE, AADHAAR_RE, PASSPORT_RE
from app.vision.ocr import OcrWord
import re


@dataclass
class IdDocumentClassification:
    doc_type: str          # "pan" | "aadhaar" | "passport" | "unknown"
    confidence: str        # "high" | "low" - explainable, not a fake float score
    matched_keywords: List[str]


def _levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def _has_fuzzy_substring(text: str, target: str, max_dist: int) -> bool:
    target_len = len(target)
    text_len = len(text)
    if text_len < target_len:
        return _levenshtein_distance(text, target) <= max_dist
    
    # Hinglish: Sliding window approach to search for fuzzy keyword matches in noisy OCR
    for i in range(text_len - target_len + 1):
        sub = text[i:i+target_len]
        if _levenshtein_distance(sub, target) <= max_dist:
            return True
    return False


def _normalize_text(text: str) -> str:
    # Hinglish: Lowercase, collapse whitespace, and strip punctuation for robust fuzzy matches
    t = text.lower()
    t = "".join(c if (c.isalnum() or c.isspace()) else " " for c in t)
    t = " ".join(t.split())
    return t


def classify_id_document(ocr_words: List[OcrWord]) -> IdDocumentClassification:
    """
    Hinglish: Saare OCR words ko normalize karke fuzzy context keywords matching
    chalate hain taaki noisy Tesseract OCR outputs (e.g. "Urique" for "Unique")
    par classification fail na ho.
    """
    raw_text = " ".join(w.text for w in ocr_words)
    normalized_text = _normalize_text(raw_text)

    # Aadhaar keywords matched
    aadhaar_kws = []
    if any(x in normalized_text for x in ["aadhaar", "aadhar", "adhar"]) or _has_fuzzy_substring(normalized_text, "aadhaar", 1):
        aadhaar_kws.append("aadhaar")
    if "uidai" in normalized_text or _has_fuzzy_substring(normalized_text, "uidai", 1):
        aadhaar_kws.append("uidai")
    if _has_fuzzy_substring(normalized_text, "unique identification authority", 4):
        aadhaar_kws.append("unique identification authority")
    if _has_fuzzy_substring(normalized_text, "government of india", 3) or _has_fuzzy_substring(normalized_text, "govt of india", 2):
        aadhaar_kws.append("government of india")

    # PAN keywords matched
    pan_kws = []
    if _has_fuzzy_substring(normalized_text, "permanent account number", 3):
        pan_kws.append("permanent account number")
    if _has_fuzzy_substring(normalized_text, "income tax department", 3):
        pan_kws.append("income tax department")
    if "pan card" in normalized_text or _has_fuzzy_substring(normalized_text, "pan card", 1):
        pan_kws.append("pan card")

    # Passport keywords matched
    passport_kws = []
    if "passport" in normalized_text or _has_fuzzy_substring(normalized_text, "passport", 1):
        passport_kws.append("passport")
    if _has_fuzzy_substring(normalized_text, "republic of india", 3):
        passport_kws.append("republic of india")
    if "type p" in normalized_text:
        passport_kws.append("type p")

    matches = {
        "pan": pan_kws,
        "aadhaar": aadhaar_kws,
        "passport": passport_kws,
    }

    # Hinglish: Numeric pattern check for classification confidence
    has_pan_number = bool(PAN_RE.search(raw_text))
    has_aadhaar_number = bool(AADHAAR_RE.search(raw_text))
    has_passport_number = bool(PASSPORT_RE.search(raw_text))

    best_type = "unknown"
    best_score = 0
    best_keywords: List[str] = []
    for doc_type, kw_list, has_number in [
        ("pan", matches["pan"], has_pan_number),
        ("aadhaar", matches["aadhaar"], has_aadhaar_number),
        ("passport", matches["passport"], has_passport_number),
    ]:
        score = len(kw_list) + (1 if has_number else 0)
        # Hinglish: Classification tabhi valid hai jab:
        # 1. Pattern match ho AUR at least 1 context keyword ho
        # 2. Ya fir pattern na ho par at least 2 distinct keywords hon
        has_sufficient_evidence = (
            (has_number and len(kw_list) >= 1) or
            (len(kw_list) >= 2)
        )
        if has_sufficient_evidence and score > best_score:
            best_score = score
            best_type = doc_type
            best_keywords = kw_list

    if best_type == "unknown":
        return IdDocumentClassification(doc_type="unknown", confidence="low", matched_keywords=[])

    # Hinglish: High confidence tabhi jab dono factors (keywords aur patterns) satisfied hon.
    has_number_matched = (
        (best_type == "pan" and has_pan_number) or
        (best_type == "aadhaar" and has_aadhaar_number) or
        (best_type == "passport" and has_passport_number)
    )
    confidence = "high" if (len(best_keywords) >= 1 and has_number_matched) else "low"
    return IdDocumentClassification(doc_type=best_type, confidence=confidence, matched_keywords=best_keywords)


# Hinglish: Har document type ke liye field-level redaction policy
# (config.py section 13 se directly liya gaya).
ID_DOCUMENT_REDACTION_FIELDS = {
    "pan": ["name", "father's name", "date of birth", "pan_number", "signature", "photo"],
    "aadhaar": ["name", "date of birth", "aadhaar_number", "address", "photo", "qr_code"],
    "passport": ["name", "passport_number", "date of birth", "photo", "signature", "mrz"],
}


# ============================================================
# Hinglish: LABEL -> VALUE line heuristic for "name" fields.
#
# WHY: spaCy NER is unreliable on short, fragmented, ALL-CAPS OCR text
# (tested and confirmed - see README limitations). ID cards, however,
# have a very PREDICTABLE structure: a printed label ("Name", "Father's
# Name") sits on one line, and the actual value sits on the line
# immediately below it. This is exactly the "OCR + contextual keywords +
# structured patterns" strategy the assignment recommends (section 13)
# instead of a heavy ML classifier.
#
# LIMITATION (documented honestly): this heuristic assumes a label-above-
# value layout, which is common for Indian PAN/Aadhaar cards but not
# universal. OCR noise on the label itself (e.g. "Name" misread as
# "Wame") is handled with a small fuzzy-match set, but a badly OCR'd
# label could still be missed - the tool cannot see the raw pixels'
# semantic meaning, only what Tesseract reports.
# ============================================================

_NAME_LABEL_VARIANTS = {"name", "wame", "narne", "nam"}
_FATHER_LABEL_VARIANTS = {"father's", "fathers", "father"}
_SIGNATURE_LABEL_VARIANTS = {"signature", "sign"}
_LINE_Y_TOLERANCE = 12  # px - words within this y-range are treated as the same line


def _group_words_into_lines(words: List[OcrWord]) -> List[List[OcrWord]]:
    """Hinglish: OCR words ko unke vertical (y) position ke aadhar par lines mein group karta hai."""
    sorted_words = sorted(words, key=lambda w: w.top)
    lines: List[List[OcrWord]] = []
    for w in sorted_words:
        placed = False
        for line in lines:
            if abs(line[0].top - w.top) <= _LINE_Y_TOLERANCE:
                line.append(w)
                placed = True
                break
        if not placed:
            lines.append([w])
    for line in lines:
        line.sort(key=lambda w: w.left)
    lines.sort(key=lambda line: line[0].top)
    return lines


def _line_bbox(line: List[OcrWord]):
    lefts = [w.left for w in line]
    tops = [w.top for w in line]
    rights = [w.left + w.width for w in line]
    bottoms = [w.top + w.height for w in line]
    return (min(lefts), min(tops), max(rights), max(bottoms))


def find_id_field_boxes(words: List[OcrWord], doc_type: str = "unknown") -> List[Tuple[str, Tuple[int, int, int, int]]]:
    """
    Hinglish: ID card ke "name", "father's name", aur "signature" fields
    ke liye bounding boxes dhoondta hai. Same-line layouts (Name: Value)
    aur next-line layouts (Name\n Value) dono ko support karta hai.
    Aadhaar card ke address aur Passport ke MRZ lines ko bhi identify karta hai.

    Output: list of (field_name, (x1,y1,x2,y2))
    """
    lines = _group_words_into_lines(words)
    boxes: List[Tuple[str, Tuple[int, int, int, int]]] = []

    # Hinglish: Step 1 - Label-value alignment and specific fields detection
    for i, line in enumerate(lines):
        line_words_lower = [w.text.strip(":").lower() for w in line]

        is_father_line = any(t in _FATHER_LABEL_VARIANTS for t in line_words_lower)
        is_name_line = (not is_father_line) and any(t in _NAME_LABEL_VARIANTS for t in line_words_lower)
        is_signature_line = any(t in _SIGNATURE_LABEL_VARIANTS for t in line_words_lower)

        # Hinglish: Same-line layout check (Label aur value ek hi line mein hain)
        if is_name_line or is_father_line:
            field = "father's name" if is_father_line else "name"
            label_idx = -1
            variants = _FATHER_LABEL_VARIANTS if is_father_line else _NAME_LABEL_VARIANTS
            for idx, w in enumerate(line):
                if w.text.strip(":").lower() in variants:
                    label_idx = idx
                    break
            
            right_words = []
            if label_idx != -1:
                label_word = line[label_idx]
                for w in line[label_idx + 1:]:
                    if w.text.strip(":/\\- ") == "":
                        continue
                    if w.left > label_word.left:
                        right_words.append(w)
            
            if right_words:
                boxes.append((field, _line_bbox(right_words)))
            elif i + 1 < len(lines):
                value_line = lines[i + 1]
                boxes.append((field, _line_bbox(value_line)))

        if is_signature_line and i - 1 >= 0:
            above_line = lines[i - 1]
            boxes.append(("signature", _line_bbox(above_line)))

        # Hinglish: Aadhaar address detection (Back of Aadhaar has Address: or पता:)
        if doc_type == "aadhaar":
            is_address_label = any(t in ("address", "address:", "addressl", "addre", "पता", "पता:") for t in line_words_lower)
            if is_address_label:
                # Label word find karte hain coordinate mapping ke liye
                label_word = None
                for w in line:
                    if w.text.strip(":").lower() in ("address", "address:", "addressl", "addre", "पता", "पता:"):
                        label_word = w
                        break
                if label_word:
                    # Sirf address region ke physical words collect karte hain over-masking se bachne ke liye
                    address_words = []
                    for w in words:
                        if (w.top >= label_word.top - 15 and
                            w.top <= label_word.top + 130 and
                            w.left >= label_word.left - 30):
                            address_words.append(w)
                    if address_words:
                        boxes.append(("address", _line_bbox(address_words)))

        # Hinglish: Passport MRZ (Machine Readable Zone) lines detection
        if doc_type == "passport":
            is_mrz_line = (
                sum(w.text.count("<") for w in line) >= 5 or 
                any(w.text.upper().startswith("P<") or w.text.upper().startswith("<P") for w in line) or
                (any("<" in w.text for w in line) and sum(len(w.text) for w in line) > 20)
            )
            if is_mrz_line:
                boxes.append(("mrz", _line_bbox(line)))

    # Hinglish: Step 2 - Fallback regions agar visual elements missing ho
    if words:
        lefts = [w.left for w in words]
        tops = [w.top for w in words]
        rights = [w.left + w.width for w in words]
        bottoms = [w.top + w.height for w in words]
        
        card_min_x, card_min_y = min(lefts), min(tops)
        card_max_x, card_max_y = max(rights), max(bottoms)
        card_w = card_max_x - card_min_x
        card_h = card_max_y - card_min_y
        
        # Hinglish: front side height calculate karte hain double-sided stacked layout ke liye
        front_h = card_h // 2 if card_h > 600 else card_h

        if doc_type == "pan":
            photo_box = (
                card_min_x + int(card_w * 0.02),
                card_min_y + int(front_h * 0.12),
                card_min_x + int(card_w * 0.28),
                card_min_y + int(front_h * 0.48)
            )
            sig_box = (
                card_min_x + int(card_w * 0.40),
                card_min_y + int(front_h * 0.70),
                card_min_x + int(card_w * 0.85),
                card_min_y + int(front_h * 0.95)
            )
            boxes.append(("fallback_photo", photo_box))
            boxes.append(("fallback_signature", sig_box))
            
        elif doc_type == "aadhaar":
            has_address = any("address" in w.text.lower() or "पता" in w.text.lower() for w in words)
            if not has_address or card_h > 600:
                photo_box = (
                    card_min_x + int(card_w * 0.05),
                    card_min_y + int(front_h * 0.15),
                    card_min_x + int(card_w * 0.24),
                    card_min_y + int(front_h * 0.85)
                )
                boxes.append(("fallback_photo", photo_box))
            
            # Hinglish: Aadhaar front-page QR code layout-fallback (top card, upper-right area)
            qr_box = (
                card_min_x + int(card_w * 0.55),
                card_min_y + int(front_h * 0.15),
                card_min_x + int(card_w * 0.90),
                card_min_y + int(front_h * 0.80)
            )
            boxes.append(("fallback_qr", qr_box))
                
        elif doc_type == "passport":
            photo_box = (
                card_min_x + int(card_w * 0.02),
                card_min_y + int(front_h * 0.20),
                card_min_x + int(card_w * 0.45),
                card_min_y + int(front_h * 0.80)
            )
            sig_box = (
                card_min_x + int(card_w * 0.45),
                card_min_y + int(front_h * 0.65),
                card_min_x + int(card_w * 0.85),
                card_min_y + int(front_h * 0.90)
            )
            boxes.append(("fallback_photo", photo_box))
            boxes.append(("fallback_signature", sig_box))

    return boxes
