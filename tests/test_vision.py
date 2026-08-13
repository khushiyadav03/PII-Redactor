import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2

from app.config import RedactionPolicy
from app.vision.face_detector import detect_faces
from app.vision.id_detector import classify_id_document
from app.vision.image_redactor import redact_image_bytes
from app.vision.ocr import run_ocr
from app.vision.qr_detector import detect_qr_codes

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
PAN_IMAGE = os.path.join(SAMPLES_DIR, "pan_card_sample.png")
AADHAAR_IMAGE = os.path.join(SAMPLES_DIR, "aadhaar_card_sample.png")


def _skip_if_missing(path):
    import pytest
    if not os.path.exists(path):
        pytest.skip(f"Sample image not found at {path} (run from project with extracted media)")


def test_ocr_extracts_pan_number_from_pan_card():
    _skip_if_missing(PAN_IMAGE)
    img = cv2.imread(PAN_IMAGE)
    words = run_ocr(img)
    all_text = " ".join(w.text for w in words)
    assert "NBWPS1951N" in all_text


def test_face_detected_on_pan_card():
    _skip_if_missing(PAN_IMAGE)
    img = cv2.imread(PAN_IMAGE)
    faces = detect_faces(img)
    assert len(faces) >= 1


def test_id_classification_identifies_pan_card():
    _skip_if_missing(PAN_IMAGE)
    img = cv2.imread(PAN_IMAGE)
    words = run_ocr(img)
    result = classify_id_document(words)
    assert result.doc_type == "pan"
    assert result.confidence == "high"


def test_image_redaction_masks_face_and_id_number():
    _skip_if_missing(PAN_IMAGE)
    with open(PAN_IMAGE, "rb") as f:
        data = f.read()
    new_bytes, report = redact_image_bytes(data, RedactionPolicy())
    assert report.modified is True
    assert report.faces_masked >= 1
    assert report.doc_type == "pan"
    # Hinglish: redacted image mein PAN number ab OCR se readable nahi hona chahiye
    import numpy as np
    arr = np.frombuffer(new_bytes, dtype=np.uint8)
    redacted_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    words_after = run_ocr(redacted_img)
    text_after = " ".join(w.text for w in words_after)
    assert "NBWPS1951N" not in text_after


def test_corrupted_image_bytes_does_not_crash():
    """Hinglish: FAILURE-HANDLING GUARD - invalid image bytes crash nahi karni chahiye."""
    garbage = b"not a real image file"
    new_bytes, report = redact_image_bytes(garbage, RedactionPolicy())
    assert report.modified is False
    assert new_bytes == garbage


def test_aadhaar_card_classified_and_number_masked():
    _skip_if_missing(AADHAAR_IMAGE)
    with open(AADHAAR_IMAGE, "rb") as f:
        data = f.read()
    new_bytes, report = redact_image_bytes(data, RedactionPolicy(), ".png")
    assert report.doc_type == "aadhaar"
    assert report.modified is True
    import numpy as np
    arr = np.frombuffer(new_bytes, dtype=np.uint8)
    redacted_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    words_after = run_ocr(redacted_img)
    text_after = " ".join(w.text for w in words_after)
    assert "2943 6593 3461".replace(" ", "") not in text_after.replace(" ", "")
    # Aadhaar Address redaction check
    assert "saray" not in text_after.lower()
    assert "katrauli" not in text_after.lower()


def test_find_id_field_boxes_same_line():
    from app.vision.ocr import OcrWord
    from app.vision.id_detector import find_id_field_boxes
    words = [
        OcrWord(text="Name:", left=50, top=100, width=40, height=20, confidence=90.0),
        OcrWord(text="Vishal", left=100, top=100, width=50, height=20, confidence=90.0),
        OcrWord(text="Singh", left=160, top=100, width=50, height=20, confidence=90.0),
    ]
    boxes = find_id_field_boxes(words, "pan")
    name_boxes = [box for name, box in boxes if name == "name"]
    assert len(name_boxes) >= 1
    x1, y1, x2, y2 = name_boxes[0]
    assert x1 >= 100
    assert y1 == 100


def test_find_id_field_boxes_passport_mrz():
    from app.vision.ocr import OcrWord
    from app.vision.id_detector import find_id_field_boxes
    words = [
        OcrWord(text="P<IND<<<<<<<<<<<<<<<<<<<<<<<<<<", left=50, top=500, width=400, height=20, confidence=90.0)
    ]
    boxes = find_id_field_boxes(words, "passport")
    mrz_boxes = [box for name, box in boxes if name == "mrz"]
    assert len(mrz_boxes) == 1
    assert mrz_boxes[0] == (50, 500, 450, 520)


def test_find_id_field_boxes_photo_fallbacks():
    from app.vision.ocr import OcrWord
    from app.vision.id_detector import find_id_field_boxes
    words = [
        OcrWord(text="INCOME TAX DEPARTMENT", left=50, top=50, width=200, height=20, confidence=90.0),
        OcrWord(text="Permanent Account Number Card", left=50, top=100, width=300, height=20, confidence=90.0),
    ]
    boxes = find_id_field_boxes(words, "pan")
    fallback_photos = [box for name, box in boxes if name == "fallback_photo"]
    assert len(fallback_photos) == 1


# --- Phase 2: Visual Hardening and QR Validation Tests ---

QR_IMAGE = os.path.join(SAMPLES_DIR, "qr_code_sample.png")

def test_id_classification_negative_singleton():
    # Hinglish: Ek sentence jisme single keyword ho wo ID document classify nahi hona chahiye
    from app.vision.ocr import OcrWord
    words = [
        OcrWord(text="This", left=10, top=10, width=30, height=15, confidence=90.0),
        # "passport" keyword present only once, no passport pattern
        OcrWord(text="passport", left=50, top=10, width=50, height=15, confidence=90.0),
        OcrWord(text="guidelines", left=110, top=10, width=60, height=15, confidence=90.0),
    ]
    result = classify_id_document(words)
    assert result.doc_type == "unknown"


def test_qr_detection_and_masking():
    import numpy as np
    _skip_if_missing(QR_IMAGE)
    
    # 1. Base QR check
    img = cv2.imread(QR_IMAGE)
    qrs = detect_qr_codes(img)
    assert len(qrs) == 1
    
    # 2. Aadhaar text add kar ke composite image banate hain taaki visual pipeline confidently classify aur redact kare
    h, w = img.shape[:2]
    canvas = np.ones((h + 100, w, 3), dtype=np.uint8) * 255
    canvas[0:h, 0:w] = img
    cv2.putText(canvas, "Unique Identification Authority of India", (10, h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(canvas, "Aadhaar Number: 2943 6593 3461", (10, h + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    success, encoded = cv2.imencode(".png", canvas)
    assert success
    
    policy = RedactionPolicy()
    policy.redact_qr_on_id = True
    
    redacted_bytes, report = redact_image_bytes(encoded.tobytes(), policy, ".png")
    assert report.doc_type == "aadhaar"
    assert report.qr_codes_masked >= 1
    
    # 3. Redacted image mein QR decodable nahi hona chahiye
    arr = np.frombuffer(redacted_bytes, dtype=np.uint8)
    redacted_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    detector = cv2.QRCodeDetector()
    decoded, _, _ = detector.detectAndDecode(redacted_img)
    assert decoded == ""


def test_normalize_and_merge_boxes():
    from app.vision.image_redactor import normalize_and_merge_boxes
    raw_boxes = [
        (-10, -10, 50, 50),
        (15, 15, 80, 80),
        (200, 200, 250, 250),
        (210, 210, 240, 240),
        (500, 500, 400, 400),
    ]
    merged = normalize_and_merge_boxes(raw_boxes, 300, 300)
    assert len(merged) == 2
    assert (0, 0, 80, 80) in merged
    assert (200, 200, 250, 250) in merged


def test_passport_classification_and_mrz_redaction():
    from app.vision.ocr import OcrWord
    from app.vision.id_detector import classify_id_document, find_id_field_boxes
    
    words = [
        OcrWord(text="REPUBLIC", left=50, top=50, width=50, height=15, confidence=90.0),
        OcrWord(text="OF", left=110, top=50, width=20, height=15, confidence=90.0),
        OcrWord(text="INDIA", left=140, top=50, width=40, height=15, confidence=90.0),
        OcrWord(text="PASSPORT", left=50, top=100, width=80, height=15, confidence=90.0),
        OcrWord(text="P<INDSHARMA<<RAHUL<<<<<<<<<<<<<<<<<<<<<<<<<<", left=50, top=800, width=400, height=15, confidence=90.0),
        OcrWord(text="Z1234567<5IND9001011M2512312<<<<<<<<<<<<<<<", left=50, top=830, width=400, height=15, confidence=90.0),
    ]
    
    res = classify_id_document(words)
    assert res.doc_type == "passport"
    
    fields = find_id_field_boxes(words, "passport")
    mrz_fields = [box for name, box in fields if name == "mrz"]
    assert len(mrz_fields) == 2


def test_pan_photo_fallback_regression():
    from app.vision.ocr import OcrWord
    from app.vision.id_detector import find_id_field_boxes
    
    words = [
        OcrWord(text="INCOME TAX DEPARTMENT", left=50, top=50, width=200, height=20, confidence=90.0),
        OcrWord(text="Permanent Account Number Card", left=50, top=100, width=300, height=20, confidence=90.0),
    ]
    
    fields = find_id_field_boxes(words, "pan")
    fallbacks = [box for name, box in fields if name == "fallback_photo"]
    assert len(fallbacks) == 1
    x1, y1, x2, y2 = fallbacks[0]
    # Hinglish: Verify portrait is on the left side (within 45% card width offset)
    assert x1 >= 50
    assert x2 <= 50 + int(300 * 0.45)


def test_id_classification_false_positives():
    from app.vision.ocr import OcrWord
    from app.vision.id_detector import classify_id_document
    
    # 1. Document containing "Aadhaar" once without Aadhaar pattern
    words1 = [
        OcrWord(text="Aadhaar", left=50, top=50, width=60, height=15, confidence=90.0),
        OcrWord(text="details", left=120, top=50, width=50, height=15, confidence=90.0),
    ]
    assert classify_id_document(words1).doc_type == "unknown"
    
    # 2. Random 12-digit number without context keywords
    words2 = [
        OcrWord(text="1234", left=50, top=50, width=30, height=15, confidence=90.0),
        OcrWord(text="5678", left=90, top=50, width=30, height=15, confidence=90.0),
        OcrWord(text="9012", left=130, top=50, width=30, height=15, confidence=90.0),
    ]
    assert classify_id_document(words2).doc_type == "unknown"
    
    # 3. PAN-like pattern in financial body text
    words3 = [
        OcrWord(text="Account", left=50, top=50, width=50, height=15, confidence=90.0),
        OcrWord(text="balance", left=110, top=50, width=50, height=15, confidence=90.0),
        OcrWord(text="is", left=170, top=50, width=15, height=15, confidence=90.0),
        OcrWord(text="NBWPS1951N", left=190, top=50, width=80, height=15, confidence=90.0),
    ]
    assert classify_id_document(words3).doc_type == "unknown"


def test_generic_qr_code_not_masked():
    _skip_if_missing(QR_IMAGE)
    with open(QR_IMAGE, "rb") as f:
        data = f.read()
    policy = RedactionPolicy()
    policy.redact_qr_on_id = True
    
    new_bytes, report = redact_image_bytes(data, policy, ".png")
    assert report.doc_type == "unknown"
    assert report.qr_codes_masked == 0
    assert report.modified is False


def test_signature_masking_verified():
    from app.vision.ocr import OcrWord
    from app.vision.id_detector import find_id_field_boxes
    
    words = [
        OcrWord(text="Holder's", left=50, top=50, width=50, height=15, confidence=90.0),
        OcrWord(text="Signature", left=110, top=50, width=60, height=15, confidence=90.0),
        OcrWord(text="John", left=50, top=25, width=40, height=15, confidence=90.0),
        OcrWord(text="Doe", left=100, top=25, width=40, height=15, confidence=90.0),
    ]
    fields = find_id_field_boxes(words, "pan")
    sigs = [box for name, box in fields if name == "signature"]
    assert len(sigs) == 1
    x1, y1, x2, y2 = sigs[0]
    assert y1 <= 25 <= y2
