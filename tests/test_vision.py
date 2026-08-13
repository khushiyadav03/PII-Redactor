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
    # Configure tesseract cmd just in case
    import app.vision.ocr as ocr_mod
    ocr_mod.pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    
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
