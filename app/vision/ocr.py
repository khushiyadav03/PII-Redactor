"""
Hinglish: Ye module image ke andar ka TEXT nikalta hai (OCR) aur har word
ka bounding box (pixel coordinates) bhi deta hai. Bounding box zaroori hai
kyunki text detect karne ke baad hume us specific PIXEL REGION ko black
box se mask karna hai - sirf text janna kaafi nahi.

Hum minimal OpenCV preprocessing karte hain (grayscale + upscale + adaptive
threshold) - sirf wahi steps jo demonstrably OCR accuracy improve karte
hain chhoti/low-res ID card images par. Bahut zyada preprocessing pipeline
jaan-boojh kar avoid kiya hai (assignment ka core-engineering-principle:
simple rakho).
"""
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
import pytesseract
import os

# Hinglish: Windows environment par Tesseract default path check karte hain,
# taaki agar user ne standard location par install kiya ho to path configure
# automatically ho jaye (TesseractNotFoundError se bachne ke liye).
_DEFAULT_TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.name == "nt" and os.path.exists(_DEFAULT_TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = _DEFAULT_TESSERACT_PATH

from app.config import OCR_MIN_CONFIDENCE


@dataclass
class OcrWord:
    text: str
    left: int
    top: int
    width: int
    height: int
    confidence: float

    @property
    def box(self):
        return (self.left, self.top, self.left + self.width, self.top + self.height)


def _preprocess_for_ocr(image_bgr: np.ndarray) -> np.ndarray:
    """
    Hinglish: Grayscale + upscale (agar image chhoti hai) + denoise.
    Ye specifically un ID-card-jaisi images ke liye help karta hai jinke
    scan/photo mein resolution kam ya lighting uneven hoti hai.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    # Hinglish: chhoti images ko upscale karte hain - Tesseract choti text
    # par better perform karta hai jab image bada ho.
    if max(h, w) < 1000:
        scale = 1000 / max(h, w)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    return gray


def run_ocr(image_bgr: np.ndarray, preprocess: bool = True) -> List[OcrWord]:
    """
    Hinglish: Image par OCR chalakar per-word bounding boxes + confidence
    return karta hai. Agar preprocessing use hui hai to bounding boxes ko
    ORIGINAL image ke coordinate-space mein wapas scale karte hain (taaki
    caller original image par hi masking kar sake).

    Input: BGR image (OpenCV format), preprocess flag
    Output: list of OcrWord (confidence 0-100 scale)
    """
    orig_h, orig_w = image_bgr.shape[:2]
    if preprocess:
        processed = _preprocess_for_ocr(image_bgr)
        proc_h, proc_w = processed.shape[:2]
        scale_x = orig_w / proc_w
        scale_y = orig_h / proc_h
    else:
        processed = image_bgr
        scale_x = scale_y = 1.0

    data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT)

    words = []
    n = len(data["text"])
    for i in range(n):
        text = data["text"][i].strip()
        if not text:
            continue
        try:
            conf = float(data["conf"][i])
        except (ValueError, TypeError):
            conf = -1.0
        if conf < OCR_MIN_CONFIDENCE:
            continue
        left = int(data["left"][i] * scale_x)
        top = int(data["top"][i] * scale_y)
        width = int(data["width"][i] * scale_x)
        height = int(data["height"][i] * scale_y)
        words.append(OcrWord(text=text, left=left, top=top, width=width, height=height, confidence=conf))
    return words


def group_words_into_line_text(words: List[OcrWord]) -> str:
    """Hinglish: Debugging/context-matching ke liye words ko ek single string mein jodta hai."""
    return " ".join(w.text for w in words)
