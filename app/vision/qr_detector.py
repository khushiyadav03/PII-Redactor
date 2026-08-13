"""
Hinglish: OpenCV ka built-in QRCodeDetector use karte hain - koi extra
dependency (jaise pyzbar/zbar system library) nahi chahiye.

POLICY: Har QR code PII nahi hota (jaise ek website ka generic promotional
QR code). Isliye hum QR ko sirf TAB mask karte hain jab wo ek confidently-
classified ID document (PAN/Aadhaar/Passport) ke andar mila ho - id_detector.py
se decision aata hai. Ye policy README mein explicitly explain ki gayi hai.
"""
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class QrBox:
    points: List[Tuple[int, int]]  # 4 corner points of the QR code

    @property
    def bounding_box(self):
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))


def detect_qr_codes(image_bgr: np.ndarray) -> List[QrBox]:
    """
    Hinglish: Image mein saare QR codes dhoondta hai. detectAndDecodeMulti ke
    sath-sath detectMulti fallback bhi use karte hain taaki non-decodable
    blurry/placeholder QR codes ko bhi hum detect aur redact kar sakein.

    Input: BGR image
    Output: list of QrBox
    """
    detector = cv2.QRCodeDetector()
    boxes = []
    
    # Hinglish: Pehle detectMulti try karte hain, isse visual boundary corners mil jaate hain
    # bina value decode kiye (placeholder/low-res QR codes isse mask ho jaate hain).
    try:
        ok, points = detector.detectMulti(image_bgr)
        if ok and points is not None:
            for pts in points:
                if len(pts) >= 4:
                    boxes.append(QrBox(points=[(float(p[0]), float(p[1])) for p in pts]))
            return boxes
    except cv2.error:
        pass

    # Hinglish: Fallback to detectAndDecodeMulti
    try:
        ok, decoded_info, points, _ = detector.detectAndDecodeMulti(image_bgr)
        if ok and points is not None:
            for pts in points:
                boxes.append(QrBox(points=[(float(p[0]), float(p[1])) for p in pts]))
    except cv2.error:
        pass
    return boxes
