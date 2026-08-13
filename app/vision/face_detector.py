"""
Hinglish: OpenCV ke built-in Haar Cascade classifier se face detection.
Ye deep-learning-based detector jaisa accurate nahi hai lekin:
  - koi extra model download nahi chahiye (already OpenCV ke saath aata hai)
  - fast hai, CPU par bhi turant chalta hai
  - explainable hai (assignment ka core principle: simple + understandable)

LIMITATION (honestly documented, config.py FACE_MIN_SIZE_PX se related):
Haar cascades tiny faces, extreme angles (profile/tilted), heavy blur,
ya unusual lighting mein miss kar sakte hain. Production system mein
DNN-based face detector (jaise OpenCV's res10 SSD model ya MTCNN) zyada
robust hota - yahan simplicity ke liye Haar cascade use kiya.
"""
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from app.config import (
    FACE_DETECTION_SCALE_FACTOR, FACE_DETECTION_MIN_NEIGHBORS, FACE_MIN_SIZE_PX,
)

_face_cascade = None


@dataclass
class FaceBox:
    x: int
    y: int
    width: int
    height: int

    @property
    def box(self):
        return (self.x, self.y, self.x + self.width, self.y + self.height)


def _get_cascade():
    global _face_cascade
    if _face_cascade is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(cascade_path)
    return _face_cascade


def detect_faces(image_bgr: np.ndarray) -> List[FaceBox]:
    """
    Hinglish: Image mein saare faces detect karta hai (ek ho, multiple ho,
    ya ID-card portrait ho - sab handle hota hai kyunki Haar cascade
    multi-scale sliding window use karta hai).

    Input: BGR image
    Output: list of FaceBox (pixel coordinates)
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # Hinglish: contrast improve karta hai, uneven lighting mein help karta hai
    cascade = _get_cascade()
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=FACE_DETECTION_SCALE_FACTOR,
        minNeighbors=FACE_DETECTION_MIN_NEIGHBORS,
        minSize=(FACE_MIN_SIZE_PX, FACE_MIN_SIZE_PX),
    )
    return [FaceBox(x=int(x), y=int(y), width=int(w), height=int(h)) for (x, y, w, h) in faces]
