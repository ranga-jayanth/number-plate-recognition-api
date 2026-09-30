import re
from typing import Tuple

import cv2
import numpy as np
import pytesseract

def decode_image(image_bytes: bytes) -> np.ndarray:
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The uploaded file is not a readable image.")
    return image

def preprocess_image(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    thresholded = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)

def normalize_plate_text(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())

def recognize_plate(image_bytes: bytes) -> Tuple[str, int, int]:
    image = decode_image(image_bytes)
    processed = preprocess_image(image)
    raw_text = pytesseract.image_to_string(processed, config="--psm 7")
    return normalize_plate_text(raw_text), int(image.shape[1]), int(image.shape[0])
