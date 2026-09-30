import cv2
import numpy as np
import pytest

from app.vision import decode_image, normalize_plate_text, preprocess_image

def test_normalize_plate_text_removes_punctuation():
    assert normalize_plate_text(" ap-09 ab 1234 ") == "AP09AB1234"

def test_preprocess_image_preserves_dimensions():
    image = np.zeros((80, 160, 3), dtype=np.uint8)
    processed = preprocess_image(image)
    assert processed.shape == (80, 160)

def test_decode_image_rejects_invalid_bytes():
    with pytest.raises(ValueError):
        decode_image(b"not-an-image")

def test_decode_image_reads_encoded_image():
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    success, encoded = cv2.imencode(".png", image)
    assert success
    decoded = decode_image(encoded.tobytes())
    assert decoded.shape == (20, 30, 3)
