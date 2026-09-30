import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pytesseract.pytesseract import TesseractNotFoundError

from .vision import recognize_plate

app = FastAPI(title="Number Plate Recognition API", version="1.0.0")
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 10 * 1024 * 1024))

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload an image file.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="The image exceeds the upload size limit.")

    try:
        plate_text, width, height = recognize_plate(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except TesseractNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Tesseract OCR is not installed on the server.") from exc

    return {
        "filename": Path(file.filename or "upload").name,
        "plate_text": plate_text,
        "image_width": width,
        "image_height": height,
    }
