# Number Plate Recognition API

A portfolio computer-vision API that accepts an image and extracts a likely vehicle number plate string using OpenCV preprocessing and Tesseract OCR.

> This is a learning prototype. It does not replace a production ANPR system, a trained plate detector, or legal/compliance review.

## Features

- Upload an image through a FastAPI endpoint
- Decode and validate the image with OpenCV
- Apply grayscale, denoising, thresholding, and morphological preprocessing
- Extract a normalized alphanumeric result with Tesseract OCR
- Return image metadata and a simple health endpoint
- Run locally or with Docker

## Tech stack

- Python 3.12+
- FastAPI and Uvicorn
- OpenCV
- Tesseract OCR through pytesseract
- NumPy

## Setup

The Python package is not enough for OCR; the Tesseract executable must also be installed.

### Linux

```bash
sudo apt-get update && sudo apt-get install -y tesseract-ocr
```

### Python environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive API documentation.

## Docker

```bash
docker build -t number-plate-api .
docker run --rm -p 8000:8000 number-plate-api
```

## API usage

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Recognize an image:

```bash
curl -X POST http://127.0.0.1:8000/recognize -F "file=@car-image.jpg"
```

Example response:

```json
{
  "filename": "car-image.jpg",
  "plate_text": "AP09AB1234",
  "image_width": 1280,
  "image_height": 720
}
```

## Accuracy limitations

The baseline reads the full image after preprocessing. Results depend on lighting, image angle, plate visibility, font, and image resolution. A stronger production version should detect the plate region first, use a trained detector, validate country-specific formats, return OCR confidence, and evaluate against a labeled dataset.

## Attribution and privacy

This is an original portfolio implementation based on the official documentation for FastAPI, OpenCV, and pytesseract. No third-party repository is presented as original work.

- https://fastapi.tiangolo.com/
- https://docs.opencv.org/
- https://pypi.org/project/pytesseract/

Only process vehicle images you are authorized to use. Avoid committing personal images, real plate data, or credentials to the repository.

## Roadmap

- Add plate-region detection before OCR
- Add country-specific format validation
- Add OCR confidence and preprocessing comparisons
- Add a small permission-safe evaluation dataset
- Add authentication and request-size/rate limits