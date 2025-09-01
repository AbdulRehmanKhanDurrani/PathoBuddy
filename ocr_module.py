# ocr_module.py

import cv2
import easyocr
import numpy as np

# Initialize OCR reader
reader = easyocr.Reader(['en'], gpu=False)  # GPU not needed

def preprocess_image(image_path):

   # Load and clean the image using OpenCV (grayscale + thresholding).

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError("Image not found at path: " + image_path)
     #Grayscaling the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
    return thresh

def extract_text(image_path):

    #OCR pipeline: preprocess → extract text with EasyOCR.

    clean_image = preprocess_image(image_path)
    result = reader.readtext(clean_image, detail=0)
    extracted_text = "\n".join(result)
    return extracted_text
