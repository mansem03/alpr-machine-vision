# ============================================================
# MALAYSIAN ALPR SYSTEM (BATCH IMAGE PROCESSING)
# YOLOv11 + EasyOCR + Robust Preprocessing
# SAVE ALL OUTPUT IMAGES INTO ONE FOLDER
# ============================================================

import os
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import matplotlib.pyplot as plt
import re
import gc

# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_PATH = "last.pt"                 # Trained YOLO model
IMAGE_FOLDER = "random_images_02"       # Folder containing images
OUTPUT_FOLDER = "all_outputs"           # Save all outputs here

CONFIDENCE = 0.2                        # Detection threshold
UPSCALE_FACTOR = 2.5                    # OCR enhancement factor

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================================
# 2. LOAD MODEL & OCR (RUN ONCE)
# ============================================================

print("[INFO] Loading YOLO model...")
plate_detector = YOLO(MODEL_PATH)

print("[INFO] Initializing EasyOCR...")
reader = easyocr.Reader(['en'], gpu=False)

# ============================================================
# 3. ROBUST PRE-PROCESSING FUNCTION
# ============================================================

def preprocess_for_robustness(plate_img):
    """Enhances OCR robustness for:
       - Distant plates
       - Angled plates
       - Low contrast / night images"""
    
    if plate_img is None or plate_img.size == 0:
        return None

    # Add padding
    padded = cv2.copyMakeBorder(
        plate_img, 15, 15, 15, 15,
        cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )

    # Grayscale
    gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)

    # Upscale
    upscaled = cv2.resize(
        gray, None,
        fx=UPSCALE_FACTOR,
        fy=UPSCALE_FACTOR,
        interpolation=cv2.INTER_LANCZOS4
    )

    # CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(upscaled)

    # Denoise
    denoised = cv2.bilateralFilter(enhanced, 11, 17, 17)

    return denoised

# ============================================================
# 4. ALPR PIPELINE FUNCTION
# ============================================================

def run_alpr_system(img_path):
    gc.collect()  # Clean memory

    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] Cannot load image: {img_path}")
        return None, []

    detected_plates = []

    results = plate_detector.predict(
        img, conf=CONFIDENCE, save=False, verbose=False
    )

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            crop = img[y1:y2, x1:x2]
            processed = preprocess_for_robustness(crop)

            if processed is None:
                continue

            ocr_results = reader.readtext(processed)
            raw_text = "".join([res[1] for res in ocr_results])
            clean_text = re.sub(r'[^A-Z0-9]', '', raw_text.upper())

            if clean_text:
                detected_plates.append(clean_text)
            else:
                detected_plates.append("DETECTED_NO_OCR")

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2),
                          (0, 255, 0), 3)

            # Draw text above box
            cv2.putText(
                img, clean_text if clean_text else "NO_OCR",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1, (0, 255, 0), 3
            )

    return img, detected_plates

# ============================================================
# 5. BATCH IMAGE PROCESSING (SAVE ALL OUTPUT TO ONE FOLDER)
# ============================================================

image_extensions = (".jpg", ".jpeg", ".png")

print("\n[INFO] Starting batch ALPR processing...\n")

for img_name in os.listdir(IMAGE_FOLDER):
    if img_name.lower().endswith(image_extensions):

        img_path = os.path.join(IMAGE_FOLDER, img_name)
        print(f"[PROCESSING] {img_name}")

        output_img, plates = run_alpr_system(img_path)

        if output_img is not None:
            save_path = os.path.join(OUTPUT_FOLDER, img_name)
            cv2.imwrite(save_path, output_img)

            if plates:
                print("   → Plates:", plates)
            else:
                print("   → No plates detected")

print("\n[INFO] All images saved into:", OUTPUT_FOLDER)
print("[INFO] Batch processing completed.")
