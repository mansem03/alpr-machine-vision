import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import matplotlib.pyplot as plt
import re
import os

# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_PATH = "last.pt"
IMAGE_FOLDER = "random_images_01"          # <-- read all images here
SAVE_FOLDER = "output_results"             # <-- results saved here

os.makedirs(SAVE_FOLDER, exist_ok=True)

print("[INFO] Initializing System...")
reader = easyocr.Reader(['en'], gpu=False)
plate_detector = YOLO(MODEL_PATH)

STATE_MAP = {
    'A': 'Perak', 'B': 'Selangor', 'C': 'Pahang', 'D': 'Kelantan',
    'F': 'Putrajaya', 'J': 'Johor', 'K': 'Kedah', 'M': 'Melaka',
    'N': 'Negeri Sembilan', 'P': 'Penang', 'R': 'Perlis', 'T': 'Terengganu',
    'V': 'Kuala Lumpur', 'W': 'Kuala Lumpur', 'S': 'Sabah', 'Q': 'Sarawak'
}

# ============================================================
# 2. GEOMETRY HELPERS
# ============================================================

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def get_state(text):
    if text and text[0] in STATE_MAP:
        return STATE_MAP[text[0]]
    return "Unknown/Special"

# ============================================================
# 3. PROCESS SINGLE IMAGE
# ============================================================

def process_single_image(img_path, save_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] Cannot read image: {img_path}")
        return

    display_img = img.copy()
    results = plate_detector.predict(img, conf=0.2, verbose=False)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            pad = 15
            crop = img[max(0, y1-pad):y2+pad, max(0, x1-pad):x2+pad]

            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            edged = cv2.Canny(cv2.GaussianBlur(gray_crop, (5,5), 0), 50, 200)
            contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            h_c, w_c = crop.shape[:2]
            pts = np.array([[0, 0], [w_c, 0], [w_c, h_c], [0, h_c]], dtype="float32")

            if contours:
                c = max(contours, key=cv2.contourArea)
                approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
                if len(approx) == 4:
                    pts = approx.reshape(4, 2)

            rect = order_points(pts)
            dst = np.array([[0,0], [400,0], [400,110], [0,110]], dtype="float32")
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(crop, M, (400, 110))

            warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            clahe_img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(warped_gray)
            _, binary_plate = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            ocr_res = reader.readtext(
                binary_plate, 
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                paragraph=False,
                contrast_ths=0.05,
                adjust_contrast=0.8,
                text_threshold=0.5
            )

            raw_text = "".join([res[1] for res in ocr_res]).replace(" ", "").upper()
            state_name = get_state(raw_text)

            # Save dashboard image
            fig = plt.figure(figsize=(15, 8))
            grid = plt.GridSpec(2, 2, wspace=0.2, hspace=0.3)

            # --- Original Detection
            ax1 = fig.add_subplot(grid[0, :])
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            ax1.imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
            ax1.set_title("1. Original Detection (YOLOv11)", fontsize=14, fontweight='bold')
            ax1.axis('off')

            # --- Binarized Warp
            ax2 = fig.add_subplot(grid[1, 0])
            ax2.imshow(binary_plate, cmap='gray')
            ax2.set_title("2. Perspective Corrected (Binarized for OCR)", fontsize=12)
            ax2.axis('off')

            # --- OCR Results
            ax3 = fig.add_subplot(grid[1, 1])
            ax3.text(0.1, 0.7, f"PLATE: {raw_text}", fontsize=20, fontweight='bold', color='green')
            ax3.text(0.1, 0.4, f"STATE: {state_name}", fontsize=16, color='blue')
            ax3.text(0.1, 0.1, f"TYPE: Standard JPJ", fontsize=14)
            ax3.set_title("3. OCR Intelligence Result", fontsize=12)
            ax3.axis('off')

            plt.savefig(save_path, dpi=150)
            plt.close()

            print(f"[SAVED] → {save_path}")
            return

# ============================================================
# 4. RUN ON ALL IMAGES IN FOLDER
# ============================================================

files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

print(f"[INFO] Found {len(files)} images in folder.")

for fname in files:
    input_path = os.path.join(IMAGE_FOLDER, fname)
    output_path = os.path.join(SAVE_FOLDER, f"result_{fname}.png")
    print(f"[PROCESSING] {fname}")
    process_single_image(input_path, output_path)

print("[DONE] All images processed and saved!")
