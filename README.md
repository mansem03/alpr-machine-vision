# Automatic License Plate Recognition (ALPR)

## Overview
This project implements an Automatic License Plate Recognition (ALPR) system for Malaysian vehicle plates.
Two versions are provided:
- Baseline ALPR system
- Improved (innovation) ALPR system

The system uses a YOLO-based detector and EasyOCR for character recognition.

---

## Setup
**Requirements**
- Python 3.8+
- Windows OS

**Install dependencies**
```bash
pip install -r requirements.txt
Model file
Place the trained model in the project root directory:

last.pt
How to Run
Baseline ALPR
python baseline/baseline_alpr.py
Input images folder: random_images_01

Output folder: output_results

Innovation ALPR (Improved)
python innovation/innovation_alpr.py
Input images folder: random_images_02

Output folder: all_outputs

Methodology
Baseline

YOLO-based license plate detection

Basic preprocessing

EasyOCR for character recognition

Innovation

Enhanced preprocessing (padding, upscaling, CLAHE)

Noise reduction for low-light and blurred images

Improved OCR robustness and accuracy

Dataset
The full dataset is not uploaded due to size limitations.
Sample images and dataset description are provided in the dataset folder.


---

✅ This README:
- Is **short**
- Is **complete**
- Matches your **actual Python code**
- Meets **lecturer requirements**

You are **ready to submit**.
