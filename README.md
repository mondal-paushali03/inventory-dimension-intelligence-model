# Inventory Dimension Intelligence

> Computer vision pipeline for automated medicine package dimension estimation (L × B × H) using YOLO, SAM, OpenCV, and ArUco calibration for warehouse smart binning optimization.
---

# Overview

Warehouse bin allocation systems rely heavily on accurate SKU dimensions.  
In many warehouse environments, product dimensions are manually entered into inventory systems, leading to incorrect storage-capacity estimation and fragmented inventory allocation.

### Example Problem
- System estimates a bin can store **50 units**
- Actual capacity is only **24 units**
- Remaining inventory spills into secondary bins

This increases:
- Picker travel distance
- Split inventory placement
- Picking TAT
- Fulfillment inefficiencies

This project automates real-world medicine package dimension estimation from images to support smarter warehouse binning decisions.

---

# Internal Hackathon Recognition

🏆 **Top 12 / 35 Teams**  
HackEasy - Internal Hackathon @ PharmEasy

---

# Sample Inputs

<img width="1622" height="961" alt="mask_1" src="https://github.com/user-attachments/assets/e793c322-18da-45a2-ba17-dca8dc48a45e" />
<img width="1622" height="862" alt="mask_2" src="https://github.com/user-attachments/assets/0ad618a6-1326-4146-b226-c9bb41fb06fa" />


---

# Final Measurement Pipeline

The system performs:

- Automatic medicine package detection
- SAM-based segmentation
- ArUco marker calibration
- Edge-depth refinement
- Perspective-aware correction
- Automated L × B × H estimation
  
<img width="2412" height="1476" alt="final" src="https://github.com/user-attachments/assets/6e45c0df-ad12-4c4b-8eb9-500981e8c4c3" />

---

# Tech Stack

- Python
- OpenCV
- YOLO (Ultralytics)
- Segment Anything Model (SAM)
- NumPy
- Matplotlib

---

# Core Pipeline

## 1. Image Quality Enhancement
- CLAHE contrast enhancement
- Gamma correction
- Denoising
- Blur handling
- Low-light preprocessing

## 2. Object Detection
- YOLO-based medicine package detection
- Confidence sweep fallback logic
- Marker-aware filtering

## 3. Segmentation
- Segment Anything Model (SAM)
- Edge-aware contour refinement
- Morphological cleanup

## 4. Calibration
- ArUco marker-based scale estimation
- Perspective-aware correction
- Real-world mm-per-pixel conversion

## 5. Geometry Measurement
- Rotated contour extraction
- Bounding rectangle estimation
- Automated L × B × H calculation

---

# Key Features

✅ Automatic medicine package detection  
✅ Real-world metric dimension estimation  
✅ Hybrid YOLO + SAM segmentation pipeline  
✅ ArUco marker calibration  
✅ Perspective correction  
✅ Low-contrast image enhancement  
✅ Edge-depth contour refinement  
✅ Marker-overlap rejection logic  
✅ Custom fallback detection pipeline  

---

# Dataset

Since standard YOLO datasets do not contain pharmaceutical packaging subclasses, a custom dataset was created manually.

### Dataset Details
- 61 manually labeled medicine package images
- Top-view and side-view captures
- Multiple packaging formats:
  - bottles
  - sachets
  - packets
  - strips
  - medicine boxes

---

# Warehouse Impact

This system was designed for warehouse smart binning optimization.

### Operational Benefits
- Reduces incorrect bin allocation
- Minimizes split inventory placement
- Improves warehouse space utilization
- Reduces picker confusion
- Supports faster fulfillment workflows

---

# Installation

```bash
git clone https://github.com/mondal-paushali03/inventory-dimension-intelligence.git

cd inventory-dimension-intelligence

pip install -r requirements.txt
```

---

# Run

```bash
python ucode_dimensions.py
```

---

# Model Weights

Due to GitHub storage limitations, model checkpoints are not included in this repository.

Download separately:
- SAM checkpoint: https://github.com/facebookresearch/segment-anything
- YOLO weights: https://github.com/ultralytics/ultralytics

---

# Future Improvements

- Depth camera integration
- Real-time warehouse camera ingestion
- Multi-object batch inference
- Streamlit deployment
- SKU-wise dimension database
- Automated slotting recommendations

---

# Author

## Paushali Mondal

Data Analyst
### Connect
- Email: mondal.paushali384@gmail.com
- LinkedIn: https://linkedin.com/in/YOUR-LINKEDIN
