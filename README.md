# YOLOv8 Face Mask Detector – Real-Time Webcam Inference

This project demonstrates a **real-time face mask detection system** using **YOLOv8**, capable of identifying whether a person is wearing a mask or not. The system can run on a **live webcam feed** and has been trained on a custom dataset with multiple augmentation strategies to improve robustness and accuracy.

---

## 🚀 Features
- ✅ Real-time mask vs. no-mask detection (Webcam)
- ✅ YOLOv8-based custom-trained model
- ✅ Fast inference (runs on CPU or GPU)
- ✅ Uses smart augmentations for better generalization
- ✅ Clean and lightweight repository

---

## 🧠 Dataset
- Custom dataset with **2 classes**: `Mask`, `No Mask`, `Mask Worn Incorrectly`
- Split defined via `dataset.yaml` 
- Images resized and trained at **640×640**

---

## 🧪 Data Augmentation & Training Strategy

To improve generalization and reduce overfitting, the following augmentations and strategies were applied:

| Technique | Setting | Purpose |
|-----------|---------|---------|
| `augment=True` | Enabled | Enables YOLO's built-in HSV, flip, scale, shear etc. |
| `mosaic=1` | ON | Enhances contextual learning by combining images |
| `mixup=0.1` | Light mixup | Improves robustness and prevents overfitting |
| `pretrained=True` | COCO weights | Transfer learning for faster convergence |
| `epochs=110` | Long training | Stable optimization |
| `patience=20` | Early stopping | Stops if no improvement |
| `batch=16` | Balanced batch | Good speed vs. stability |
| `imgsz=640` | Standard YOLO size | Best performance trade-off |

---

## 🏋️ Training (Google Colab)

Training was performed using YOLOv8s (Ultralytics):

```python
from ultralytics import YOLO

model = YOLO('yolov8s.pt')  # load pretrained YOLOv8s weights

model.train(
    data='/content/dataset/dataset.yaml',
    epochs=110,
    augment=True,
    mosaic=1,
    mixup=0.1,
    imgsz=640,
    batch=16,
    patience=20,
    workers=2,
    name='mask_detection_yolov8s',
    project='/content/runs',
    pretrained=True,
    device=0
)
```
---
## 📊 Results

| Metric | Score |
|--------|-------|
| `mAP50` |	82.281% |
| `mAP50-95` | 54.542% |


Training output visualizations (from YOLO runs/train/):

|Result	| Description |
|-------|-------------|
| `results.png`	| Training curves |
| `confusion_matrix.png` | Class performance |
| `BoxPR_curve.png` | Precision-Recall |
| `BoxF1_curve.png` | F1-score |
| `labels.jpg` | Dataset distribution |
| `val_batch0_labels.jpg` | Sample predictions |

---

## 🖥️ Run Real-Time Inference (Webcam)
```
python src/face_detection_app.py
```
This script:

Loads your best.pt

Opens webcam

Draws bounding boxes + labels + FPS

Press q to quit

---

## 📌 Repository Structure
```
Face-Mask-Detection-YOLOv8/
│
├── models/
│   └── best.pt
│
├── src/
│   ├── face_detection_app.py
│   └── Face_Mask_Detection.ipynb
│
├── runs_results/
│   ├── results.png
│   ├── confusion_matrix.png
│   ├── BoxF1_curve.png
│   ├── BoxPR_curve.png
│   ├── labels.jpg
│   ├── val_batch0_labels.jpg
│
├── data/
│   └── dataset.yaml
│
├── README.md
├── requirements.txt
└── .gitignore

```
---
## 📦 Installation
```
pip install -r requirements.txt
```
---
## 🛠️ Tech Stack
```
YOLOv8 (Ultralytics)

PyTorch

OpenCV

Python 3
```

