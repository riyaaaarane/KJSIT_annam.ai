# 🌾 Soil Classification Challenge - 2025

This repository contains the complete solution to a soil classification task as part of **Kaggle Challenge-1**, where the goal is to classify images of soil into four distinct categories using Convolutional Neural Networks (CNNs).

---

## 🗂️ Directory Structure

challenge-1/
├── data/
│ └── download.sh # Script to download dataset
│
├── docs and cards/
│ ├── artitecture.png # CNN model architecture diagram
│ └── ml-metrics.json # Model evaluation metrics (min F1-score etc.)
│
├── notebooks/
│ ├── leaderboard-score # Scoring file (F1-score or result logs)
│ ├── soil-classification.ipynb# Main notebook with model training & evaluation
│ └── src/ # (Optional) additional scripts/helpers
│
└── requirements.txt # Required Python packages


---

## 📌 Problem Statement

The task is to **predict soil types from image data**, with the following labels:

- `Clay soil`
- `Red soil`
- `Alluvial soil`
- `Black Soil`

The evaluation metric is **minimum F1-score across all classes**, emphasizing balanced performance even on underrepresented soil types.

---

## 🧪 Approach

### 🔢 Preprocessing
- Label encoding of soil types
- Image resizing to `224x224`
- Train-validation split (stratified)

### 🧰 Data Augmentation
Using Keras `ImageDataGenerator`:
- Rotation
- Width/Height Shift
- Brightness adjustment
- Zoom

### 🧠 Model Architecture
Custom CNN:
Conv2D → ReLU → MaxPooling → Conv2D → ReLU → MaxPooling → Flatten → Dense → Dropout → Output (softmax)


> Diagram: See `docs and cards/artitecture.png`

### 📊 Evaluation
- Used `classification_report` & `confusion_matrix`
- Stored final scores in `ml-metrics.json`

---

## ✅ Results

| Metric        | Value   |
|---------------|---------|
| F1 (Clay soil)| 0.6818  |
| F1 (Red soil) | 0.9901  |
| F1 (Alluvial) | 0.7469  |
| F1 (Black)    | 0.9500  |
| **Min F1**    | **0.6818** ✅ |

---

## 🚀 Running the Project

1. **Clone the repo**
   ```bash
   git clone https://github.com/your-username/soil-classification-2025.git
   cd challenge-1
2. **Install dependencies**
   bash
   Copy
   Edit
   pip install -r requirements.txt
3. **Download dataset**
   cd data
   bash download.sh
4. **Run notebook**
   Open notebooks/soil-classification.ipynb and run all cells in order


---

## 🧩 Future Improvements

-Replace custom CNN with pre-trained models (e.g., ResNet50, EfficientNet)
-Handle noisy/incomplete test images more robustly
-Consider ensembling multiple CNN models
