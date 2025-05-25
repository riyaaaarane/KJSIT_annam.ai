
# 🌱 Soil vs Not-Soil Classification - Kaggle Challenge 2025

This repository contains a full pipeline for **binary classification** of soil vs non-soil images using **Convolutional Neural Networks (CNNs)**. The goal is to build a model that can accurately distinguish real soil images from synthetically generated “not soil” images using deep learning and image processing techniques.

---

## 🗂️ Directory Structure

```
challenge-2/
├── data/
│   ├── download.sh             # Script to download dataset
├── notebooks/
│   ├── soil_classification- part 2.ipynb  # Main notebook (renamed)
│   └── leaderboard-score/         # F1-score or result logs
├── docs and cards/
│   ├── task2_artitecture.png         # CNN model architecture diagram
│   ├── ml-metrics.json         # Model evaluation metrics (min F1-score etc.)
├── README.md                   # Project overview and usage guide
└── requirements.txt            # List of required Python packages

```

---

## 📌 Problem Statement

Soil plays a foundational role in agriculture and the environment. This challenge focuses on **automating soil identification from images**. However, rather than classifying soil types, the task here is binary:  
📷 **Is the given image soil or not soil?**

This can help in building robust systems that **filter irrelevant data** before fine-tuned classification or in preprocessing stages of larger agriculture-based ML pipelines.

---

## 🔄 Data Processing & Augmentation

### 🔍 Preprocessing
- All images resized to `224x224`
- Pixel values normalized (`rescale=1./255`)
- Paths were validated to **avoid loading errors**

### 🎨 Data Augmentation
Using Keras `ImageDataGenerator`:
- Horizontal flips
- Zoom-in transformations

Synthetic **"Not Soil"** images were generated using:
- Random noise
- Color stripes
- Grid overlays
- Gradients
- Blobs

This diversity helped the model learn **clear visual distinctions** between soil and artificial patterns.

---

## 🧠 Model Architecture

We implemented a **custom CNN** using TensorFlow/Keras with the following architecture:

```
Conv2D → ReLU → MaxPooling  
Conv2D → ReLU → MaxPooling  
Flatten → Dense → Dropout → Dense (Sigmoid)
```

📐 **Input shape**: `(224, 224, 3)`  
🧮 **Output**: Binary classification (1 = Soil, 0 = Not Soil)

---

## 🚧 Challenges Faced & Solutions

### ⚠️ Low F1 Score  
To improve generalization:
- We applied **stronger data augmentation**
- Used **class weights** to handle imbalance

✅ This helped improve the F1 scores for both soil and not-soil classes.

---

### ❌ Invalid Image Paths  
We encountered this warning:  
```
UserWarning: Found 8 invalid image filename(s)
```

✅ **Fix**:
```python
train_df = train_df[train_df['full_path'].apply(os.path.exists)]
val_df = val_df[val_df['full_path'].apply(os.path.exists)]
```

This ensured that only valid image paths were passed to the data generators.

---

## 📊 Evaluation

We used `classification_report` and `confusion_matrix` for evaluation.  
Below is the **confusion matrix** visualized from validation predictions:

![Confusion Matrix](outputs/confusion_matrix.png)

| Metric         | Value   |
|----------------|---------|
| F1 (Soil)      | 0.94 ✅ |
| F1 (Not Soil)  | 0.93 ✅ |
| **Min F1**     | **0.93** 🔥 |

All evaluation metrics are stored in:  
📄 `outputs/metrics.json`

---

## 🚀 Running the Project

```bash
# Clone the repository
git clone https://github.com/your-username/soil_classification- part 2.git
cd soil-vs-notsoil

# Install dependencies
pip install -r requirements.txt
```

### (Optional) Generate synthetic non-soil images  
*This step is already handled in the notebook.*

### Run the notebook
Open the notebook:

```
notebooks/soil_classification- part 2.ipynb
```

➡️ **Run all cells** in sequence to train, validate, and evaluate the model.

---

## ✅ Conclusion

This project showed how combining **synthetic data generation** with **CNNs** and **augmentation techniques** can build a **robust binary soil classifier**. Even with limited real-world "not soil" samples, we effectively taught the model to ignore noise and artificial distractions.

---
