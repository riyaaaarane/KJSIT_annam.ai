# 🧪 Soil vs. Not Soil Classification Using CNN

A deep learning-based binary image classification project to distinguish real soil images from synthetically generated "non-soil" images, using Convolutional Neural Networks (CNNs) in TensorFlow/Keras.

---

## 📌 Introduction

Soil plays a vital role in agriculture, ecology, and land-use planning. Automating the identification of soil versus non-soil images is a key step toward reliable soil classification systems in larger pipelines. In this challenge, we use a CNN-based computer vision pipeline to predict whether a given image depicts soil or not.

This project leverages both real labeled soil images and synthetically generated "non-soil" images, using color, texture, and spatial patterns for training.

---

## 🧠 Model Architecture

We used a simple yet effective Convolutional Neural Network (CNN) architecture built in Keras:

- **Input Shape**: 224x224 RGB images  
- **Layers**:
  - Conv2D → ReLU → MaxPooling2D  
  - Conv2D → ReLU → MaxPooling2D  
  - Flatten → Dense → Dropout  
  - Final Dense with Sigmoid activation for binary classification  
- **Optimizer**: Adam  
- **Loss Function**: Binary Crossentropy  
- **Metrics**: Accuracy  

### 🧱 Model Summary (Code Snippet)

```python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

---

## 🗂️ Dataset Preparation

- **Soil Images**: Real soil images from 4 classes (treated as "Soil")  
- **Non-Soil Images**: 500 synthetic images generated using random patterns *(noise, gradients, stripes, grids, blobs)* via **NumPy** and **OpenCV**

---

## 🧼 Image Preprocessing

- Resizing to **224×224**
- Normalizing pixel values
- Augmentation: `zoom_range`, `horizontal_flip` to improve generalization

---

## 📈 Data Augmentation

To make the model more robust to image variability (lighting, orientation, scale), we used:

```python
ImageDataGenerator(
    rescale=1./255,
    zoom_range=0.2,
    horizontal_flip=True
)

