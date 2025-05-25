## 🧪 Soil vs. Not Soil Classification Using CNN

> A deep learning-based binary image classification project to distinguish real soil images from synthetically generated "non-soil" images, using Convolutional Neural Networks (CNNs) in TensorFlow/Keras.

---

## 📌 Introduction

Soil plays a vital role in agriculture, ecology, and land-use planning. Automating the identification of soil versus non-soil images is a key step toward reliable soil classification systems in larger pipelines. In this challenge, we use a CNN-based computer vision pipeline to predict whether a given image depicts soil or not.

This project leverages both real labeled soil images and synthetically generated "non-soil" images, using color, texture, and spatial patterns for training.

---

## 🧠 Model Architecture

We used a simple yet effective **Convolutional Neural Network (CNN)** architecture built in Keras:

- **Input Shape:** 224x224 RGB images
- **Layers:**
  - Conv2D → ReLU → MaxPooling2D
  - Conv2D → ReLU → MaxPooling2D
  - Flatten → Dense → Dropout
  - Final Dense with Sigmoid activation for binary classification
- **Optimizer:** Adam  
- **Loss Function:** Binary Crossentropy  
- **Metrics:** Accuracy  

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

## 🧪 Dataset Preparation
Soil Images: Real soil images from 4 classes (treated as "Soil").

Non-Soil Images: 500 synthetic images generated using random patterns (noise, gradients, stripes, grids, blobs) via NumPy and OpenCV.

Image Preprocessing:

Resizing to 224×224

Normalizing pixel values

Augmentation: zoom_range, horizontal_flip to improve generalization

## 📈 Data Augmentation
To make the model more robust to image variability (lighting, orientation, scale), we used:

python
CopyEdit
ImageDataGenerator(
    rescale=1./255,
    zoom_range=0.2,
    horizontal_flip=True
)
This helped prevent overfitting and improved validation F1 score.

## ⚖️ Class Balancing
Class weights were computed using sklearn.utils.compute_class_weight() to address the imbalance between soil and non-soil examples.

## 🚧 Challenges Faced & Fixes
Challenge	Solution
Low F1 Score	Added data augmentation to improve generalization and robustness.
Mismatch in Submission Size (Expected 341, Got 339)	Merged test_df with predictions using a left join and filled missing entries with the most common soil type.
Predicted Labels Instead of Names	Mapped numeric labels to actual class names using a label dictionary.
Invalid Image Filenames Warning	Filtered out non-existent or corrupted image paths using os.path.exists.

## 📊 Confusion Matrix
To evaluate performance on the validation set, we plotted a confusion matrix:

Visualizes the distribution of true vs predicted labels.

## 📌 Results
Minimum F1 Score: Focused on ensuring both soil and not-soil classes perform well.

Final Model: Achieved balanced precision and recall, with stable performance across classes.

Use Case: The model can be extended to pre-filter datasets for soil classification pipelines.

## 🧾 Conclusion
This project demonstrates a reliable deep learning approach to differentiate between soil and non-soil images. By combining synthetic data generation, data augmentation, and a simple CNN, we achieved strong validation results. It can serve as a baseline or preprocessing step for more advanced multi-class soil classification tasks.

## 🛠️ Tech Stack
Python 3

TensorFlow/Keras

OpenCV + NumPy

Pandas & Scikit-learn

Matplotlib & Seaborn

## 🚀 Future Improvements
Integrate soil type classification (e.g., Clay, Red, Black, Alluvial)

Use Transfer Learning with pretrained CNNs (like ResNet or EfficientNet)

Add domain-specific features (e.g., histograms, GLCM)

## 📂 Project Structure

├── generate_non_soil_images.py  # Synthetic image generation
├── train_model.py               # CNN training pipeline
├── evaluate_model.py            # Evaluation scripts and F1 analysis
├── utils.py                     # Helper functions
├── conf_matrix.png              # Final confusion matrix plot
└── README.md                    # Project documentation



