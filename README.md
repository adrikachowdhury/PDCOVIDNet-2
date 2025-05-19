# 🫁 COVID-19 Detection using PDCOVIDNet  
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This deep learning project implements a custom convolutional neural network architecture named **PDCOVIDNet** to detect **COVID-19, viral pneumonia, and normal** chest X-ray images. Developed as part of the **_Computer Vision and Robotics_** course, the project was inspired by the research paper:

📄 **PDCOVIDNet: A Parallel-Dilated Convolutional Neural Network Architecture for Detecting COVID-19 from Chest X-ray Images**  
*Chowdhury et al., Health Information Science and Systems (2020)*  
[DOI: 10.1007/s13755-020-00119-3](https://doi.org/10.1007/s13755-020-00119-3)

---

## 🚀 Features
- Custom CNN architecture (PDCOVIDNet) with parallel dilated convolution blocks
- Binary and multi-class classification of chest X-ray images (COVID-19, Pneumonia, Normal)
- Handles class imbalance using WeightedRandomSampler and class-weighted loss
- Image augmentation techniques for robust training (rotation, affine transform, grayscale)
- Macro-averaged evaluation metrics (Accuracy, Precision, Recall, F1)
- Visualization of dataset distribution and classification results

## 📊 Dataset
- **Kaggle Dataset**: [Chest X-ray Images](https://www.kaggle.com/datasets/alsaniipe/chest-x-ray-image)
- 3 Classes: `COVID19`, `PNEUMONIA`, `NORMAL`
- Images are resized to **224×224**, and grayscale conversion is applied

## 🧠 Model Architecture
- **Input**: 224×224 grayscale chest X-ray image
- **Two Parallel Branches**:
  - 5× DilatedConvBlock (Conv2D → ReLU → Conv2D → ReLU → MaxPool2D)
    - Branch 1: Dilation rate = 1
    - Branch 2: Dilation rate = 2
- **Fusion**:
  - Element-wise addition of both branches
  - Conv2D (3×3, 512 filters) to refine merged features
- **Flatten**
- **Fully Connected Layers**:
  - Dense(1024) → ReLU → Dropout(0.3)
  - Dense(1024) → ReLU → Dropout(0.3)
  - Dense(3) → Output logits (COVID-19, Pneumonia, Normal)

---

## 🛠️ Tools & Libraries
- PyTorch, Torchvision
- NumPy
- pandas
- os
- scikit-learn
- matplotlib
- Kaggle API
- WeightedRandomSampler

---

## ⚙️ Setup
1. Download the repository from GitHub.
2. Install dependencies.
3. Upload your Kaggle API key (`kaggle.json`) and configure it.
4. Download and extract the dataset from Kaggle.
5. Open `PDCOVIDNet_2.ipynb` in Google Colab or Jupyter Notebook.
6. Run all cells to train and evaluate the model.
