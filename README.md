# 🐦 Bird Species Classifier (MobileNetV2 + Transfer Learning)

A deep learning project built with **TensorFlow** and **Keras**, designed to classify bird species from images. It uses **MobileNetV2** for transfer learning, along with data augmentation, class-wise evaluation metrics, and visualization graphs.

---

## 🚀 Features

- 🧠 **Transfer Learning** using pre-trained MobileNetV2
- 📸 **Real-time Image Prediction** via terminal input
- 📊 **Evaluation Dashboard**:
  - Confusion Matrix Heatmap
  - Class-wise Accuracy Bar Chart
  - Dataset Distribution Visualization
- ⚙️ **Enhanced Data Augmentation** for robust model training
- 💾 Saves and loads model from `bird_classifier.h5`

---

## 🧠 Model Architecture

- Base: `MobileNetV2` (Frozen layers)
- Global Average Pooling
- Dense Layer: 512 units + ReLU + Dropout
- Output Layer: `softmax` activation for multi-class classification
