# 🎭 Facial Emotion Recognition with ResNet18

A deep learning application for recognizing facial emotions using ResNet18 with transfer learning, trained on the FER2013 dataset.

## 💾 Pre-trained Model

The trained model is too large to upload directly to GitHub (>100MB). You can download the pre-trained model from Google Drive:

**Download Link:** [best_resnet18_emotion_model.pth](https://drive.google.com/file/d/1Mgwim0tjK3sRA0XkoF1Qhon6O-EZPgeX/view?usp=sharing)

## 🎯 Project Overview

This project implements an automated facial emotion recognition system that classifies facial expressions into seven categories:

**Emotions Recognized:**
- 😠 Angry
- 🤢 Disgust
- 😨 Fear
- 😊 Happy
- 😢 Sad
- 😲 Surprise
- 😐 Neutral

The model achieves **65.42% accuracy**, performing at human-level capability on the challenging FER2013 dataset.

### ✨ Key Features
- 🧠 **ResNet18 Architecture** - Deep residual network with skip connections
- 🔄 **Transfer Learning** - Pretrained on ImageNet for better performance
- ⚖️ **Class Balancing** - Weighted loss function to handle imbalanced data
- 📊 **Comprehensive Metrics** - Confusion matrices, precision, recall, F1-scores
- 🌐 **Web Application** - Interactive Streamlit app for real-time predictions

## 📊 Model Performance

### Overall Metrics

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | 65.42% |
| **Training Accuracy** | 83.00% |
| **Validation Accuracy** | 65.42% |
| **Macro F1-Score** | 0.64 |
| **Weighted F1-Score** | 0.65 |

### Per-Class Performance

| Emotion | Accuracy | Precision | Recall | F1-Score | Support |
|---------|----------|-----------|--------|----------|---------|
| 😊 Happy | 84.50% | 0.84 | 0.84 | 0.84 | 1774 |
| 😲 Surprise | 81.47% | 0.74 | 0.81 | 0.77 | 831 |
| 🤢 Disgust | 70.27% | 0.70 | 0.70 | 0.70 | 111 |
| 😐 Neutral | 65.86% | 0.58 | 0.66 | 0.62 | 1233 |
| 😠 Angry | 58.56% | 0.56 | 0.59 | 0.57 | 958 |
| 😢 Sad | 49.40% | 0.55 | 0.49 | 0.52 | 1247 |
| 😨 Fear | 44.24% | 0.54 | 0.44 | 0.49 | 1024 |

### Key Insights

✅ **Strong Performers:** Happy and Surprise (>80% accuracy)  
⚠️ **Challenging Emotions:** Fear and Sad (<50% accuracy)  
🎯 **Balanced Learning:** Disgust achieved 70% despite being the rarest class (only 111 samples)

## 🗂️ Dataset

### FER2013 (Facial Expression Recognition 2013)

**Dataset Statistics:**
- 📁 **Total Images:** 35,887 grayscale images
- 📏 **Image Size:** 48×48 pixels
- 🎯 **Classes:** 7 emotion categories
- 📚 **Training Set:** 28,709 images
- 🧪 **Test Set:** 3,589 images
- 🌐 **Source:** [Kaggle FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)

### Class Distribution

| Emotion | Training Samples | Test Samples | Percentage |
|---------|-----------------|--------------|------------|
| Happy | ~8,989 | 1,774 | 28.7% |
| Neutral | ~6,198 | 1,233 | 19.7% |
| Sad | ~6,077 | 1,247 | 19.3% |
| Fear | ~5,121 | 1,024 | 16.3% |
| Angry | ~4,953 | 958 | 15.8% |
| Surprise | ~4,002 | 831 | 12.8% |
| Disgust | ~547 | 111 | 1.7% |

⚠️ **Note:** Severe class imbalance (Happy: 8,989 vs Disgust: 547)
