# Fruit Classification using CNNs — 98.18% Accuracy on 77 Classes

> A custom Convolutional Neural Network trained on 39,249 images across 77 fruit categories from the Fruits360 dataset. Achieved **98.18% test accuracy** in 10 epochs using aggressive data augmentation and a lightweight 4-layer architecture.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-TensorFlow-red?logo=keras)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-98.18%25-brightgreen)
![Classes](https://img.shields.io/badge/Classes-77%20Fruits-orange)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | **98.18%** |
| Test Loss | 0.0572 |
| Train Accuracy (epoch 10) | 96.50% |
| Validation Accuracy (epoch 10) | 98.18% |
| Training Images | 31,399 |
| Test Images | 7,850 |
| Number of Classes | 77 fruit types |
| Epochs | 10 |

### Training Progression

| Epoch | Train Accuracy | Val Accuracy | Val Loss |
|-------|:--------------:|:------------:|:--------:|
| 1 | 54.32% | 77.85% | 0.6756 |
| 3 | 90.12% | 94.10% | 0.1803 |
| 5 | 94.07% | 93.08% | 0.2029 |
| 7 | 95.72% | 96.29% | 0.1089 |
| 10 | 96.50% | **98.18%** | 0.0549 |

Validation accuracy consistently exceeded training accuracy, indicating strong generalization rather than overfitting — a direct result of aggressive data augmentation.

---

## Dataset

- **Source:** [Fruits360 Dataset (Kaggle)](https://www.kaggle.com/moltean/fruits)
- **Total images (filtered):** 39,249
- **Classes:** 77 fruit types (filtered from 131 total categories — vegetables excluded)
- **Image size:** 100×100px originals, resized to 50×50px for training
- **Split:** 80% train (31,399) / 20% test (7,850), `random_state=42`

---

## Model Architecture

A custom CNN built with Keras Sequential API:

```
Input: (50, 50, 3)
    ↓
Conv2D(32, 3×3, ReLU)       → feature extraction
MaxPooling2D(2×2)            → spatial down-sampling
    ↓
Conv2D(64, 3×3, ReLU)       → deeper feature extraction
MaxPooling2D(2×2)            → spatial down-sampling
    ↓
Flatten()
Dense(128, ReLU)             → classification head
Dense(77, Softmax)           → output: 77 fruit classes
    ↓
Output: class probabilities
```

**Optimizer:** Adam | **Loss:** Categorical Crossentropy | **Batch size:** 64

---

## Data Augmentation

`ImageDataGenerator` applied to training set only:

| Technique | Value |
|-----------|-------|
| Rotation | ±45° |
| Width shift | 20% |
| Height shift | 20% |
| Shear | 20% |
| Zoom | 20% |
| Horizontal flip | Yes |
| Rescaling | 1/255 |

Augmentation is the primary reason validation accuracy (98.18%) exceeds raw training accuracy — the model sees harder versions of training images than what it's tested on.

---

## Project Structure

```
fruit-classification-CNN/
│
├── fruit_classification.ipynb      # Full pipeline: EDA → augmentation → training → evaluation
├── fruit_classification_report.pdf # Detailed project report
├── requirements.txt                # Dependencies
└── README.md
```

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/Bufatima-Nk/fruit-classification-CNN
cd fruit-classification-CNN
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download [Fruits360 from Kaggle](https://www.kaggle.com/moltean/fruits) and update the path in the notebook:
```python
dataset_path = '/your/local/path/fruits-360/Training'
```

### 4. Run the notebook
```bash
jupyter notebook fruit_classification.ipynb
```

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Deep Learning | TensorFlow, Keras |
| Data Processing | NumPy, pandas |
| Visualization | Matplotlib, Seaborn |
| ML Utilities | scikit-learn (train/test split, LabelEncoder) |

---

## Key Observations

**1. Lightweight architecture, high accuracy.** Only 4 layers (2 Conv + 2 Dense) achieve 98.18% on a 77-class problem. This suggests the Fruits360 dataset has clear visual boundaries between classes — the model is not overparameterized.

**2. Augmentation prevents overfitting despite small architecture.** Val accuracy (98.18%) > Train accuracy (96.50%) at epoch 10, which is unusual and indicates the augmented training distribution is harder than the clean test set.

**3. Fast convergence.** The model reaches 94% validation accuracy by epoch 3, suggesting the task is learnable with relatively few gradient updates. Future work could explore early stopping or learning rate scheduling to reduce training time.

---

## Future Work

- **Transfer Learning:** Apply MobileNetV2 or EfficientNet-B0 to compare performance vs. this custom architecture
- **Grad-CAM visualization:** Show which image regions the model attends to for each class
- **Deployment:** Wrap the saved model in a Streamlit or Gradio app for live inference
- **Full 131-class version:** Extend to all categories including vegetables

---

## Author

**Bufatima N.K.**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-bufatima--n--k-blue?logo=linkedin)](https://linkedin.com/in/bufatima-n-k)
[![GitHub](https://img.shields.io/badge/GitHub-Bufatima--Nk-black?logo=github)](https://github.com/Bufatima-Nk)
