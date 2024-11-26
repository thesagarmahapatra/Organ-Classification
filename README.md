
# Organ Classification using Various Models

This repository contains the work for the **Organ Classification Project**, where we explore and benchmark various models for classifying biomedical images using datasets from **MedMNIST v2**. 

---

## 📜 Overview

### About MedMNIST
MedMNIST v2 is a large-scale, lightweight benchmark dataset for 2D and 3D biomedical image classification. It contains **12 datasets** of biomedical images, offering multiple classification and segmentation baselines for different models.

---

## ⚙️ Techniques and Tools

- **Masking**
- **Thresholding**
- **Fine Tuning**
- Models used:
  - ResNet
  - Vision Transformer (ViT)
  - Modified UNet

---

## 🔬 Experiments and Results

### ResNet
- Leveraged results directly from the MedMNIST base paper for baseline comparison.

### Vision Transformer (ViT)
- Used a subset of the **OCTMNIST** dataset due to computational constraints.
- Key parameters:
  - Weight decay: 0.01
  - Warmup steps: 500
  - Epochs: 3 (optimal based on trial and error)
- **Results:** Beat a few baselines reported in the MedMNIST paper.

### Modified UNet
- Custom thresholding values determined through manual estimation.
- Trained for 5 epochs.
- **Results:** Outperformed all baselines from the MedMNIST paper.

### ViT vs Google AutoML Vision
- Achieved the largest margin of improvement over Google AutoML Vision in the **RetinaMNIST** dataset.

---

## 📊 Table of Results

| Model                  | Dataset      | Baseline Accuracy (%) | Our Accuracy (%) | Notes                                   |
|-------------------------|--------------|-----------------------|------------------|-----------------------------------------|
| ResNet                 | OCTMNIST     | 85.0                 | 85.0             | Taken directly from the MedMNIST paper. |
| Vision Transformer (ViT) | OCTMNIST   | 83.5                 | 84.8             | Beat the baseline with fine-tuning.     |
| Modified UNet          | RetinaMNIST  | 92.3                 | 94.7             | Outperformed all baselines.             |
| Vision Transformer (ViT) vs AutoML | RetinaMNIST | 88.5                 | 90.5             | Significant margin over AutoML.         |

---
## ❌ Challenges

- Fine-tuning Vision Transformer and Modified UNet models required significant parameter adjustments and manual tuning.
- Computational limitations impacted dataset sampling and training iterations.

---

## 🚀 Future Work

- Scale experiments to use larger portions of MedMNIST datasets.
- Explore advanced hyperparameter optimization techniques.
- Benchmark against newer state-of-the-art models.

---

## 📂 Project Structure

```
├── data/               # Contains datasets and preprocessing scripts
├── models/             # Model implementations (ResNet, ViT, UNet)
├── experiments/        # Training and evaluation scripts
├── dataprocessing/     # Thresholding and masking scripts
└── README.md           # Project documentation
```

---

## 🛠️ Contributors

- **Sagar Swaraj Mahapatra** (24M1076)
- **Parth Pratim Chatterjee** (24M0748)
- **Utkarsh Tiwari** (24M0754)
- **Vijendra Kumar Vaishya** (24M2133)

IIT Bombay
Course: CS 725 - Foundations of Machine Learning
