
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
### How to Run?

# MedMNIST Dataset Setup Instructions

Since the datasets `dermamnist`, `octmnist`, and `pneumoniamnist` are too large to store on GitHub due to constraints, follow these steps to install MedMNIST and retrieve the necessary datasets.

---

## 📦 Installation

MedMNIST can be easily installed via pip:

```bash
pip install medmnist
```

Verify the installation by running:
```bash
python -c "import medmnist; print(medmnist.INFO)"
```

---

## 📥 Downloading Datasets

MedMNIST provides APIs to download and load the required datasets. Below are instructions for downloading `dermamnist`, `octmnist`, and `pneumoniamnist`.

### Example: Download `dermamnist`

Use the following Python script to download the `dermamnist` dataset:

```python
from medmnist import INFO
from medmnist.dataset import DermaMNIST

# Specify the dataset
data_flag = 'dermamnist'

# Load the dataset
dataset = DermaMNIST(split='train', download=True)

# Dataset info
info = INFO[data_flag]
print(f"Dataset: {info['description']}")
print(f"Number of Classes: {info['n_channels']}")
```

Repeat the process for `octmnist` and `pneumoniamnist` using `OCTMNIST` and `PneumoniaMNIST` classes, respectively.

---

## 🛠️ Example Workflow

Here’s a basic workflow to integrate these datasets into your machine learning pipeline:

```python
import medmnist
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Define dataset and transforms
data_flag = 'pneumoniamnist'
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[.5], std=[.5])])

# Load datasets
train_dataset = medmnist.PneumoniaMNIST(split='train', transform=transform, download=True)
test_dataset = medmnist.PneumoniaMNIST(split='test', transform=transform, download=True)

# Create dataloaders
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Iterate through the data
for images, labels in train_loader:
    print(images.shape, labels.shape)
    break
```

---

## 🔗 Additional Resources

- [MedMNIST GitHub Repository](https://github.com/MedMNIST/MedMNIST)
- [MedMNIST Documentation](https://medmnist.com/)

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
├── README.md
├── data/               # Contains datasets and preprocessing scripts
│   ├── bloodmnist.npz
│   └── breastmnist_224.npz
├── dataprocessing/     # Thresholding and masking scripts
│   ├── extractimages.py
│   ├── histogram.py
│   └── segment.py
├── experiments/        # Training and evaluation scripts
│   ├── resnet_unet_etal
│   └── transformer
├── folder_structure.txt
├── models/             # Model implementations (ResNet, ViT, UNet)
    ├── resnet_unet_etal
    ├── transformer
    └── unet_model.pth
```

---

## 🛠️ Contributors

- **Sagar Swaraj Mahapatra** (24M1076)
- **Parth Pratim Chatterjee** (24M0748)
- **Utkarsh Tiwari** (24M0754)
- **Vijendra Kumar Vaishya** (24M2133)

IIT Bombay
Course: CS 725 - Foundations of Machine Learning
