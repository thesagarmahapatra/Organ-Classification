
# Organ Classification using Various Models

This repository contains the work for the **Organ Classification Project**, where we explore and benchmark various models for classifying biomedical images using datasets from **MedMNIST v2** for the **COURSE: CS 725 - FOUNDATIONS OF MACHINE LEARNING**  
at **IIT Bombay**.

---

#### 🖥️ Slides and Reference Paper

- [This is the project presentation for our work.](https://docs.google.com/presentation/d/1U_InYhK74b4Cgh15gAtAl6vdCXSeL2Myc4AG_wGqn6M/edit?usp=sharing)

- [Paper.](https://arxiv.org/pdf/2110.14795)


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

data_flag = 'dermamnist'

dataset = DermaMNIST(split='train', download=True)

info = INFO[data_flag]
print(f"Dataset: {info['description']}")
print(f"Number of Classes: {info['n_channels']}")
```

Repeat the process for `octmnist` and `pneumoniamnist` using `OCTMNIST` and `PneumoniaMNIST` classes, respectively.

---

## 🔗 Additional Resources

- [MedMNIST GitHub Repository](https://github.com/MedMNIST/MedMNIST)
- [MedMNIST Documentation](https://medmnist.com/)

---

# Image Processing Workflow

This repository contains scripts to process images and labels from datasets, including extracting images, generating histograms, creating masks, updating CSV files with mask paths, and applying thresholding to create masked images.

---
## **Requirements**
   ```bash
   pip install matplotlib numpy os pandas PIL opencv-python
   ```
  
## 📜 Steps to Run

### 1. **Extract Image and CSV to a Folder**
   **Script**: `extractimages.py`

   **Command**:
   ```bash
   python extractimages.py --input_file <dataset_file> --output_dir <output_folder>
   ```
### 2. **Create Histogram**
   **Script**: `histogram.py`
  Although saving the output is not necessary per se since the view pops up, you can always observe the histogram and set a mask value by intuition accordingly. We check the slope change points and use that grayscale value. 
   **Command**:
   ```bash
   python histogram.py --image_path <image_file> --output_path <histogram_output>
   ```

### 3. **Thresholding and Creating masks**
   **Script**: `thresholding.py`
  Create the mask as based off histogram output value accordingly by changing the threshold values in the code. 
   **Command**:
   ```bash
   python thresholding.py --input_dir <images_folder> --mask_dir <masks_folder> --output_dir <masked_images_folder>
   ```
We have conducted experiments for training and testing in the Python Notebook files that can be run on Google Colab without the hassle of running all of these scripts for preprocessing. 

Refer to **folder**: `experiments/` for the same. 

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

