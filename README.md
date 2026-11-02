# MC-DDI: Monte Carlo Dropout for Multi-Class Drug-Drug Interaction Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A deep learning framework for predicting drug-drug interactions (DDI) across 86 interaction types using Monte Carlo Dropout for uncertainty quantification and pseudo-label self-training.

---

## 📊 Overview

This project implements a multi-class classification model for predicting drug-drug interactions from the DrugBank dataset using:

- **Monte Carlo Dropout** for uncertainty-aware predictions
- **Self-training with pseudo-labels** to leverage unlabeled data
- **86-class multi-class classification** for biologically meaningful interaction types

### Key Results

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Baseline | 96.01% | 95.05% | 89.75% | 91.26% |
| MC-DDI (Self-Trained) | 96.82% | 96.06% | 96.09% | 95.66% |
| **Improvement** | **+0.85%** | **+1.06%** | **+7.07%** | **+4.82%** |

---

## 🏗️ Architecture

Input (4096-dim Morgan Fingerprints)
↓
Encoder Network (3-layer MLP with Dropout)
↓
Classifier (86 interaction types)
↓
Monte Carlo Dropout (30 forward passes)
↓
Confidence-based Pseudo-labeling

text

---

## 📁 Project Structure

MC_DDI_Project/
├── model.py # MC-DDI model architecture
├── train_baseline_2.py # Baseline training script
├── self_training_2.py # Self-training with pseudo-labels
├── evaluate_2.py # Evaluation and visualization
├── requirements_2.txt # Python dependencies
├── README.md # This file
├── X_features.npy # Preprocessed drug pair features (not in repo)
├── y_labels.npy # Interaction type labels (not in repo)
└── outputs/ # Generated models and plots
├── baseline_model_2.pth
├── mc_ddi_model_self_trained_2.pth
├── training_progress_2.png
├── model_comparison_2.png
└── confusion_matrix_*.png

text

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CPU or GPU with CUDA support (optional)

### Installation

1. **Clone the repository**
git clone https://github.com/YOUR_USERNAME/MC-DDI-Project.git
cd MC-DDI-Project

text

2. **Create virtual environment**
python3 -m venv mc_ddi_env
source mc_ddi_env/bin/activate # On Windows: mc_ddi_env\Scripts\activate

text

3. **Install dependencies**
pip install -r requirements_2.txt

text

4. **Prepare data**
- Download DrugBank DDI dataset or use Therapeutics Data Commons (TDC)
- Preprocess data to generate `X_features.npy` and `y_labels.npy`
- Place files in the project root directory

### Usage

#### Step 1: Train Baseline Model
python3 train_baseline_2.py

text
**Outputs:**
- `baseline_model_2.pth` - Trained baseline model
- `baseline_results_2.json` - Evaluation metrics
- `training_progress_2.png` - Training curves

#### Step 2: Self-Training with Pseudo-Labels
python3 self_training_2.py

text
**Outputs:**
- `mc_ddi_model_self_trained_2.pth` - Self-trained model
- `confidence_distribution_2.png` - Pseudo-label confidence plot

#### Step 3: Evaluate and Compare Models
python3 evaluate_2.py

text
**Outputs:**
- `final_results_2.json` - Comprehensive comparison
- `model_comparison_2.png` - Metric comparison chart
- `confusion_matrix_baseline_2.png` & `confusion_matrix_mcddi_2.png`

---

## 📈 Visualization Examples

### Training Progress
![Training Progress](outputs/training_progress_2.png)

### Model Comparison
![Model Comparison](outputs/model_comparison_2.png)

### Confusion Matrix
![Confusion Matrix](outputs/confusion_matrix_mcddi_2.png)

---

## 🔬 Methodology

### 1. Baseline Training
- Architecture: 3-layer MLP with batch normalization and dropout
- Loss: Cross-entropy for 86 classes
- Optimizer: Adam (lr=0.001)
- Epochs: 20

### 2. Self-Training Strategy
- **Pseudo-label generation**: MC Dropout with 30 forward passes
- **Confidence threshold**: 0.7
- **Rounds**: 3 iterative refinement cycles
- **Learning rate**: Reduced to 0.0001 for fine-tuning

### 3. Uncertainty Quantification
Monte Carlo Dropout provides:
- Mean prediction across stochastic passes
- Variance-based confidence scores
- Selective pseudo-labeling of high-confidence samples

---

## 📊 Dataset

- **Source**: DrugBank via Therapeutics Data Commons (TDC)
- **Samples**: 191,808 drug pairs
- **Classes**: 86 interaction types
- **Features**: 4096-dimensional Morgan fingerprints (concatenated)
- **Split**: 70% train, 15% validation, 15% test (stratified)

---

## 🛠️ Dependencies

Core libraries:
- PyTorch 2.3.1+
- NumPy 1.26.4
- scikit-learn 1.5.1
- Matplotlib 3.8.4
- Seaborn 0.13.2

Full list in `requirements_2.txt`

---
