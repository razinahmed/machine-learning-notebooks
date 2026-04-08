<div align="center">

# 📊 Machine Learning & Data Science Notebooks

<img src="https://placehold.co/900x250/1e1e2e/22c55e.png?text=Machine+Learning+%7C+Deep+Learning+%7C+PyTorch" alt="Machine Learning Notebooks Banner" />

<br/>

**A curated collection of production-quality Jupyter Notebooks covering Exploratory Data Analysis, Classical ML, and Deep Learning with PyTorch and Scikit-Learn.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=for-the-badge)](http://makeapullrequest.com)

[View Notebooks](#-notebook-catalog) · [Quick Start](#-quick-start) · [Results](#-results--benchmarks) · [Contributing](#-contributing)

</div>

---

## 📋 Table of Contents

- [About](#-about)
- [Notebook Catalog](#-notebook-catalog)
- [Datasets](#-datasets-used)
- [Model Architectures](#-model-architectures)
- [Results & Benchmarks](#-results--benchmarks)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔍 About

This repository is a hands-on resource for anyone learning **Machine Learning**, **Deep Learning**, and **Data Science**. Each notebook is self-contained, heavily commented, and follows best practices for reproducible research. Whether you are a student exploring neural networks for the first time or a practitioner looking for clean reference implementations, these notebooks provide clear, end-to-end workflows from data loading to model evaluation.

---

## 📓 Notebook Catalog

| # | Notebook | Topic | Difficulty | Framework |
|:-:|---|---|:-:|---|
| 01 | **`01_Exploratory_Data_Analysis.ipynb`** | EDA, feature engineering, missing value imputation, distribution analysis | 🟢 Beginner | Pandas, Seaborn, Matplotlib |
| 02 | **`02_Random_Forest_Classifier.ipynb`** | Classification pipeline with cross-validation, hyperparameter tuning (GridSearchCV), feature importance | 🟡 Intermediate | Scikit-Learn |
| 03 | **`03_PyTorch_CNN_MNIST.ipynb`** | Convolutional Neural Network from scratch, training loop, GPU acceleration, confusion matrix | 🔴 Advanced | PyTorch |

> **Tip:** Each notebook includes inline explanations, mathematical intuition, and visualization of intermediate results.

---

## 📊 Datasets Used

| Dataset | Source | Size | Task | Notebook |
|---|---|---|---|---|
| **Titanic Survival** | [Kaggle](https://www.kaggle.com/c/titanic) | 891 rows, 12 features | Binary Classification / EDA | 01, 02 |
| **MNIST Handwritten Digits** | [Yann LeCun](http://yann.lecun.com/exdb/mnist/) | 70,000 images (28x28) | Multi-class Classification | 03 |

---

## 🧠 Model Architectures

### Random Forest Classifier (Notebook 02)

- **Ensemble Method:** Bagging with 200 decision trees
- **Tuning:** GridSearchCV over `max_depth`, `n_estimators`, `min_samples_split`
- **Evaluation:** 5-fold stratified cross-validation, ROC-AUC, precision-recall curves

### PyTorch CNN (Notebook 03)

```
Input (1x28x28)
  → Conv2d(1, 32, 3x3) → BatchNorm → ReLU → MaxPool(2x2)
  → Conv2d(32, 64, 3x3) → BatchNorm → ReLU → MaxPool(2x2)
  → Flatten
  → Linear(1600, 128) → ReLU → Dropout(0.5)
  → Linear(128, 10) → Softmax
```

- **Optimizer:** Adam (lr=0.001)
- **Loss:** CrossEntropyLoss
- **Epochs:** 15
- **Batch Size:** 64

---

## 📈 Results & Benchmarks

| Model | Dataset | Accuracy | Precision | Recall | F1 Score |
|---|---|:-:|:-:|:-:|:-:|
| Random Forest (tuned) | Titanic | **84.2%** | 82.1% | 79.5% | 80.8% |
| PyTorch CNN | MNIST Test Set | **99.1%** | 99.0% | 99.1% | 99.0% |

> Results are reproducible with the seeds set in each notebook. Your results may vary slightly depending on hardware.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- pip or conda package manager
- (Optional) NVIDIA GPU with CUDA for accelerated training

### Option 1: Local Setup

```bash
# Clone the repository
git clone https://github.com/razinahmed/machine-learning-notebooks.git
cd machine-learning-notebooks

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# Install all dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook
```

### Option 2: Google Colab (No Setup Required)

Don't have a local GPU? Open any notebook directly in Google Colab:

| Notebook | Colab Link |
|---|---|
| 01 - EDA | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#) |
| 02 - Random Forest | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#) |
| 03 - PyTorch CNN | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#) |

### Option 3: Conda Environment

```bash
conda create -n ml-notebooks python=3.10
conda activate ml-notebooks
pip install -r requirements.txt
jupyter notebook
```

---

## 📁 Project Structure

```
machine-learning-notebooks/
├── 01_Exploratory_Data_Analysis.ipynb    # EDA with Pandas & Seaborn
├── 02_Random_Forest_Classifier.ipynb     # Scikit-Learn ML pipeline
├── 03_PyTorch_CNN_MNIST.ipynb            # Deep Learning with PyTorch
├── data/                                 # Raw and processed datasets
│   ├── titanic_train.csv
│   └── titanic_test.csv
├── models/                               # Saved model checkpoints
├── figures/                              # Generated plots and charts
├── requirements.txt                      # Python dependencies
├── LICENSE
└── README.md
```

---

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| **Languages** | Python 3.10+ |
| **Data Analysis** | Pandas, NumPy, Matplotlib, Seaborn |
| **Machine Learning** | Scikit-Learn, XGBoost |
| **Deep Learning** | PyTorch, torchvision |
| **Environment** | Jupyter Notebook, Google Colab |
| **Utilities** | tqdm, joblib, pickle |

---

## 🤝 Contributing

Contributions are welcome! If you have a notebook that demonstrates a useful ML technique:

1. **Fork** this repository
2. **Create** a feature branch (`git checkout -b add-new-notebook`)
3. **Add** your notebook following the naming convention `XX_Notebook_Name.ipynb`
4. **Ensure** all cells run top-to-bottom without errors
5. **Submit** a Pull Request with a description of the notebook and results

Please make sure your notebooks include:
- Clear markdown explanations between code cells
- Reproducible results with random seeds
- Properly labeled visualizations

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with passion for Data Science by [Razin Ahmed](https://github.com/razinahmed)**

If these notebooks helped you learn, please consider giving the repo a ⭐

<img src="https://komarev.com/ghpvc/?username=razinahmed&style=flat-square&color=22c55e&label=REPO+VIEWS" alt="Repo Views" />

</div>
