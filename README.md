# DL_Project — AI-Generated vs Real Artwork Detection

Binary classification (**Real vs Fake**) of AI-generated artwork.  
Built for the Deep Learning course (A.Y. 2025-26), Master's in AI, UniVR.

---

## What it does

Compares three learning paradigms to detect AI-generated images:

- **Supervised** — ResNet18 and ViT-B/16 baselines trained on labeled data
- **Self-supervised (SimCLR v2)** — contrastive pretraining (NT-Xent loss, τ=0.5) followed by linear probe / full fine-tuning
- **Semi-supervised (pseudo-labeling)** — iterative training with confidence gating (threshold 0.95) across labeled ratios (10 %, 20 %, 50 %)

Additionally studies how common image augmentations (JPEG compression, Gaussian blur, noise, color jitter, random crop) erode the discriminative signal.


---

## Project structure

```
DL_Project/
├── src/                          # Reusable Python modules
│   ├── models.py                 # ResNet18 / ViT-B/16 definitions
│   ├── datasets.py               # Dataset loading & splits
│   ├── augmentations.py          # Augmentation pipelines
│   ├── train_supervised.py       # Supervised training loop
│   ├── utils.py                  # Metrics, logging, helpers
│   ├── ssl/                      # SimCLR v2 pretraining code
│   └── semi_supervised/          # Pseudo-labeling pipeline
├── experiments/                  # Jupyter notebooks (7 studies)
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_supervised_baseline.ipynb
│   ├── 03_augmentation_study.ipynb
│   ├── 04_simclr_pretrain.ipynb
│   ├── 05_simclr_finetune.ipynb
│   ├── 06_semi_supervised_training.ipynb
│   └── 07_error_analysis_and_demo.ipynb
├── datasets/                     # Raw / processed data (not tracked by git)
├── reports/                      # Figures, CSV results, final report
├── requirements.txt
└── README.md
```

---

## Setup & usage

```bash
# 1. Clone the repo
git clone https://github.com/Novaaz/DL_Project.git
cd DL_Project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run supervised baseline
python src/train_supervised.py

# 4. Or open the notebooks in order
jupyter notebook experiments/
```

> **Note:** place your dataset inside `datasets/` following the structure expected by `src/datasets.py`.

---

## Tech stack

- Python 3.10+, PyTorch, torchvision
- ResNet18, ViT-B/16, SimCLR v2
- Jupyter Notebooks for experiments
