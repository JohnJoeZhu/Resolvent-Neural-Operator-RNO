# Resolvent Neural Operator (RNO)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**Resolvent Neural Operator (RNO)** is a transformation-free deep learning framework for learning solution operators of partial differential equations (PDEs).

---

## Project Structure

```markdown
├── config/
├── data/                       # Datasets (download separately)
├── model/                      # Model implementations
│   ├── RNO/
├── utils/                      # Utility functions
├── logs/                       # Training logs and checkpoints
├── train.py                    # Training entry point
├── requirements.txt            # Dependencies
└── README.md
```

---

## Quick Start

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (recommended)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Data Preparation

All datasets and benchmark implementations used in this work are publicly available from their official repositories. Please download the datasets and place them under `data/`.

| Dataset | Source |
| :--- | :--- |
| Beam / Diffusion / Burgers |   [Laplace Neural Operator (GitHub)](https://github.com/qianyingcao/Laplace-Neural-Operator) |
| Darcy / Pipe / Composite / Heat Transfer | [NORM (GitHub)](https://github.com/gengxiangc/NORM) |
| Shape‑Net Car | [Transolver (GitHub)](https://github.com/thuml/Transolver) |

### Training

```bash
# Single GPU training
python train.py --task=<task_name> --local_rank=<gpu_id> --config=<config_name>.yaml
```

---

## Citation

To be added.