# MaizeBKN
This repository is used to store datasets and code related to the research of the paper "MaizeBKN: A Weakly-Supervised Boundary-Aware Keypoint Network for Precise Phenotyping of Maize Seedlings."
<div align="center">

# MaizeBKN
**A Weakly-Supervised Boundary-Aware Keypoint Network for Precise Phenotyping of Maize Seedlings**

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](你的论文链接)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/PyTorch-1.8+-ee4c2c.svg)](https://pytorch.org/)
[![Star](https://img.shields.io/github/stars/bbtwozzz/MaizeBKN?style=social)](https://github.com/bbtwozzz/MaizeBKN)

<img src="assets/teaser.png" width="800px">

</div>

---

## 📖 Introduction

This repository contains the official implementation and dataset for the paper **"MaizeBKN: A Weakly-Supervised Boundary-Aware Keypoint Network for Precise Phenotyping of Maize Seedlings"**.

**MaizeBKN** is a lightweight yet powerful keypoint detection network designed for agricultural edge devices. It achieves state-of-the-art accuracy with minimal computational cost.

### ✨ Key Features
- **🚀 Ultra-Lightweight:** Only **0.70M** parameters (38% lower than Lite-HRNet) and **1.13 GFLOPs**.
- **🎯 High Precision:** Achieves **98.89% AP** and **99.60% AR** on the MaizeSeedling-Calib dataset.
- **📏 Boundary-Aware:** Introduces a weakly-supervised boundary mining mechanism to handle blurry boundaries and tiny organs.
- **🌽 Phenotyping System:** Includes a pipeline for extracting phenotypic traits like leaf sheath length and mesocotyl length.

---

## 🔥 Updates
* **[2026-01-08]** Code and partial dataset are released!
* **[Date]** Paper accepted by [Journal Name].

---

## 🛠️ Installation

### Requirements
* Linux or Windows
* Python 3.8+
* PyTorch ≥ 1.8.0
* CUDA ≥ 11.0

### Setup
```bash
# Clone the repository
git clone [https://github.com/bbtwozzz/MaizeBKN.git](https://github.com/bbtwozzz/MaizeBKN.git)
cd MaizeBKN

# Install dependencies
pip install -r requirements.txt
