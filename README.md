# MaizeBKN
**A Weakly-Supervised Boundary-Aware Keypoint Network for Precise Phenotyping of Maize Seedlings**


<img src="assets/MaizeBKN.png" width="1280px">

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
*  Windows
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
