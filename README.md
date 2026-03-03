# MaizeBKN

<img src="assets/MaizeBKN.png" width="100%">
</div>

## 📖 Introduction

**MaizeBKN** is a specialized keypoint detection model designed to overcome the inefficiency and subjectivity of traditional manual measurements in maize breeding. It focuses on the rapid and precise extraction of critical agronomic traits—specifically **leaf sheath** and **mesocotyl length**—which are vital indicators for high-density planting potential and lodging resistance.

Built upon a streamlined **Lite-HRNet** backbone, MaizeBKN achieves a superior balance between lightweight deployment and feature representation through channel pruning and module reorganization. Crucially, it introduces an innovative **boundary feature enhancement mechanism** guided by **pseudo-label weak supervision** and **morphological priors**. This allows the network to adaptively strengthen feature responses at the subtle structural connections of seedlings, ensuring high robustness even in complex environments.

### ✨ Key Features

- **🚀 Ultra-Lightweight Design:** Through channel pruning and reorganization, the model reduces parameters by **38.1%** and computational complexity by **5%** compared to the benchmark, making it ideal for high-throughput phenotyping tasks.
- **🎯 State-of-the-Art Precision:** Achieves an Average Precision (**AP**) of **98.89%** and an average confidence score of **87.15%** on the standardized **MaizeSeedling-Calib** dataset.
- **📏 High-Resolution Phenotyping:** The average measurement error for phenotypic parameters is **less than 2 mm**, providing millimetric accuracy that rivals or surpasses manual measurement.
- **🔍 Weakly-Supervised Boundary Awareness:** Novel integration of a boundary detector with **pseudo-labels**, allowing the model to focus on hard-to-distinguish organ boundaries without requiring pixel-level semantic segmentation labels.
- **🧬 Practical Breeding Value:** Validated in real-world scenarios, the model successfully quantified **shade avoidance responses** induced by low light, proving its effectiveness in screening excellent germplasm for dense planting.
---

## 📂 MaizeSeeding-Calib

We provide the **MaizeSeedling-Calib** dataset, a millimeter-calibrated dataset for maize seedling phenotyping.
<img src="assets/Data_show.png" width="100%">

 

## 📊 Model Zoo & Results
###  Results of comparative experiments
| Model | Params (M)  | GFLOPs  | AP (%)  | AR (%)  | Score  | PCK@0.05  | PCK@0.1  | NME  |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| HRNetV1 | 28.54 | 40.89 | 98.53 | 99.25 | 86.03 | 87.92 | 96.18 | 3.78 |
| Lite-HRNet | 1.13 | 1.19 | 98.67 | 99.41 | 82.78 | 83.15 | 95.34 | 3.67 |
| ViTPose | 89.84 | 116.62 | 97.81 | 98.77 | 87.14 | 86.42 | 95.92 | 3.84 |
| EfficientNetV2-S | 20.77 | 15.31 | 84.27 | 86.87 | 33.87 | 9.62 | 36.27 | 15.44 |
| MobileNetV3 | 2.90 | 24.47 | 98.46 | 99.23 | 84.75 | 80.47 | 94.35 | 4.15 |
| ShuffleNetV2-x1.0 | 1.52 | 0.86 | 87.44 | 89.38 | 37.70 | 9.40 | 38.33 | 14.07 |
| **MaizeBKN (Ours)** | **0.70** | **1.13** | **98.89** | **99.60** | **87.15** | **99.12** | **99.67** | **0.76** |

## 🖼️ Visualization
###  Visual comparison of keypoint detection performance across different models
<img src="assets/Visual_comparison.png" width="100%">

## ⚠️ Note
#### As the project is still in progress, this repository currently contains a portion of the dataset. The complete dataset will be released after the project concludes.

