# DSCAN: Dual-Stream Cross-Attention Network for AKI Prediction

![Python Version](https://img.shields.io/badge/python-3.12.2-blue.svg)
![PyTorch Version](https://img.shields.io/badge/pytorch-2.7.0%2Bcu128-orange.svg)
![CUDA Version](https://img.shields.io/badge/CUDA-12.8-green.svg)

## 📌 Introduction
This project is the official implementation of the **DSCAN** model, specifically designed for predicting **Acute Kidney Injury (AKI)** in clinical medical scenarios.

The model adopts a dual-stream architecture:
- **Dynamic Stream**: Extracts features from patient physiological indicators over time based on a Transformer.
- **Static Stream**: Processes fixed characteristics such as patient demographics and medical history via an MLP.
- **Feature Fusion**: Deeply integrates temporal and static features using a Gated Fusion mechanism.

---

## 🛠 Prerequisites

Please ensure your development environment meets the following version requirements:
* **Python**: `3.12.2`
* **PyTorch**: `2.7.0 + cu128`
* **CUDA**: `12.8`

### Installation
Run the following command in the project root directory to complete the environment setup:
```bash
pip install -r requirements.txt
