# 3D Model Hashing System with Deep Neural Networks

This repository contains the implementation of a **deep learning-based hashing framework** for efficient **3D model retrieval** using **point clouds**.  
The system generates compact, similarity-preserving binary hash codes from 3D data, enabling fast and scalable retrieval.

---

## 📄 Overview

With the increasing adoption of 3D models in industries like CAD, VR, and medical imaging, fast and robust retrieval systems are essential.  
Our approach uses a **Siamese neural network** with PointNet-inspired encoders to learn **64-bit binary hash codes** directly from 3D point cloud representations.

Key features:
- **End-to-end deep learning pipeline** for 3D model hashing
- **Rotation-invariant preprocessing** (PCA alignment, normalization, augmentation)
- **Custom multi-objective loss** (similarity preservation, bit balance, bit independence)
- **Efficient retrieval system** using Hamming similarity

---

## 🗂 Dataset

We use the **[ModelNet10](https://modelnet.cs.princeton.edu/)** dataset, which contains 10 categories of CAD models.  
Each model is converted to a **point cloud with 1024 points** sampled using the [Trimesh](https://trimsh.org/) library.

Preprocessing steps:
1. **Centering** (align centroid to origin)
2. **PCA alignment** (principal axis rotation)
3. **Scaling** (normalize to unit sphere)
4. **Augmentation** (random rotations + Gaussian jittering)

---

## ⚙️ Methodology

### 1. Network Architecture
- Siamese network with shared PointNet-inspired encoders
- Point-wise MLP layers: `64 → 128 → 256 → 512`
- Global max + average pooling
- Fully connected layers: `512 → 256 → hash code`
- Skip connections for residual learning
- `tanh` activation for final hash layer

### 2. Loss Function
Multi-objective loss:
- **Contrastive loss** (cosine similarity-based)
- **Bit balance loss** (distribution around zero)
- **Bit independence loss** (minimize inter-bit correlation)

### 3. Retrieval System
- Precompute binary hash codes for all database models
- For each query:
  1. Preprocess point cloud
  2. Generate hash code
  3. Compute Hamming similarity with database entries
  4. Return top-k matches

---

## 📊 Results

Performance was evaluated in two ways:

1. **Trained Model Performance** (generating similarity-preserving hash codes):
   - Precision: **97%**
   - Recall: **86%**
   - F1-score: **91%**
   - Rotation invariance score: **0.97**

2. **Retrieval System Performance** (retrieving similar/dissimilar models):
   - Average cosine similarity (similar pairs): **0.9027**
   - Average cosine similarity (dissimilar pairs): **0.6839**

---

## 🚀 Installation

```bash
# Clone repository
git clone https://github.com/DularaWijerathne/3D-Model-Hashing-with-Deep-Neural-Network.git
cd <3D-Model-Hashing-with-Deep-Neural-Network>

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
