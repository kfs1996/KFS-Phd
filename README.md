
<div align="center">
  <h1>Cost-Sensitive Hybrid NLP Architecture for Software Requirements Classification</h1>
  <p><strong>A Self-Correcting Hierarchical Approach to Neutralizing Majority Class Bias in Highly Imbalanced Datasets</strong></p>
</div>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--Learn%20%7C%20XGBoost-orange" alt="ML">
  <img src="https://img.shields.io/badge/Hyperparameter%20Optimization-Optuna-brightgreen" alt="Optuna">
  <img src="https://img.shields.io/badge/Status-PhD%20Thesis%20Code-purple" alt="Status">
</p>

---

## 📖 Overview

This repository contains the complete experimental codebase for researching and proving the efficacy of a **Self-Correcting Hierarchical NLP Pipeline** designed for Software Requirements Classification.

In modern Requirements Engineering, datasets are massively imbalanced—Functional Requirements (FR) vastly outnumber Non-Functional Requirements (NFR) such as Security, Performance, and Fault Tolerance. This creates an **Accuracy Paradox**: standard flat classifiers can achieve 90%+ global accuracy simply by memorizing the majority class, completely sacrificing the detection of critical minority NFRs.

This thesis mathematically proves that while standard 2-Stage Hierarchies fail due to **Cascading Errors**, a **Self-Correcting Hierarchy with a Catch-Bin** can perfectly restore global accuracy while maintaining strict, unbiased fairness for minority classes.

---

## 🔬 Visualizing the Methodology

### 1. The Accuracy Paradox (Phase 2-C)
Standard Flat Classifiers overfit the majority FR class. A flat `LinearSVC` achieves **90.35% global accuracy**, but close inspection of its class-wise metrics reveals it scores **0.00 Precision/Recall** on minority classes like Fault Tolerance. 

<p align="center">
  <img src="Phase%201/images/plot_3_minority.png" alt="Accuracy Paradox" width="800">
</p>

### 2. The Cascading Error & Catch-Bin Recovery
To force the model to identify minority classes, a standard 2-Stage hierarchy is used. However, any NFR falsely predicted as an FR in Stage 1 is permanently dropped. This "Cascading Drop" bleeds the architecture's accuracy down to **86.78%**.

Our **Adaptive Self-Correcting Hierarchy** introduces a Catch-Bin. If Stage 2 determines a routed requirement was actually an FR, the Catch-Bin rescues it. **Result:** The Self-Correcting Hierarchy successfully restores accuracy back to **90.16%**!

<p align="center">
  <img src="Phase%201/images/plot_1_recovery.png" alt="Catch-Bin Recovery" width="600">
</p>

### 3. The Extreme Rescue Case (Word2Vec)
In weaker embeddings like Word2Vec, the Cascading Drop is catastrophic (crashing to 57.39%). The Catch-Bin flawlessly rescues the architecture, restoring it to **85.37%**.

<p align="center">
  <img src="Phase%201/images/plot_2_extreme.png" alt="Extreme Rescue" width="600">
</p>

---

## 📂 Repository Structure (Phase 1)
| Directory | Description |
| :--- | :--- |
| 📁 **`Phase_2_Baselines/`** | Baseline 1-Stage flat classifiers. |
| 📁 **`Phase_4_Simple_Hierarchies/`** | Classical 2-Stage Hierarchies (Cascading Drop). |
| 📁 **`Phase_4_NEW_Self_Correcting_Hierarchies/`** | **(The Proposed Architecture)** Advanced codebase featuring the 5x Paranoid Penalty and the dynamic Catch-Bin. |
| 📁 **`Analysis_and_Comparisons/`** | Scripts for class-wise metrics and Ultimate Thesis Tables. |

---
<div align="center">
  <i>Defended as part of PhD Research in Software Engineering & Machine Learning.</i>
</div>
