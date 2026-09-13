
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

## 🔬 The Core Methodology

### 1. The Baseline Problem (Phase 2-C)
Standard Flat Classifiers overfit the majority FR class. A flat `LinearSVC` with `TF-IDF` vectors achieves **90.35% global accuracy**, but close inspection of its class-wise metrics reveals it scores **0.00 Precision/Recall** on minority classes like Fault Tolerance and Scalability. 

### 2. The Cascading Error (Phase 4-Simple)
To force the model to identify minority classes, a standard 2-Stage hierarchy is used (Stage 1 separates FR vs. NFR, Stage 2 classifies specific NFRs). However, any NFR falsely predicted as an FR in Stage 1 is permanently dropped. This "Cascading Drop" bleeds the architecture's accuracy down to **86.78%**.

### 3. The Solution: Adaptive Self-Correcting Hierarchy (Phase 4-New)
Our proposed architecture introduces two novel mechanics to solve the hierarchy bleed:
1. **The 5.0x Paranoid Multiplier:** A dynamic cost-sensitive penalty injected into the Optuna optimizer forcing Stage 1 to heavily penalize dropping NFRs.
2. **The FR Catch-Bin:** Stage 2 is dynamically expanded to include a Catch-Bin. If Stage 1 is unsure, it pushes the requirement to Stage 2. If Stage 2 determines it was actually an FR, the Catch-Bin rescues it and corrects the prediction. 

**Result:** The Self-Correcting Hierarchy successfully restores accuracy back to **90.16%** while protecting the minority classes—achieving both Fairness and Accuracy.

---

## 📂 Repository Structure (Phase 1)

All experimental data, tuning scripts, and mathematical proofs are organized under the `Phase 1` directory:

| Directory | Description |
| :--- | :--- |
| 📁 **`Phase_2_Baselines/`** | Contains the baseline 1-Stage flat classifiers. Used to demonstrate the "Accuracy Paradox" and minority class starvation. |
| 📁 **`Phase_4_Simple_Hierarchies/`** | Contains classical 2-Stage Hierarchies. Used to demonstrate the "Cascading Error" drop in global accuracy. |
| 📁 **`Phase_4_NEW_Self_Correcting_Hierarchies/`** | **(The Proposed Architecture)** Contains the advanced codebase featuring the 5x Paranoid Penalty and the dynamic Stage-2 Catch-Bin. |
| 📁 **`Analysis_and_Comparisons/`** | Contains scripts that extract detailed class-wise metrics (Precision/Recall) and merge all outputs into the Ultimate Thesis Tables. |

---

## 📊 Key Findings

The mathematical proof of the thesis, tested across 6 different text vectorization techniques (TF-IDF, Word2Vec, GloVe, BERT, SBERT, MPNet) and 6 classification algorithms on the FNFC Dataset.

| Vectorization | Algorithm | Phase 2-C <br>(Biased Flat) | Phase 4-2C <br>(Cascading Drop) | Phase 4-2C-NEW <br>(Self-Correcting) |
| :--- | :--- | :--- | :--- | :--- |
| **TF-IDF** | **LinearSVC** | 90.35% | *86.78%* | **90.16%** |
| **MPNet** | **XGBoost** | 89.09% | *88.47%* | **89.14%** |
| **Word2Vec** | **LinearSVC** | 85.41% | *57.39%* | **85.37%** |

*Notice the severe accuracy drop in the Simple Hierarchy (Phase 4-2C) and the near-perfect recovery by the Catch-Bin (Phase 4-2C-NEW).*

---
<div align="center">
  <i>Defended as part of PhD Research in Software Engineering & Machine Learning.</i>
</div>
