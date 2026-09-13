
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

## 🧠 Explainable AI (XAI) Analysis

To ensure absolute transparency and validate the mathematical hypotheses of this thesis, distinct Explainable AI (XAI) techniques were applied to each phase of the architecture to decode the "black box" routing logic.

### Exposing the Baseline Bias (Phase 2)
To prove the "Accuracy Paradox," we extracted the direct decision coefficients from the Phase 2 baseline model. The visual below demonstrates severe **vocabulary bias**: the model artificially inflates its 90% accuracy by blindly memorizing generic keywords (e.g., "system", "user") to predict the Functional class, which causes the starvation of complex minority classes.

<p align="center">
  <img src="Phase%201/images/XAI_Plots/Phase2_Vocabulary_Bias.png" alt="Vocabulary Bias" width="700">
</p>

### Decoding the Cascading Error (Phase 4-Simple)
To mathematically prove *why* the Cascading Drop occurs, we hooked directly into XGBoost's native internal XAI engine in Stage 1. This visualization maps the exact lexical thresholds that dictate FR vs NFR routing, exposing the precise features that trigger false routes and permanent cascading drops.

<p align="center">
  <img src="Phase%201/images/XAI_Plots/Phase4_Stage1_XAI.png" alt="Stage 1 XAI" width="600">
</p>

### Proving the Catch-Bin Rescue (Phase 4-New)
Using a counterfactual "What-If" visualization, we demonstrate the decision-boundary shift introduced by the Self-Correcting Catch-Bin. It visually proves how the architecture intercepts false positives dropping out of Stage 1 and miraculously flips the model's confidence back to the correct class in Stage 2.

<p align="center">
  <img src="Phase%201/images/XAI_Plots/Phase4_New_CatchBin_Rescue.png" alt="Catch-Bin Counterfactual" width="600">
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
