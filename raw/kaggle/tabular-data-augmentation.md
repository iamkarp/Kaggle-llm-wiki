# Tabular Data Augmentation for Kaggle Competitions

Compiled from arXiv papers (2020–2024), NeurIPS/ICLR/ICML proceedings, Kaggle community discussions. April 2026.

---

## Key Finding Upfront

Feature engineering dominates tabular competitions. Augmentation matters most when: (a) training data is small, (b) using neural net architectures, or (c) doing self-supervised pre-training. For GBDTs on large datasets, ROI on augmentation is lower than on feature engineering.

---

## 1. Mixup for Tabular

`x_new = lambda*x_i + (1-lambda)*x_j`, same for labels.

**When it helps:** Almost exclusively neural networks (MLPs, TabNet, FT-Transformer). Trees make discrete splits — smooth interpolation provides no geometric regularization benefit.

**Competition success:** MoA (Mechanisms of Action) Kaggle competition — mixup was used by top solutions.

**When it hurts:** Datasets with many categorical features — interpolating category codes produces semantically meaningless values.

**Variants:**
- **Manifold Mixup (ICML 2019):** Interpolation in hidden embedding layers. Research (arXiv 2305.10308) found it performs similarly to no augmentation for tabular data — not reliable.
- **RC-Mixup (KDD 2024):** Variant for noisy regression tasks on tabular data.
- **Learned Mixing Weights:** Architecture-agnostic method that learns per-feature mixing weights — improves multiple neural architectures.

---

## 2. Feature Noise Injection

Add `N(0, sigma)` noise to continuous features during training only.

**Research findings:** Provides regularization comparable to weight decay. Particularly effective for tabular neural nets on small datasets. Sigma should be proportional to each feature's std (noise as % of feature range, not absolute).

**Bonus:** Enables TTA at inference — add noise at test time and average predictions.

---

## 3. Swap Noise (SCARF, VIME, DAE)

Replace feature values with randomly sampled values from the same column from different rows. Corrupts with realistic values (unlike Gaussian noise).

**SCARF (ICLR 2022 Spotlight):** arXiv 2106.15147
- Corrupts a random 60% of features using swap noise, creates contrastive pairs.
- Pre-trains encoders using SimCLR-style contrastive loss.
- Evaluated on 69 OpenML-CC18 datasets — improves accuracy in fully-supervised, semi-supervised, and noisy-label settings.

**VIME (NeurIPS 2020):**
- Corruption + mask estimation pretext task. Encoder reconstructs original; classifier predicts which features were corrupted.
- Works as both pre-training augmentation and semi-supervised learning.

**Competition relevance:** Swap noise as pre-training for neural nets is the most evidence-backed tabular augmentation technique in the literature.

---

## 4. SAINT: CutMix + Embedding Mixup (2021)

Paper: arXiv 2106.01342

**Augmentation:**
- Input space: CutMix — combines features from two different samples (swaps entire features, not interpolates)
- Embedding space: Mixup — interpolates learned embeddings of two samples
- Both combined for self-supervised contrastive pre-training

**Results:** SAINT outperforms TabTransformer, XGBoost, CatBoost, and LightGBM on average across benchmark suite. CutMix + embedding Mixup together outperforms either alone.

**Practical note:** Only applies to transformer-based tabular architectures.

---

## 5. Test-Time Augmentation (TTA) for Tabular

At inference: create N slightly perturbed copies of each test sample, average predictions.

**Implementations:**
- Add small Gaussian noise to continuous features (sigma = small % of feature std), make K predictions, average
- Randomly zero out a fraction of features (dropout at inference), average
- Use different feature subsets and aggregate

**Competition use:** Free performance boost at end of competition — zero retraining required. Works with any model including GBDTs when predictions are probabilities.

---

## 6. TabMDA: Manifold Data Augmentation for Any Classifier (2024)

Paper: arXiv 2406.01805

**Key innovation:** Uses pre-trained in-context model (TabPFN) as encoder, maps samples to learned embedding space, performs label-invariant transformations by re-encoding with different context subsets. Decodes back to feature space.

**Why unique: Works as training-time augmentation for tree-based models (XGBoost, Random Forest)** — extremely rare.

**Results:** Consistently improves XGBoost and Random Forest performance. Makes KNN competitive with tree ensembles.

---

## 7. SubTab (NeurIPS 2021)

Divides features into multiple subsets, treats each as a different "view" of the same sample. Multi-view contrastive learning — learns representations that can reconstruct full sample from any subset.

**Results:** 98.31% on MNIST tabular, significant improvements on real-world datasets. Adding Gaussian noise as secondary augmentation further improves performance.

---

## 8. SMOTE Variants as General Augmentation

Key SMOTE family for augmentation (beyond imbalance fixing):
- **K-Means SMOTE:** Interpolates only in sparse areas, reducing noisy synthetic samples.
- **G-SMOTE:** Uses geometric regions (hyper-spheres) rather than linear interpolation.
- **SMOTE + WCGAN-GP hybrid:** Improvements shown in healthcare tasks (MDPI 2023).

**Core insight:** SMOTE-style interpolation within a class is a general augmentation strategy — interpolating between two samples of the same class along a straight line very likely stays in that class.

---

## 9. Synthetic Data Generation

- **CTAB-GAN+:** Outperforms baselines by 33.5% accuracy and 56.4% AUC in benchmarks.
- **TabDDPM:** Diffusion model for tabular — outperforms GANs in recent benchmarks.
- **TAPTAP / AIGT:** Use generative models + pseudo-labels to generate labeled synthetic data.

**Note:** Kaggle Tabular Playground Series was generated using CTGANs, so synthetic generation is validated at scale. Using generated data for training augmentation has mixed results — GAN quality must be high enough to not introduce distribution shift.

---

## Competition-Tested Summary

| Technique | Best Model | Evidence | Competition-Tested |
|---|---|---|---|
| Swap noise / SCARF pre-training | Neural nets | High (ICLR 2022) | Indirect |
| CutMix + embedding Mixup (SAINT) | Transformers | High | Indirect |
| Feature noise + TTA | Any | Medium-High | Yes (Kaggle community) |
| Input Mixup | Neural nets only | Medium (MoA) | Yes |
| TabMDA | Any (incl. XGBoost) | Medium (2024) | Not yet widespread |
| SMOTE variants | Any | Medium | Yes (imbalanced comps) |
| Pseudo-labeling | Any | High | Yes (many top solutions) |
| Manifold Mixup | Neural nets | Low | No |

---

Sources:
- arXiv 2305.10308 — Rethinking Data Augmentation for Tabular Data
- arXiv 2407.21523 — Tabular Data Augmentation survey
- arXiv 2106.01342 — SAINT
- arXiv 2106.15147 — SCARF (ICLR 2022)
- arXiv 2406.01805 — TabMDA (2024)
- arXiv 2110.04361 — SubTab (NeurIPS 2021)
- NeurIPS 2020 — VIME
- arXiv 2408.07579 — TabularBench (NeurIPS 2024)
- RC-Mixup KDD 2024: https://dl.acm.org/doi/10.1145/3637528.3671993
