# Kaggle LLM Wiki

A structured knowledge base of Kaggle competition strategies, winning solutions, and ML techniques — designed to be queried by an LLM at the start of any competition to generate a domain-specific, competition-validated playbook instantly.

---

## What This Is

This wiki is the result of ingesting:
- **21 first-place solution writeups** from major Kaggle competitions (tabular, CV, NLP, financial, medical)
- **12 research reports** from systematic web research across competition domains
- **40 concept pages** synthesizing techniques across competitions
- **Competition-specific playbooks** for active and reference competitions

Every page cites primary sources and cross-links related pages. The wiki grows with each competition.

---

## Structure

```
Kaggle-llm-wiki/
│
├── CLAUDE.md              ← Schema contract (read this first)
│
├── raw/                   ← Immutable source material
│   └── kaggle/
│       ├── solutions/     ← 21 first-place writeups
│       └── *.md           ← Research reports (12 topic areas)
│
└── wiki/                  ← Living knowledge base (LLM-maintained)
    ├── index.md           ← Full content catalog
    ├── log.md             ← Chronological audit trail
    ├── overview.md        ← High-level knowledge map
    │
    ├── competitions/      ← Active & reference competition pages
    ├── concepts/          ← 40 reusable technique pages
    ├── strategies/        ← End-to-end competition playbooks
    └── entities/          ← Tools, frameworks, people
```

### The Three-Layer Architecture

```
raw/        ← Drop sources here. Never edit.
wiki/       ← LLM synthesizes raw sources into structured pages.
CLAUDE.md   ← Schema governing all operations.
```

---

## What's Inside

### 21 First-Place Solution Writeups

| Competition | Key Technique |
|---|---|
| Porto Seguro Safe Driver | Swap-noise denoising autoencoder + RankGauss |
| Optiver Realized Volatility | t-SNE time ordering + 360 nearest-neighbor features |
| Home Credit Default Risk | 1,800 features + KNN target mean + 3-level stacking (90+ models) |
| IEEE-CIS Fraud Detection | UID discovery + time consistency selection + NaN-pattern PCA |
| Jane Street Market Prediction | Supervised AE + purged 31-gap CV + sample weighting |
| Santander Transaction | Value uniqueness features + attention NN + shuffle augmentation |
| Optiver Trading at Close | CatBoost + GRU + Transformer + online retraining every 12 days |
| AmEx Default Prediction (1st) | RNN with pack_padded_sequence for sequential credit statements |
| AmEx Default (14th Gold) | LGBM→NN distillation + nested 10-in-10 K-fold + RAPIDS cuDF |
| ICR Age-Related Conditions | VSN + extreme dropout (0.75) + 30× repeat training |
| MoA Prediction | Non-scored targets as auxiliary signal + DeepInsight + EfficientNet |
| Jane Street 2025 (8th) | GRU online learning +0.008 + 1-step inference update |
| TalkingData Ad Fraud | LDA+NMF+SVD on click sequences → drop raw categoricals |
| Otto Group Product | 33-model L1 + geometric mean L3 + t-SNE/KMeans features |
| M5 Forecasting | LightGBM recursive horizon + Tweedie objective + hierarchical reconciliation |
| Human Protein Atlas | ArcFace metric learning + NN retrieval (+0.03 AUC) |
| Bengali Grapheme | CycleGAN zero-shot from TTF fonts + EfficientNet-b7 + OHEM |
| TGS Salt Identification | 3-stage pseudo-labeling + BCE→Lovász + scSE decoder |
| RSNA Aneurysm Detection | nnU-Net coarse-to-fine 3D + Location-Aware Transformer + FocalTversky++ |
| SIIM-ISIC Melanoma | 9-class diagnosis auxiliary target (+0.01 AUC) + rank ensembling |
| Playground S5E4 (2025) | RAPIDS cuML minimal stacking playbook + 3-level stack |

### 40 Concept Pages

**Tabular & Feature Engineering**
- `feature-engineering-tabular` — 5-stage process: domain → interactions → dates → target transforms → group aggregations
- `feature-selection-advanced` — 6-stage pipeline: correlation pruning → MI → adversarial validation → null importance → BorutaShap → LOFO
- `shap-feature-engineering` — SHAP dependency plots reveal needed transformations; interaction matrix identifies top pairs
- `target-encoding` — Weighted blend formula (k=6, √n), OOF implementation
- `target-encoding-advanced` — GLMM (benchmark winner), CatBoost ordered TS, James-Stein, entity embeddings
- `tabular-augmentation` — SCARF swap noise (ICLR 2022), SAINT CutMix+Mixup, TTA with feature noise, TabMDA
- `memory-optimization` — reduce_mem_usage (65-80%), Polars 10-15x, cuDF 150x GPU, parquet zstd caching
- `nearest-neighbor-features` — KNN target mean, 360 NN aggregation features
- `categorical-embeddings` — LDA+NMF+SVD on interaction sequences
- `text-feature-engineering` — Embeddings+PCA vs TF-IDF+SVD vs LLM-guided regex

**Validation & Leakage**
- `validation-strategy` — CV design, adversarial validation, CV-LB breakdown threshold, post-cutoff CV
- `external-data-leakage` — ID leakage scan, file metadata EXIF check, SMOTE-before-CV warning, leakage checklist
- `imbalanced-data` — scale_pos_weight + StratifiedKFold + threshold optimization + downsampling sandwich
- `time-series-cv` — Walk-forward, purged CV with embargo gap, post-cutoff CV

**Ensembling & Stacking**
- `ensembling-strategies` — Hill climbing, Ridge blending, Optuna weight optimization, hillclimbers library
- `stacking-deep` — 3-level architecture (GBDT → NN → Ridge), 90+ model diversity, OOF discipline
- `post-processing` — RankGauss, temperature scaling, probability clipping, rank blending
- `calibration` — Platt scaling, isotonic regression, when to calibrate

**Deep Learning (Tabular)**
- `deep-learning-tabular` — When DNNs beat GBMs; VSN; extreme dropout; DeepInsight; TabNet
- `tabpfn-tabm` — TabPFN (100% win rate ≤10K rows, Nature 2024); TabM (ICLR 2025, avg rank 1.7)
- `knowledge-distillation` — LGBM→NN soft labels; 4-cycle cosine; nested 10-in-10 K-fold
- `multi-target-learning` — MoA non-scored targets (+0.003); Ventilator physics auxiliaries; ordinal decomposition
- `denoising-autoencoders` — Swap noise, RankGauss, supervised AE, DAE as stacker

**Gradient Boosting**
- `gradient-boosting-advanced` — XGB dart/lossguide, LGB goss/linear_tree, CatBoost border_count, Optuna GPSampler
- `negative-downsampling` — Prior correction formula; 5-bag averaging; Porto Seguro pattern

**Time Series & Finance**
- `time-series-features` — Lag features, EWMA rolling stats, Fourier cyclical encoding
- `financial-competition-patterns` — Purged CV, WAP/OFI features, GRU+Transformer ensemble, t-SNE 360-NN trick
- `online-learning` — 12-day retraining, HDF5 incremental loading, GRU 1-step inference update

**Computer Vision**
- `medical-imaging-patterns` — CLAHE, 2.5D slice stacking, Phikon MIL, BirdCLEF pseudo-labeling, BELKA SMILES
- `image-augmentation` — Albumentations recipes, 8-way TTA, CycleGAN synthetic data, AutoAugment
- `image-classification-tricks` — AdaptiveConcatPool, diagnosis-as-target (+0.01 AUC), rank ensembling
- `loss-functions-cv` — BCE→Lovász progression, FocalLoss, Tversky++, SkeletonRecall
- `segmentation-architectures` — scSE decoder, FPA center block, nnU-Net, coarse-to-fine 3D
- `metric-learning-cv` — ArcFace loss, NN retrieval for label transfer (+0.03 AUC)
- `pseudo-labeling-cv` — 3-stage segmentation pipeline, ensemble agreement filter

**NLP / LLM**
- `llm-fine-tuning-kaggle` — AWP, LLRD, multisample dropout, two-stage training, LoRA/QLoRA
- `pseudo-labeling` — High-confidence thresholds, shuffle augmentation, K-fold discipline

**Meta**
- `universal-kaggle-tricks` — Cross-competition priority matrix from NVIDIA GM playbook + Neptune.ai (65+ comps) + KazAnova
- `external-data-leakage` — Comprehensive leakage prevention and detection checklist

### Strategy Pages
- `kaggle-meta-strategy` — Trust CV, CV-LB breakdown threshold, 2-week framework, shake-up survival
- `kaggle-competition-playbook` — End-to-end workflow: framing → EDA → features → models → ensembling

---

## How to Use This Wiki

### Querying (with an LLM)

Point an LLM at this repo and ask it to generate a competition plan:

```
"Read wiki/index.md and the relevant concept pages, then give me a 
competition strategy for [X-ray cancer classification / tabular fraud 
detection / time-series forecasting / etc.]"
```

The LLM will:
1. Scan `wiki/index.md` to identify relevant pages by tags and summaries
2. Pull the matching concept pages (e.g. `medical-imaging-patterns`, `imbalanced-data`, `loss-functions-cv`)
3. Synthesize a domain-specific, competition-validated playbook citing specific techniques and measured gains

### Example: X-Ray Cancer Detection

Starting a radiology competition, the LLM would pull:

| Page | What it contributes |
|---|---|
| `medical-imaging-patterns` | CLAHE preprocessing, 2.5D slice stacking, two-stage detect→classify |
| `loss-functions-cv` | FocalLoss gamma=2 for severe imbalance, Lovász for segmentation |
| `imbalanced-data` | Threshold sweep (optimal often 0.1-0.3 not 0.5), downsampling sandwich |
| `external-data-leakage` | X-ray EXIF metadata leak check (file size AUC scan on day 1) |
| `validation-strategy` | GroupKFold on patient_id (not image_id), adversarial validation |
| `image-augmentation` | Albumentations recipe, 8-way TTA, no vertical flip (anatomical) |
| `ensembling-strategies` | EfficientNet + ConvNeXt + ViT diversity + hill climbing weights |
| `pseudo-labeling-cv` | NIH ChestX-ray14 as external unlabeled source, confidence=0.90 |

Result: a concrete, measurable plan in seconds rather than hours of research.

---

## Adding to the Wiki

### INGEST a new source

1. Drop the raw file into `raw/kaggle/<slug>.md`
2. Identify which wiki pages to create or update
3. Write/update wiki page with YAML frontmatter + structured sections
4. Add entry to `wiki/index.md`
5. Append to `wiki/log.md`

### Page Frontmatter (required)

```yaml
---
title: "Human-readable title"
tags: [kaggle, ensemble, xgboost]
date: YYYY-MM-DD
source_count: 3
status: active | draft | archived
---
```

### Body Structure

```markdown
## Summary
1-3 sentence plain-English summary.

## Key Facts / Details
Tables, bullet lists, code snippets.

## What Worked / What Didn't
(competitions and strategies only)

## Sources
- [[../raw/kaggle/filename.md]] — description

## Related
- [[concepts/related-page]] — why it's related
```

See `CLAUDE.md` for the full schema.

---

## Key Cross-Links (Most Connected Pages)

These are the hub pages — reading them gives the widest coverage:

| Page | Inbound Links | Why It's Central |
|---|---|---|
| `validation-strategy` | 5 | Every technique needs a CV strategy |
| `gradient-boosting-advanced` | 4 | GBDTs appear in 80%+ of solutions |
| `feature-engineering-tabular` | 4 | FE is the highest-ROI activity |
| `ensembling-strategies` | 3 | All top solutions ensemble |
| `pseudo-labeling` | 3 | Semi-supervised signal appears everywhere |

---

## Stats

| Category | Count |
|---|---|
| Total wiki pages | 99 |
| Concept pages | 40 |
| Competition pages | 4 |
| Strategy pages | 4 |
| First-place solution writeups (raw) | 21 |
| Research reports (raw) | 12 |
| Competitions covered (tabular, CV, NLP, finance, medical) | 5 domains |
| Last updated | April 2026 |

---

## Sources & Credits

This wiki synthesizes publicly available information from:
- Kaggle competition discussion forums and winning writeups
- NVIDIA Grandmasters Playbook
- Neptune.ai competition analyses (65+ competitions)
- arXiv papers (TabPFN, TabM, SCARF, SAINT, TabMDA, SMOTE leakage, Pargent target encoding benchmark)
- KazAnova (Kaggle #3) tutorials
- ML Contests 2024 State of Competitive ML report
- Individual Grandmaster blog posts and notebooks

All techniques are cited with primary sources in each wiki page.
