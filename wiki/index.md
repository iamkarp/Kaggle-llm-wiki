---
title: "Wiki Index"
tags: [meta]
date: 2026-04-16
source_count: 0
status: active
---

# Wiki Index

Content catalog — every page listed with link and summary. Updated on every create/update/archive operation.

---

## Meta

| Page | Tags | Status | Summary |
|------|------|--------|---------|
| [[overview]] | meta | active | High-level map of the knowledge base — domain areas, current status, how to use |
| [[log]] | meta | active | Chronological audit trail of all wiki operations |

---

## Competitions

| Page | Tags | Status | Summary |
|------|------|--------|---------|
| [[competitions/march-mania-2026]] | kaggle, ncaa, ensemble, xgboost | active | March Mania 2026 — best score 0.02210 Stage 1 with v6 weighted ensemble; Stage 2 submitted |
| [[competitions/autopilot-vqa-2026]] | kaggle, cvpr, vqa, vision, qwen, claude | active | AUTOPILOT VQA — 25-column dashcam classification; deadline April 15, 2026; 3-stage VLM pipeline |
| [[competitions/horse-health-prediction]] | kaggle, tabular, multiclass, f1, ensemble | active | Horse Health Playground — 988×29 multiclass (lived/died/euthanized); OOF micro-F1 0.71862 with seed-avg LGB+XGB+CAT |
| [[competitions/playground-s5-s6]] | kaggle, playground, seed-averaging, smote-leakage, tabular | active | Playground S5/S6 2025-2026 — GPU hill climbing, 100-seed +0.003, SMOTE leakage case study, LLM-augmented features |
| [[competitions/stock-return-prediction]] | kaggle, tabular, regression, financial, rmse, cross-sectional | active | 1-Year US Stock Return Prediction — RMSE on fat-tailed returns; GroupKFold beat expanding window; Huber delta scaling; best LB 15217 |

---

## 1st-Place Solution References

*Summaries of external winning solutions ingested as reference material. Full details in `raw/kaggle/solutions/`.*

| Page | Competition | Key Technique |
|------|------------|---------------|
| `raw/kaggle/solutions/porto-seguro-1st-jahrer.md` | Porto Seguro (insurance) | Swap-noise DAE + RankGauss + 6-model blend |
| `raw/kaggle/solutions/optiver-volatility-1st-nyanp.md` | Optiver Volatility | t-SNE time ordering + 360 NN aggregation features |
| `raw/kaggle/solutions/home-credit-1st-tunguz.md` | Home Credit Default Risk | 1800 features, KNN target mean, 3-level stacking (90+ models) |
| `raw/kaggle/solutions/ieee-fraud-1st-chris-deotte.md` | IEEE Fraud Detection | UID discovery, time consistency selection, NaN-pattern PCA |
| `raw/kaggle/solutions/jane-street-1st-yirun-zhang.md` | Jane Street Market Prediction | Supervised AE, purged 31-gap CV, Swish, sample weighting |
| `raw/kaggle/solutions/santander-transaction-1st-fl2o.md` | Santander Transaction | Value uniqueness features, attention NN, pseudo-labeling, shuffle augmentation |
| `raw/kaggle/solutions/optiver-close-1st-hyd.md` | Optiver Trading at the Close (2024) | CatBoost+GRU+Transformer blend; online retraining every 12 days; rank features |
| `raw/kaggle/solutions/amex-default-1st-correlation.md` | AmEx Default Prediction (2022) | LGB + RNN (pack_padded_sequence); sequential credit statement modeling |
| `raw/kaggle/solutions/amex-default-14th-chris-deotte.md` | AmEx Default 14th Gold (2022) | LGBM→NN distillation; 4-cycle cosine; nested 10-in-10 K-fold; RAPIDS cuDF |
| `raw/kaggle/solutions/icr-age-conditions-1st-room722.md` | ICR Age-Related Conditions (2023) | VSN; extreme dropout (0.75); repeat 30×, keep 2; hardness-stratified CV |
| `raw/kaggle/solutions/moa-1st-mark-peng.md` | MoA Prediction (2020) | Non-scored targets as meta-features; DeepInsight+EfficientNet; 7-model blend |
| `raw/kaggle/solutions/jane-street-2025-8th-grigoreva.md` | Jane Street 2025 (8th, 290 votes) | GRU online learning +0.008; 1-step inference update on auxiliary targets |
| `raw/kaggle/solutions/talkingdata-fraud-1st-komaki.md` | TalkingData Ad Fraud (2018) | LDA+NMF+SVD on click sequences; drop raw cats after embedding; 5-bag downsampling |
| `raw/kaggle/solutions/otto-group-1st-giba-semenov.md` | Otto Group (2015, 499 votes) | Classic 33-model L1; geometric mean L3; KNN multi-K; t-SNE features; diversity principle |
| `raw/kaggle/solutions/m5-forecasting-1st-yeonjun.md` | M5 Forecasting (2020) | LightGBM recursive horizon features; Tweedie objective; hierarchical reconciliation |
| `raw/kaggle/solutions/playground-s5e4-1st-chris-deotte.md` | Playground S5E4 (2025) | RAPIDS cuML minimal stacking playbook; original dataset inclusion |
| `raw/kaggle/solutions/human-protein-atlas-1st-bestfitting.md` | Human Protein Atlas (2019) | ArcFace metric learning + NN retrieval (+0.03); AdaptiveConcatPool2d; FocalLoss + Lovász |
| `raw/kaggle/solutions/bengali-grapheme-1st-deoxy.md` | Bengali Grapheme (2020) | CycleGAN zero-shot from TTF fonts; EfficientNet-b7; AutoAugment SVHN; OHEM |
| `raw/kaggle/solutions/tgs-salt-1st-babakhin.md` | TGS Salt Identification (2018) | 3-stage pseudo-labeling; BCE→Lovász; scSE decoder; FPA center block; snapshot ensemble |
| `raw/kaggle/solutions/rsna-aneurysm-1st-tomoon33.md` | RSNA Aneurysm Detection (2025) | nnU-Net coarse-to-fine 3D; Location-Aware Transformer; SkeletonRecall loss; FocalTversky++ |
| `raw/kaggle/solutions/siim-isic-melanoma-1st-bo.md` | SIIM-ISIC Melanoma (2020) | 9-class diagnosis CE (+0.01 AUC); rank ensembling; multi-resolution; metadata fusion |
| `raw/kaggle/solutions/tabular-classics-batch.md` | Rossmann 1st + Crowdflower 1st + Homesite 1st | Entity embeddings (pioneered); distribution-based QWK decoding; StackNet 3-level stacking |
| `raw/kaggle/solutions/cv-segmentation-batch.md` | Clouds 1st + Severstal 1st + Kuzushiji 1st + Recursion 4th + APTOS 1st | UNet+FPN dual; Cascade R-CNN+HRNet; GeM pooling; channel subset ensembling |
| `raw/kaggle/solutions/nlp-audio-medical-batch.md` | Jigsaw 1st + CHAMPS 1st + Freesound 1st + SIIM 5th + RSNA 2nd | Bias-aware loss; MPNN molecular graphs; SpecAugment; CT 3-window; deep supervision |
| `raw/kaggle/solutions/code-repo-solutions-batch.md` | Diabetic Retinopathy 1st + DSB 2017 1st + Avazu CTR 1st + NDSB 1st + Tradeshift 1st + Amazon Employee 1st + Avito 1st | SparseConvNet; 3D Faster R-CNN; FFM; cyclic pooling; Owen Zhang features |
| `raw/kaggle/solutions/mixed-tier-solutions-batch.md` | Instant Gratification 1st + YouTube-8M 1st + Generative Dogs 1st + 7 more | QDA; NeXtVLAD+MoE; BigGAN truncation; keypoint alignment; PointPillars BEV; multi-level stacking |
| `raw/kaggle/solutions/historical-interviews-batch.md` | Give Me Credit 1st + Rossmann 2nd/3rd + Prudential 2nd + 10 more | Early ensembling; entity embeddings; feature selection; physics-informed features; zero-shot GAN |
| `raw/kaggle/solutions/srk-batch-1.md` | Toxic Comment 1st + Web Traffic 1st + H&M RecSys 1st + 11 more | Multi-embedding LSTM; seq2seq RNN; retrieve-then-rank; capsule networks; EfficientNet frame-level |
| `raw/kaggle/solutions/srk-batch-2.md` | OpenVaccine 1st + Feedback Prize 1st + SETI 1st + 12 more | RNA graph features; ARC program synthesis; Mercari NLP+FM ridge; LLM prompt recovery; PANDA pathology |
| `raw/kaggle/solutions/srk-batch-3.md` | BirdCLEF+ 2025 1st + LANL Earthquake 1st + PLAsTiCC 1st + 12 more | SpecAugment+Mixup audio; Bayesian LGBM; OTTO multi-stage recsys; LMSYS arena preference |
| `raw/kaggle/solutions/srk-batch-4.md` | Favorita Grocery 1st + AIMO Prize 1st + ASL Signs 1st + 12 more | Seq2seq demand; symbolic math solvers; landmark DELG retrieval; Enefit weather features |
| `raw/kaggle/solutions/srk-batch-5.md` | chaii QA 1st + Eedi 1st + Riiid 1st + 12 more | Multilingual QA; knowledge tracing SAKT; sleep event detection; Universal Image Embedding |
| `raw/kaggle/solutions/srk-batch-6.md` | TF QA 1st + RSNA PE 1st + Indoor Nav 1st + 12 more | Long-context NLP; 3-stage CT; WiFi fingerprinting; RNA ribonanza; RL Halite |
| `raw/kaggle/solutions/srk-batch-7.md` | HuBMAP Kidney 1st + CSIRO Biomass 1st + BirdCLEF 2023 1st + 12 more | Adaptive tiling; biomass regression; NER entity linking; KKBox survival; CryoET 3D detection |
| `raw/kaggle/solutions/srk-batch-8.md` | Open Polymer 2025 1st + HMS Brain 1st + Vesuvius Ink 1st + 12 more | GNN SMILES; dual EEG+spectrogram; 3D ink segmentation; TrackML particle tracking; G2Net GW |
| `raw/kaggle/solutions/srk-batch-9.md` | PII Detection 1st + AMP Parkinson's 1st + Google Football 1st + 15 more | DeBERTa NER ensemble; RL self-play; image matching SuperGlue; whale identification; GAN detection |
| `raw/kaggle/solutions/srk-batch-10.md` | BELKA Drug 1st + YouTube-8M 1st + HuBMAP Vasculature 1st + 15 more | Molecular GNN; video NetVLAD; HPA cell segmentation; RSNA breast screening; Mayo pathology |
| `raw/kaggle/solutions/srk-batch-11.md` | Image Matching 2025 1st + UBC Ovarian 1st + Facebook Check-Ins 1st + 15 more | LoFTR/SuperGlue; ovarian pathology MIL; check-in KNN; LEAP force fields; single-cell perturbation |
| `raw/kaggle/solutions/srk-batch-12.md` | Kore 2022 1st + Hotel-ID 1st + Airbus Ship 1st + 15 more | Halite RL; hotel FGVC retrieval; ship U-Net; Lyft 3D detection; wheat global detection |
| `raw/kaggle/solutions/srk-batch-13.md` | PlantTraits 2024 1st + Inclusive Images 1st + Avito Demand 1st + 15 more | Plant trait regression; domain adaptation CV; demand FM stacking; malware prediction |
| `raw/kaggle/solutions/srk-batch-14.md` | Yelp Photos 1st + Flavours of Physics 1st + ECML Taxi 1st + 13 more | Photo classification CNN; physics-aware ML; taxi route prediction; flight optimization |
| `raw/kaggle/ndres-past-solutions-catalog.md` | Master catalog (55 competitions) | Full index from ndres.me/kaggle-past-solutions with all solution links and code repos |

### Non-1st-Place Solutions (100+ upvotes)

*SRK Round 2: 137 solutions across 82 competitions, ranked 2nd–40th place. Full writeups in `raw/kaggle/solutions/`.*

| Page | Solutions Covered | Key Techniques |
|------|------------------|----------------|
| `raw/kaggle/solutions/srk-r2-batch-1.md` | IEEE Fraud 2nd + Home Credit 2nd + HMS Brain 1st + 22 more | User-ID construction; DAE tabular features; multi-spectrogram EEG; ArcFace retrieval |
| `raw/kaggle/solutions/srk-r2-batch-2.md` | Toxic Comment 2nd + Avito 3rd + RSNA Hemorrhage 2nd + 22 more | Translation TDA; DPCNN; CNN→LSTM CT pipeline; delta embeddings; video-ID CV |
| `raw/kaggle/solutions/srk-r2-batch-3.md` | Great Barrier Reef 3rd/5th + DFL Bundesliga 1st + Shopee 6th + 20 more | CenterNet on DeepLabV3+; TSM; ArcFace+DistilBERT; two-stage PE detection; symmetric Lovász |
| `raw/kaggle/solutions/srk-r2-batch-4.md` | Avito 4th + Home Credit 5th + RSNA Mammography + 20 more | Text-cluster price agg; user image CNN+BiLSTM; landmarks-as-spectrogram; translation TTA |
| `raw/kaggle/solutions/srk-r2-batch-5.md` | ELL 2nd + BirdCLEF 2021 1st + AIMO Prize 2 2nd + 20 more | Rank loss; WaveNet seq2seq; DPO length reduction; W4KV8 quantization; trainable frontend |
| `raw/kaggle/solutions/srk-r2-batch-6.md` | NFL Impact 2nd + Jigsaw Multilingual 4th + PLAsTiCC 4th + 19 more | Temporal stacking; progressive language fine-tuning; Bazin curve fitting; deep mutual learning |

### ndres.me Extended Catalog

| Page | Content | Coverage |
|------|---------|----------|
| `raw/kaggle/solutions/ndres-catalog.md` | 63 competitions, 227 solution writeups, 49 code repos | Full ndres.me/kaggle-past-solutions catalog with links and tags |
| `raw/kaggle/solutions/code-repo-solutions-ndres.md` | 49 GitHub repositories from winning solutions | Code repos indexed by competition |

---

## Strategies

| Page | Tags | Status | Summary |
|------|------|--------|---------|
| [[strategies/kaggle-competition-playbook]] | kaggle, playbook, workflow, eda, ensemble | active | End-to-end playbook: framing → EDA → features → models → ensembling → validation |
| [[strategies/kaggle-meta-strategy]] | kaggle, meta, strategy, cv, shake-up, grandmaster | active | Grandmaster principles: trust CV, CV-LB breakdown threshold, 2-week framework, leakage playbook |
| [[strategies/march-mania-v6-ensemble]] | ensemble, xgboost, lightgbm, calibration | active | v6 weighted ensemble (35% v2.9 Elite + 35% v2.8 Advanced + 30% v5 Hybrid); best 0.02210 LB |
| [[strategies/nfp-straddle-forex]] | forex, trading, straddle, oanda | active | NFP straddle on USD_JPY/EUR_USD/GBP_USD; 4-pip stop + 5-pip trail; 25 units; live OANDA |
| [[strategies/autopilot-vqa-pipeline]] | kaggle, cvpr, vqa, vision, pipeline | draft | 3-stage VLM pipeline: Qwen-VL scene description → Claude Sonnet JSON extraction → post-processing |

---

## Concepts

| Page | Tags | Status | Summary |
|------|------|--------|---------|
| [[concepts/xgboost-ensembles]] | xgboost, ensemble, gradient-boosting | active | Patterns for ensembling XGBoost at multiple depths; hyperparameter guide |
| [[concepts/ensembling-strategies]] | ensemble, stacking, fourth-root, oof, diversity | active | Fourth-root weighted blend, stacking with Ridge, diversity via OOF correlation |
| [[concepts/stacking-deep]] | stacking, multi-level, oof, geometric-mean, otto, auxiliary-targets | active | Otto 33-model L1 blueprint; geometric mean L3; MoA auxiliary targets; Home Credit 90+ models |
| [[concepts/calibration]] | calibration, probabilities, platt-scaling | active | Platt scaling and isotonic regression for probability calibration; when and how to apply |
| [[concepts/denoising-autoencoders]] | dae, autoencoder, swap-noise, rankgauss, tabular | active | Swap noise (not Gaussian), RankGauss, standard + supervised AE architectures |
| [[concepts/feature-engineering-tabular]] | feature-engineering, tabular, basketball, interactions | active | 5-stage process: domain → interactions → dates → target transforms → group aggregations |
| [[concepts/feature-selection]] | feature-selection, time-consistency, forward-selection, permutation | active | Time consistency, Ridge forward selection (1600→240), permutation importance, NaN-PCA |
| [[concepts/nearest-neighbor-features]] | knn, nearest-neighbor, aggregation, target-encoding | active | KNN target mean (Home Credit top feature); 360 NN aggregation features (Optiver) |
| [[concepts/pseudo-labeling]] | pseudo-labeling, semi-supervised, confidence-threshold | active | High-confidence test pseudo-labels; shuffle augmentation; when it helps vs. hurts |
| [[concepts/target-encoding]] | target-encoding, categorical, oof, leakage | active | Weighted blend formula (k=6, sqrt(n)); OOF implementation to prevent leakage |
| [[concepts/text-feature-engineering]] | nlp, text, embeddings, tfidf, regex, llm | active | Three strategies: embeddings+PCA, TF-IDF+SVD, LLM-guided regex; when to use each |
| [[concepts/categorical-embeddings]] | lda, nmf, pca, topic-models, high-cardinality, fraud | active | LDA+NMF+SVD on interaction sequences; drop raw cats after embedding; 0.9821→0.9828 |
| [[concepts/deep-learning-tabular]] | dnn, vsn, dropout, deepinsight, tabnet, tabular | active | When DNNs beat GBMs; VSN; extreme dropout (0.75); DeepInsight; TabNet; repeated training |
| [[concepts/knowledge-distillation]] | distillation, soft-labels, lgbm, cosine-schedule, rapids | active | LGBM→NN soft labels; 4-cycle cosine; nested 10-in-10 K-fold; RAPIDS cuDF 10–100× speedup |
| [[concepts/negative-downsampling]] | downsampling, imbalanced, fraud, calibration, bagging | active | 99.8% negative discard; prior correction formula; 5-bag averaging; when vs. class weights |
| [[concepts/online-learning]] | online-learning, retraining, hdf5, gru, auxiliary-targets | active | 12-day retraining; GRU +0.008 from 1-step inference update; HDF5 loading; rank features |
| [[concepts/validation-strategy]] | validation, cv, adversarial, gap-tracking, oof | active | CV design, adversarial validation, gap tracking, final submission selection |
| [[concepts/metric-learning-cv]] | cv, arcface, embedding, retrieval, label-transfer | active | ArcFace loss, cosine similarity NN retrieval, +0.03 from label transfer for rare classes |
| [[concepts/pseudo-labeling-cv]] | cv, pseudo-labeling, segmentation, confidence-threshold | active | 3-stage segmentation pipeline, ensemble agreement filter, BCE→Lovász progression |
| [[concepts/image-augmentation]] | cv, albumentations, tta, cyclegan, autoaugment | active | Albumentations recipes, 8-way TTA, CycleGAN synthetic data, AutoAugment policy transfer |
| [[concepts/loss-functions-cv]] | cv, focal-loss, lovasz, dice, tversky, skeleton-recall | active | BCE→Lovász progression, FocalLoss, Tversky++, SkeletonRecall for tubular structures |
| [[concepts/segmentation-architectures]] | cv, unet, scse, fpa, nnunet, coarse-to-fine | active | scSE decoder, FPA center block, nnU-Net auto-config, coarse-to-fine 3D pipeline |
| [[concepts/image-classification-tricks]] | cv, efficientnet, pooling, rank-ensembling, metadata-fusion | active | AdaptiveConcatPool, diagnosis-as-target (+0.01), rank ensembling, metadata fusion |
| [[concepts/tabpfn-tabm]] | tabpfn, tabm, foundation-model, tabular, 2024, 2025 | active | TabPFN (100% win rate ≤10K rows, Nature 2024); TabM ICLR 2025 SOTA avg rank 1.7 |
| [[concepts/imbalanced-data]] | imbalanced, smote, focal-loss, threshold-optimization, downsampling | active | scale_pos_weight + StratifiedKFold + threshold optimization dominates; SMOTE before split = leakage |
| [[concepts/feature-selection-advanced]] | feature-selection, null-importance, boruta, lofo, adversarial-validation | active | 6-stage pipeline: correlation pruning → MI → adversarial val → null importance → BorutaShap → LOFO |
| [[concepts/shap-feature-engineering]] | shap, feature-engineering, interactions, drift-detection | active | SHAP dependency plots reveal transformations; interaction matrix finds top pairs; error analysis drives FE |
| [[concepts/memory-optimization]] | memory-optimization, polars, cudf, parquet, pandas, rapids | active | reduce_mem_usage (65-80%), Polars 10-15x faster, cuDF 150x GPU speedup, parquet zstd caching |
| [[concepts/universal-kaggle-tricks]] | universal-tricks, ensemble, seed-averaging, groupby, pseudo-labeling | active | Cross-competition priority matrix from NVIDIA GM playbook + Neptune.ai (65+ comps) + KazAnova |
| [[concepts/target-encoding-advanced]] | target-encoding, glmm, james-stein, catboost, entity-embeddings, woe | active | GLMM benchmark winner (Pargent 2022); CatBoost ordered TS; entity embeddings for NNs |
| [[concepts/multi-target-learning]] | multi-target, auxiliary-learning, distillation, ordinal | active | Non-scored targets as auxiliary signal (+0.002-0.010); teacher-student distillation; ordinal decomposition |
| [[concepts/external-data-leakage]] | leakage, external-data, id-leakage, metadata | active | ID leakage scan, file metadata leakage, adversarial validation for external data gate, leakage checklist |
| [[concepts/financial-competition-patterns]] | financial, purged-cv, jane-street, optiver, trading | active | Purged CV with embargo; GRU+Transformer ensemble; t-SNE time ordering (360 NN features); Fibonacci HMA |
| [[concepts/medical-imaging-patterns]] | medical-imaging, 2.5d, foundation-model, pathology, pseudo-labeling | active | 2.5D slice stacking, CLAHE, Phikon MIL for pathology, Stable Diffusion augmentation, BirdCLEF patterns |
| [[concepts/tabular-augmentation]] | tabular-augmentation, scarf, vime, saint, tta | active | SCARF swap noise (ICLR 2022), SAINT CutMix+Mixup, TTA with feature noise, TabMDA for tree models |
| [[concepts/gradient-boosting-advanced]] | xgboost, lightgbm, catboost, optuna, gpu, 2024, 2025 | active | Advanced GBDT params: dart, goss, linear_tree, border_count, Optuna GPSampler recipe |
| [[concepts/post-processing]] | post-processing, rankgauss, calibration, temperature-scaling, clipping | active | RankGauss, temperature scaling (tunes log-loss, not AUC), probability clipping, rank blending |
| [[concepts/time-series-features]] | time-series, feature-engineering, lags, rolling-stats, fourier | active | Lag features, EWMA rolling stats, Fourier cyclical encoding, one-model-per-horizon strategy |
| [[concepts/time-series-cv]] | time-series, cross-validation, walk-forward, purged-cv, embargo | active | Walk-forward CV, sliding window, purged CV (embargo), post-cutoff CV, multiple window validation |
| [[concepts/llm-fine-tuning-kaggle]] | nlp, llm, deberta, transformer, awp, llrd, lora, 2024, 2025 | active | AWP, LLRD, multisample dropout, two-stage training, LoRA/QLoRA, knowledge distillation |
| [[concepts/multimodal-classification]] | vision, vlm, multimodal, classification, vqa | draft | VLM pipelines, feature fusion, end-to-end multimodal for image+tabular competitions |
| [[concepts/straddle-strategy]] | trading, forex, options, straddle, nfp | draft | Simultaneous long/short orders around volatility events; entry distance, trailing stops, whipsaw risks |

---

## Entities (People, Orgs, Tools, Frameworks)

| Page | Tags | Status | Summary |
|------|------|--------|---------|
| [[entities/jason-profile]] | person, human, profile | active | Jason — human collaborator; CST; Kaggle + forex trading; approval gate on submissions |
| [[entities/machine-learning-advisor]] | system, shiny, rag, chromadb, hybrid-retrieval | active | MachineLearningAdvisor — hybrid RAG Shiny app; ChromaDB + index search + gpt-5.4-mini; Competition Strategy Mode |
| [[entities/xgboost]] | xgboost, framework, tool | active | XGBoost — primary gradient boosting framework; key params used in Jason's work |
| [[entities/lightgbm-catboost]] | lightgbm, catboost, framework | active | LightGBM (fast, leaf-wise) and CatBoost (categorical native) — comparison and usage notes |
| [[entities/qwen-vl]] | vlm, vision, qwen, multimodal, tool | draft | Qwen-VL vision-language model; Stage 1 of AUTOPILOT VQA pipeline for scene descriptions |
| [[entities/claude-sonnet]] | llm, claude, anthropic, json-extraction, tool | draft | Claude Sonnet for structured JSON extraction; Stage 2 of AUTOPILOT VQA pipeline |
| [[entities/oanda]] | forex, broker, api, trading, tool | draft | OANDA forex broker; v20 REST API; NFP straddle execution across accounts 006/007 |

---

## Comparisons

| Page | Tags | Status | Summary |
|------|------|--------|---------|
| [[comparisons/nfp-stop-configurations]] | trading, forex, nfp, straddle, comparison | draft | Fixed vs trailing vs tighter stop configs for NFP straddle; backtest comparison |

---

## Synthesis

*(empty — add cross-cutting analyses and evolving theses here)*

---

*Last updated: 2026-04-17 | Page count: 63 wiki pages + 49 raw solution files + ndres catalog indexed above*
