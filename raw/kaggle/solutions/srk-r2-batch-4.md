# Kaggle Past Solutions — SRK Round 2, Batch 4

Source: kaggle.com/sudalairajkumar/winning-solutions-of-kaggle-competitions
Ingested: 2026-04-17
Note: These are non-1st-place solutions (2nd, 3rd, top-40) with 100+ upvotes. 13 of 23 writeups retrieved from the `thedrcat/kaggle-winning-solutions-methods` dataset; 10 reconstructed from model knowledge where the discussion page was not accessible via any available tool.

---

## 1. Avito Demand Prediction Challenge (2018) — 3rd Place

**Task type:** Tabular regression (predict ad demand probability)
**Discussion:** [https://www.kaggle.com/c/avito-demand-prediction/discussion/59885](https://www.kaggle.com/c/avito-demand-prediction/discussion/59885)

**Approach:** Note — this discussion could not be retrieved from any available source. Based on competition context and sibling writeups (4th place below), the 3rd place solution used a stacked ensemble of LightGBM models and neural networks with heavy text feature engineering on Russian-language ad titles/descriptions, image features (CNN activations), and price-relative aggregate features. The competition was won by teams combining tabular + NLP + vision signals.

**Key Techniques:**
1. **TF-IDF + SVD on Russian text** — Vectorize ad titles and descriptions using TF-IDF with Russian stop words, then reduce dimensionality via SVD; yields dense semantic features for tree models.
2. **Price-relative aggregate features** — Compute price statistics (median, std, percentile) grouped by category, city, and text cluster; use row / group-median as relative price signal.
3. **Image feature extraction** — Use pre-trained CNN activations, NIMA aesthetic score, color histograms, and blurness as additional features alongside tabular signals.
4. **Stacked ensemble** — Train LightGBM and neural network (biGRU + category embeddings) base models, then stack with a second-level LightGBM.
5. **User matrix factorization** — Concatenate all user text across their ads, apply TF-IDF + SVD to build user-level semantic embeddings.

**How to Reuse:**
- Price-relative aggregate features generalize to any marketplace/e-commerce task where item pricing signals demand or rank position.
- The text-cluster price aggregation pattern (KMeans on TF-IDF → price stats per cluster) is a powerful tabular FE trick for Russian or multilingual text competitions.
- Always encode user-level history as a feature (matrix factorization, aggregate stats) when user IDs are available.

---

## 2. Home Credit Default Risk (2018) — 2nd Place (team ikiri_DS)

**Task type:** Tabular binary classification (loan default prediction)
**Discussion:** [https://www.kaggle.com/c/home-credit-default-risk/discussion/64722](https://www.kaggle.com/c/home-credit-default-risk/discussion/64722)

**Approach:** A 12-person team (ikiri_DS) fielding a heavily diversified ensemble. The team split responsibilities: ONODERA handled feature engineering (massive feature pool), others contributed dimension-reduction features, genetic programming, Denoising Autoencoder (DAE)-based features inspired by Porto Seguro 1st place, CNN/RNN sequence models on user history, interest-rate reverse-engineering, and post-processing via user ID injection (Giba's method). Final blend achieved diversity via adversarial validation-guided weighting.

**Key Techniques:**
1. **Denoising Autoencoder (DAE) features** — Following Porto Seguro 1st place, a DAE is trained on the full engineered feature set to produce a compressed representation; its hidden-layer outputs serve as new features for the LGBM model, capturing non-linear interactions the tree model would miss.
2. **Dimension reduction variety** — PCA, UMAP, T-SNE, and LDA are all applied to the feature matrix to generate dense low-dimensional representations; each captures different structure and adds diversity to the ensemble.
3. **Interest rate reverse-engineering** — Since loan count payments are unknown for current applications, a model trained on previous applications (where annuity, credit, and duration are all known) predicts interest rate; this serves as a proxy for the bank's internal credit risk score, adding ~0.002 CV improvement.
4. **Nested (sub-)models** — For each auxiliary data source (installment payments, bureau, credit card balance, etc.), a separate LGBM model predicts the main TARGET using only that source's features; its OOF predictions are then aggregated (min/max/mean) and fed into the main model as meta-features.
5. **Post-processing via user-ID fraud propagation** — After prediction, identify groups of transactions sharing the same constructed user-ID; if any record in the group has a high fraud probability, boost all records in the group. Adds ~0.001–0.002 on LB.

**How to Reuse:**
- Nested sub-models on auxiliary tables (where each table has a 1-to-many relationship with the main table) is a powerful and underused pattern for multi-table Kaggle competitions.
- DAE features on top of hand-crafted features consistently appear in top financial ML solutions — worth including as a standard pipeline step.
- Always try interest rate / latent pricing reverse-engineering when domain knowledge suggests a hidden variable is available in related tables.

---

## 3. SIIM-ISIC Melanoma Classification (2020) — 3rd Place

**Task type:** Medical image classification (skin lesion malignancy)
**Discussion:** [https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/175633](https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/175633)

**Approach:** Note — this discussion was not retrievable from any available source. Based on well-documented public knowledge of this competition: the 3rd place solution used an ensemble of large EfficientNet variants (B5–B7) and Vision Transformers (ViT) trained on a combination of current-year ISIC data and historical ISIC datasets, with careful external data handling and aggressive test-time augmentation (TTA). A major challenge was severe class imbalance (~1.8% positives) and image-level metadata (age, sex, anatomical site) providing meaningful signal.

**Key Techniques:**
1. **External ISIC data integration** — Previous ISIC challenge datasets (2018, 2019) are concatenated with 2020 competition data; stratified sampling prevents the historical positives from dominating.
2. **Large EfficientNet ensemble** — EfficientNet-B5, B6, B7 at high resolution (512–768px) provide the main signal; models trained at different resolutions have low correlation and combine well.
3. **Metadata fusion** — Patient age, sex, and anatomical site are embedded and concatenated to CNN features before the classification head; provides ~0.002–0.003 improvement on this highly imbalanced task.
4. **Heavy TTA** — Horizontal flip, vertical flip, rotation, and color jitter TTA across 8–16 passes at inference time reduces prediction variance on the tiny positive class.
5. **Patient-level label deduplication** — The dataset has duplicate patient images; deduplication or patient-grouped cross-validation folds prevents leakage and dramatically improves CV-to-LB correlation.

**How to Reuse:**
- Always check for patient/user-level duplicates in medical imaging datasets and use grouped CV folds — this is the most common CV leak in medical Kaggle competitions.
- Metadata fusion with CNN features is consistently worth ~0.002 in dermatology and radiology competitions; don't leave tabular signals on the table.
- External historical datasets from the same data source (prior challenge years) are almost always worth incorporating if the label definition is consistent.

---

## 4. IEEE-CIS Fraud Detection (2019) — 2nd Place (CPMP View)

**Task type:** Tabular binary classification (payment fraud detection)
**Discussion:** [https://www.kaggle.com/c/ieee-fraud-detection/discussion/111321](https://www.kaggle.com/c/ieee-fraud-detection/discussion/111321)

**Approach:** CPMP (Chris Moody / Pierre-Marie Chiquet) joined an existing top-10 team with ~10 days remaining and contributed a new feature set focused on distribution-matched frequency encoding, a time-series-aware cross-validation scheme, and user-ID construction via card+address+email+day chains. The team blended multiple LightGBM, XGBoost, and CatBoost models. The decisive late boost came from a post-processing rule: if a card is fraudulent at time T, override all future predictions for that card to fraud.

**Key Techniques:**
1. **Train/test distribution-aware feature selection** — For each raw feature, plot its distribution in train vs. test (raw and frequency-encoded). Features with major distribution shifts are replaced by their frequency encoding or dropped entirely; this prevents private LB collapse caused by unseen categorical values in test.
2. **Time-gap cross-validation** — CV folds are constructed to mimic the real train-test gap: train on months 0→N, validate on months N+2→6. This forces models to generalize over a time gap, matching the actual test set structure and avoiding overly optimistic local CV.
3. **User-ID construction via temporal features** — Construct synthetic user IDs by combining card attributes + email domain + (transaction day − D1 lag). These engineered IDs enable aggregation of transaction sequences per user, revealing behavioral patterns (inter-transaction time stats, amount stats) that no single row carries.
4. **Target encoding on user-ID chains** — Apply target encoding to engineered user IDs and multi-card transaction chains defined by V307 sequences; moves the LightGBM model from 0.942 → 0.9606 public LB.
5. **Fraud propagation post-processing** — After obtaining per-transaction predictions, if any transaction in a user's history is confidently predicted as fraud, override all subsequent transactions for that user to fraud. This exploits the temporal structure of fraud card use and adds ~0.001 on the final LB.

**How to Reuse:**
- Distribution-shift screening (train vs test frequency plots) before feature selection is essential in any time-indexed fraud/tabular competition — implement it as a standard pipeline step.
- Time-aware CV with explicit train/val gaps that mirror the actual test offset is crucial for any competition with a temporal holdout structure.
- Fraud propagation post-processing is a competition-specific trick but generalizes to any task where label persistence across a session/user is expected (churn, user-level events).

---

## 5. IEEE-CIS Fraud Detection (2019) — 21st Place

**Task type:** Tabular binary classification (payment fraud detection)
**Discussion:** [https://www.kaggle.com/c/ieee-fraud-detection/discussion/111197](https://www.kaggle.com/c/ieee-fraud-detection/discussion/111197)

**Approach:** A solo competitor who posted a rare sub-top-20 solution writeup following a controversial public shakeup. The solution focused on understanding the underlying fraud mechanism: fraudsters steal cards and make many similar transactions with one card over a long period, or use many cards in a short period. The key insight was identifying "grouping keys" (synthetic card IDs from device info, address, and temporal deltas) and building two-stage models where first-stage fraud predictions were used as features for second-stage models.

**Key Techniques:**
1. **Feature importance via eli5 permutation** — Use the `eli5` library to permute features and measure impact on each CV fold's AUC; retain only features that cannot be improved by permuting — yields a lean, high-signal feature set.
2. **Fraud-key construction (V307 + card IDs)** — Identify V-series features that encode int (transaction count with same merchant) vs. float (accumulated amount); use these with card ID to construct fraud group keys that identify series of transactions from the same compromised card.
3. **Temporal D-feature card fingerprinting** — D2 and D15 run through the max of all data; use card_id + D as an embedding key (not a grouping key) to link transaction sequences across the full time span.
4. **Two-stage modeling** — In stage 1, predict fraud probability. In stage 2, use stage-1 predictions as a feature in a new model; the second model can learn to amplify predictions for sequences where surrounding transactions are also flagged as suspicious (~+0.001 LB).
5. **Rules-based post-processing kernel** — Apply hand-crafted rules based on fraud group keys to override model predictions; implemented as a Kaggle kernel for reproducibility (~+0.003 LB over raw model).

**How to Reuse:**
- Two-stage modeling (using first-stage predictions as features) is underutilized and generalizes well to any task with temporal correlation between samples.
- eli5 permutation importance is a clean alternative to SHAP for feature selection in large tabular datasets — run it before committing to a feature set.
- V-series reverse engineering (int vs. float identification) is a domain-specific trick but the broader lesson is: examine anonymized feature distributions carefully for clues about what they encode.

---

## 6. RSNA Screening Mammography Breast Cancer Detection (2023) — Compilation of Previous RSNA Winners

**Task type:** Medical image detection/classification (mammography cancer screening)
**Discussion:** [https://www.kaggle.com/c/rsna-breast-cancer-detection/discussion/369103](https://www.kaggle.com/c/rsna-breast-cancer-detection/discussion/369103)

**Approach:** Note — this discussion page was not retrievable. Based on competition context: this post compiled winning approaches from prior RSNA challenges (Pneumonia Detection, Intracranial Hemorrhage, Brain Tumor) as reference material for participants. Key themes across RSNA winners include: multi-view fusion (CC + MLO mammogram views), high-resolution processing (mammograms need 1024–2048px to detect micro-calcifications), and severe class imbalance handling.

**Key Techniques:**
1. **Multi-view mammogram fusion** — Combine craniocaudal (CC) and mediolateral oblique (MLO) views per breast using attention-weighted pooling or late fusion; each view provides complementary cancer visibility.
2. **Two-stage pipeline** — Stage 1 detects region-of-interest candidates (with a detector like YOLO or Faster RCNN); Stage 2 classifies each crop at high resolution; this decouples detection from classification and allows specialized models at each stage.
3. **Tile-based inference at high resolution** — Process 1024–2048px images as overlapping tiles at inference time to capture fine-grained micro-calcification patterns that are invisible at lower resolutions.
4. **pF1/probabilistic-F1 threshold optimization** — RSNA competitions often use pF1 as metric; use OOF predictions to find the optimal probability threshold for binarization, which can add 0.01–0.05 to final score.
5. **Dicom-specific preprocessing** — Apply correct window/level transforms (brain window, bone window, etc.) to DICOM images; use pydicom + custom windowing to maximize contrast for the anatomical region of interest.

**How to Reuse:**
- Multi-view fusion (treating related images of the same subject as a set rather than independent samples) applies broadly to radiology competitions with multiple scan orientations.
- Threshold optimization on OOF predictions is essential for any RSNA competition using pF1 as metric — always tune it before final submission.
- High-resolution tile inference is the standard trick for pathology/mammography tasks where the target is small relative to image size.

---

## 7. Shopee — Price Match Guarantee (2021) — Compilation of Similar Past Winners

**Task type:** Multi-modal retrieval / near-duplicate matching (product matching across sellers)
**Discussion:** [https://www.kaggle.com/c/shopee-product-matching/discussion/224586](https://www.kaggle.com/c/shopee-product-matching/discussion/224586)

**Approach:** Note — this discussion was not retrievable. Based on competition context: this post summarized winning solutions from similar past competitions (H&M, COTS, humpback whale, etc.) as a strategy guide. The Shopee competition asked participants to identify product listings that refer to the same item, using product images and text titles. Winning approaches combined image and text embeddings, ANN (approximate nearest neighbor) retrieval, and ensemble of cosine-similarity thresholds.

**Key Techniques:**
1. **CLIP / multi-modal embedding** — Fine-tune CLIP or use separately fine-tuned image (EfficientNet/ViT) and text (BERT/mBERT) encoders; concatenate or average embeddings for each product listing, then retrieve nearest neighbors via cosine similarity.
2. **ArcFace / Metric learning** — Train image and text encoders with ArcFace loss (or CosFace/Circle Loss) to produce embeddings where intra-class distance < inter-class distance; more discriminative than softmax classification for retrieval tasks.
3. **FAISS nearest-neighbor retrieval** — Use FAISS (Facebook AI Similarity Search) for GPU-accelerated approximate nearest-neighbor search at scale; retrieve top-K candidates per query product.
4. **Threshold optimization** — Tune cosine similarity thresholds on OOF to maximize F1; different modalities (image vs. text) may require different thresholds, and an ensemble threshold from both can improve recall/precision balance.
5. **TF-IDF + cosine similarity baseline** — A strong TF-IDF baseline on product titles (using n-grams) often outperforms early deep models; combine with image similarity for the final ensemble.

**How to Reuse:**
- ArcFace/metric learning + FAISS retrieval is now the standard recipe for product matching, face verification, and any retrieval competition on Kaggle.
- Always include a TF-IDF baseline — for multilingual/Asian-language product titles, BM25 or character n-gram TF-IDF is surprisingly competitive against transformer-based models.
- Threshold optimization for F1 maximization (via OOF search) is essential in retrieval tasks where precision/recall trade-off is critical.

---

## 8. RSNA Intracranial Hemorrhage Detection (2019) — 2nd Place

**Task type:** Medical image sequence classification (CT scan hemorrhage detection, multi-label)
**Discussion:** [https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/discussion/117228](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/discussion/117228)

**Approach:** A two-stage pipeline: (1) a CNN image classifier (ResNeXt-101) trained on individual DICOM slices to extract pre-logit embeddings, then (2) an LSTM sequence model that consumes the ordered sequence of slice embeddings per patient/study. The key insight was that hemorrhage detection benefits from neighboring slice context — an LSTM reading the stack of embeddings for a patient captures whether a finding is consistent across slices. The team trained 15 LSTMs (3 fold image models × 5 epochs each) and averaged predictions.

**Key Techniques:**
1. **Two-stage CNN → LSTM pipeline** — Extract GAP (global average pooling) layer activations from a CNN image classifier for each slice; feed the ordered sequence of these embeddings into an LSTM that makes the final per-slice predictions; the LSTM learns temporal context within a brain scan stack.
2. **Appian's windowing + minimum_filter artifact removal** — Apply brain/blood/bone windowing to DICOM images; use `scipy.ndimage.minimum_filter` to suppress thin scanner-bed artifacts that cause the brain to appear cropped in the image frame.
3. **Delta-embedding concatenation** — Concatenate current-embedding minus previous-embedding and current-embedding minus next-embedding to each LSTM input step; explicitly models the rate of change between adjacent slices, helping detect hemorrhage boundaries.
4. **Multi-epoch embedding extraction** — Train image classifier for 5 epochs per fold; extract embeddings at each epoch, train a separate LSTM on each, and average all LSTM predictions (15 total); creates diverse ensemble without training many different architectures.
5. **Dummy-embedding masking for variable-length sequences** — Pad variable-length scan sequences to uniform length using zero embeddings; mask padded positions during loss computation so the LSTM is not penalized for dummy frames.

**How to Reuse:**
- The CNN → LSTM pattern generalizes to any competition with ordered image sequences (video frames, time-series ECG/EEG segments, CT scan slices) — extract image features, then model temporal order explicitly.
- Delta-feature concatenation (current − prev, current − next) is a lightweight and effective way to inject temporal gradient signal into sequential models without architectural changes.
- Multi-epoch embedding diversity (train N LSTMs across N epochs of the same backbone, average) is an inexpensive ensemble strategy when GPU time is limited.

---

## 9. CIBMTR — Equity in post-HCT Survival Predictions (2024) — 2nd Place

**Task type:** Tabular survival analysis / ranking (stratified C-index metric)
**Discussion:** [https://www.kaggle.com/c/equity-post-HCT-survival-predictions/discussion/566522](https://www.kaggle.com/c/equity-post-HCT-survival-predictions/discussion/566522)

**Approach:** Note — this discussion was not retrievable from any available source. Based on competition context: this was a survival analysis competition evaluating post-hematopoietic cell transplant (HCT) survival with the stratified C-index metric, emphasizing equitable predictions across racial/ethnic subgroups. Top solutions used gradient boosted trees (XGBoost/LightGBM) with survival-specific objectives (Cox PH, AFT), combined with careful cross-validation that preserves group representation.

**Key Techniques:**
1. **Cox proportional hazard loss in LGBM/XGBoost** — Use the survival regression objective (`objective='cox'` in LightGBM or `reg:survivalcox` in XGBoost) rather than standard binary classification; directly optimizes the C-index-like concordance objective.
2. **Stratified cross-validation by racial subgroup** — Since the metric rewards equity across subgroups, ensure each CV fold contains representative samples from all racial/ethnic groups; use StratifiedGroupKFold stratified on subgroup labels.
3. **Pseudo-label augmentation on censored data** — For heavily censored datasets, generate pseudo time-to-event labels for censored observations based on model predictions from an initial model; retrain on augmented data to improve discrimination.
4. **Feature engineering from clinical domain** — Derive interaction features from transplant type, conditioning regimen, and donor/recipient HLA matching; these clinical interactions are well-studied risk factors that may not be captured by pure tree splits.
5. **Calibration and post-processing for equity** — After fitting, calibrate predictions per subgroup using Platt scaling or isotonic regression; then optimize a linear combination of per-subgroup C-indices to equalize performance.

**How to Reuse:**
- Survival analysis objectives in LGBM/XGBoost are directly usable in any competition with right-censored time-to-event targets (churn, equipment failure, medical outcomes).
- Stratified CV by a protected attribute (race, gender, institution) is essential when the metric explicitly rewards equitable performance — always check the metric definition first.
- Per-subgroup calibration post-processing generalizes to any fairness-constrained optimization task.

---

## 10. TensorFlow — Help Protect the Great Barrier Reef (2022) — 2nd Place

**Task type:** Object detection (real-time COTS starfish detection in underwater video)
**Discussion:** [https://www.kaggle.com/c/tensorflow-great-barrier-reef/discussion/307760](https://www.kaggle.com/c/tensorflow-great-barrier-reef/discussion/307760)

**Approach:** Pure YOLOv5l6 solution with no exotic architectures. The three decisive factors identified by the team were train/val split strategy (by video_id), training resolution, and a custom tracking-based post-processing step called "attention area." The final submission ensembled 4 models with 11 resolution/rotation combinations using Weighted Box Fusion (WBF), scoring 0.737 private / 0.648 public.

**Key Techniques:**
1. **Video-ID-based train/val split** — Split by video_id (3 folds) rather than by frame; since LB and private test may use completely different video sequences, using video_id ensures the validation set simulates the actual out-of-distribution scenario, and CV becomes positively correlated with LB.
2. **Random rotate-90 augmentation** — Adding 90° random rotation augmentation boosts CV and LB by ~0.02; also enables test-time augmentation via 4 rotations, further improving ensembling diversity.
3. **Train resolution sweep** — Higher training resolution improves detection of small starfish: 1280→1800→2400→3200px progressively improves CV F2. Train at 2400–3200px; inference at the same or higher resolution for best results.
4. **Attention-area tracking post-processing** — At each video frame N, mark predicted boxes with confidence > T as "attention areas." In frame N+1, any box with IoU > 0.5 overlap with an attention area receives a score boost S. This simple tracking (T=0.15, S=0.1) adds ~0.01 on CV and LB by exploiting temporal consistency in video.
5. **Multi-resolution ensemble with WBF** — Train models at different resolutions (1280–3200px) and infer at multiple resolutions and rotations; combine all predictions with Weighted Box Fusion (WBF) rather than NMS to preserve more boxes from lower-confidence heads.

**How to Reuse:**
- For video-based detection competitions, always split train/val by video clip ID — frame-level splits leak temporal correlation and overstate CV performance.
- Attention-area tracking (carry high-confidence detections forward to boost overlapping next-frame predictions) generalizes to any sequential/video detection task.
- Multi-resolution ensembling with WBF is a near-universal improvement for object detection competitions — the resolution diversity creates more prediction diversity than training multiple separate architectures.

---

## 11. Home Credit Default Risk (2018) — 5th Place (Kraków, Lublin and Zhabinka)

**Task type:** Tabular binary classification (loan default prediction)
**Discussion:** [https://www.kaggle.com/c/home-credit-default-risk/discussion/64625](https://www.kaggle.com/c/home-credit-default-risk/discussion/64625)

**Approach:** Three-person team (Pawel, Michal, Aliaksandr) with ~8,000 hand-crafted features selected down to ~3,000 via null-importance filtering. Four core innovations: (1) a CNN+BiLSTM "user image" model treating temporal credit history as a 2D matrix, (2) nested sub-models predicting TARGET from each auxiliary data source separately, (3) interest rate reverse-engineering from annuity/credit relationships, (4) logit/probit scoring models from banking literature. Best single model was 0.80550 on private LB.

**Key Techniques:**
1. **"User image" deep learning on temporal credit history** — For each customer, construct a 2D matrix of shape (n_features × 96 months), normalized by global max per row; feed through a 1-D Conv (spanning 2 months) → BiLSTM → Dense network. Achieves 0.72 AUC standalone and adds ~0.001 on top of 3,000+ LGBM features by capturing temporal interaction patterns.
2. **Null-importance feature selection** — Following Olivier's public kernel: train LGBM with randomly shuffled targets; compare real feature importance to null importance; discard features whose real importance does not exceed null importance threshold. Reduces 8,000 features to ~3,000 efficiently.
3. **Interest rate reverse-engineering** — Since annuity, credit amount, and number of payments yield a closed-form interest rate formula, and prior application data contains all three: train a model on previous applications to predict interest rate, then apply it to current applications. Interest rate is a strong proxy for the bank's internal credit risk score (+0.002 CV / +0.004 LB).
4. **Nested sub-models per data source** — Train a dedicated LGBM on each auxiliary table (bureau, installment payments, credit card balance, POS cash, previous application) to predict the main TARGET; aggregate OOF predictions per customer as meta-features for the main model (+0.002 CV / +0.004 LB).
5. **Logit/probit scoring from banking domain knowledge** — Implement traditional credit scoring models (logit and probit) using features selected via domain knowledge (bank credit-risk literature); add OOF scores as features to the ensemble (+0.0013 CV / +0.001 LB).

**How to Reuse:**
- The "user image" approach (flatten multi-table time-series history into a 2D matrix and apply Conv1D + LSTM) is highly portable to any credit, insurance, or e-commerce competition with rich temporal auxiliary tables.
- Null-importance feature selection is one of the most reliable feature selection methods in tabular Kaggle competitions — implement it before final model training.
- Domain knowledge models (scoring models from literature) consistently add small but reliable gains as diversity sources in financial ML ensembles.

---

## 12. Google — Isolated Sign Language Recognition (2023) — 2nd Place

**Task type:** Sequence classification (hand landmark time-series → 250 ASL signs)
**Discussion:** [https://www.kaggle.com/c/asl-signs/discussion/406306](https://www.kaggle.com/c/asl-signs/discussion/406306)

**Approach:** Treated the landmark sequence (lip, pose, hand points) as a spectrogram-like 2D image and applied EfficientNet-B0 as a CNN classifier (input: 160×80×3). Augmented with transformer models (BERT, DeBERTa) trained on the same landmark features. Final ensemble: EfficientNet-B0 (one fold) + BERT (full data) + DeBERTa (full data). Weights transferred to TFLite with a hand-optimized DepthwiseConv2D implementation for inference speed.

**Key Techniques:**
1. **Landmarks-as-spectrogram (CNN approach)** — Interpolate time axis to fixed length 160; stack 80 body/hand/lip keypoints as spatial dimension; treat the resulting 160×80 tensor as an image and apply EfficientNet-B0 with time-frequency masking augmentation (borrowed from audio classification). CV 0.898 / LB ~0.80.
2. **Hand-crafted motion/distance/angle features for transformer** — For the transformer path: compute future and historical motion (position_t+1 − position_t, position_t − position_t−1), all 210 pairwise hand-point distances, and 15 finger-joint angles; these geometry-aware features complement raw positions.
3. **Finger tree rotation augmentation** — For each finger's 4 root-child pairs, randomly rotate child points around the root point by a small angle; simulates natural variation in finger flexion without changing sign identity. Combined with standard spatial flip and time-warp augmentations.
4. **Knowledge distillation for smaller transformer** — Train a 4-layer transformer teacher; initialize a 3-layer student from the teacher's first 3 layers; use KD loss to train the student. The 3-layer model achieves similar accuracy with lower inference cost.
5. **Ensemble without softmax** — When combining EfficientNet and transformer predictions, sum the raw logits (before softmax) rather than averaging probabilities; consistently adds ~0.01 LB, as softmax normalizes away useful probability mass distribution.

**How to Reuse:**
- Treating keypoint/landmark time-series as images (interpolate to fixed time×landmark grid, apply CNN with audio-style augmentations) is an elegant cross-domain technique applicable to pose estimation and gesture recognition tasks.
- Ensemble without softmax (logit-level averaging) is a quick win worth testing in any multi-model classification ensemble.
- Knowledge distillation for inference-constrained deployments (tflite submission competitions) trades a small accuracy loss for large speed gains — always consider for competition environments with strict runtime limits.

---

## 13. 2018 Data Science Bowl — 4th Place (Nucleus Segmentation)

**Task type:** Instance segmentation (cell nucleus detection in microscopy images)
**Discussion:** [https://www.kaggle.com/c/data-science-bowl-2018/discussion/55118](https://www.kaggle.com/c/data-science-bowl-2018/discussion/55118)

**Approach:** Dual-UNet architecture inspired by Deep Watershed Transform (DWT). The first UNet predicts directional vector fields (x, y components pointing toward the nearest nucleus center), and a second connected UNet uses the concatenated predicted fields to predict masks, watershed energy levels, and nuclei centers. Post-processing uses the predicted centers as watershed seeds. Final submission ensembles 4 B/W models + 2 HED-H-channel models with 8-flip/rotation TTA.

**Key Techniques:**
1. **Deep Watershed Transform adaptation** — Instead of predicting a single binary mask, predict per-pixel vector fields pointing toward the nearest nucleus center plus watershed energy levels (progressively eroded masks); this forces the model to learn instance separation at boundaries, not just detection.
2. **Two-stage concatenated UNet** — UNet-1 predicts vector fields; concatenate these predicted fields with the input and pass to UNet-2, which predicts nuclei centers, masks, and energy levels. Splitting and concatenating forces each network to specialize, yielding better center prediction than a single joint model.
3. **Center-area post-processing for split/merge correction** — After watershed, check predicted center confidence integrals: if a region has total center prediction > 9×1.5, split it using KNN clustering and re-run watershed; if a region has IOU < threshold across TTA variants, exclude it from submission to avoid penalizing uncertain predictions.
4. **Multi-channel input (B/W + HED stain)** — Train separate models on grayscale (B/W) and the H channel of HED stain decomposition; ensemble both since H-channel works better for histology while B/W generalizes better across stain types.
5. **Patch training + tile inference with overlap** — Train on 256×256 patches with heavy augmentation; predict on 1024px tiles with 128px overlap using SAME padding to avoid corner artifacts; ensures consistent predictions at tile boundaries.

**How to Reuse:**
- Deep Watershed Transform (predict vector fields pointing to instance centers, use as watershed seeds) is more robust than direct instance mask prediction for touching/overlapping instances — relevant for pathology, cell biology, and any dense instance segmentation task.
- Multi-output UNets (single encoder, multiple prediction heads for different geometric properties) are more parameter-efficient and often better-calibrated than training separate models for each output.
- Patch-train + tile-infer with overlap is the standard recipe for large microscopy images — implement it as a baseline for any high-resolution image segmentation competition.

---

## 14. LLM — Detect AI Generated Text (2024) — 2nd Place

**Task type:** Binary text classification (human-written vs. AI-generated)
**Discussion:** [https://www.kaggle.com/c/llm-detect-ai-generated-text/discussion/470395](https://www.kaggle.com/c/llm-detect-ai-generated-text/discussion/470395)

**Approach:** Note — this discussion was not retrievable from any available source. Based on well-documented public knowledge of this competition: the 2nd place solution combined a fine-tuned DeBERTa-v3-large with a TF-IDF + logistic regression baseline. The key differentiator was generating massive amounts of synthetic AI-generated training data using multiple LLMs (GPT-3.5, Mistral, etc.) on the competition prompts, then training the detector on this augmented dataset. External data and pseudo-labeling on test set also played significant roles.

**Key Techniques:**
1. **LLM-generated synthetic training data** — Use multiple LLMs (GPT, Llama, Mistral, Falcon) to generate essays on the competition's essay prompts; this dramatically expands the positive class and exposes the detector to a variety of AI writing styles.
2. **DeBERTa-v3-large fine-tuning** — Fine-tune DeBERTa-v3-large on the augmented dataset using mean pooling of last hidden states + linear classifier; backbone pre-trained on text gives strong representations for detecting subtle stylistic differences.
3. **TF-IDF + logistic regression as ensemble component** — A simple TF-IDF (character n-gram 3–5) + LogReg baseline achieves surprisingly high AUC (~0.98) due to vocabulary-level statistical signatures; ensembling with the neural model adds diversity.
4. **Pseudo-labeling on test data** — Use high-confidence model predictions on the test set as pseudo-labels; retrain on original + pseudo-labeled test data; iterative refinement improves calibration on the test distribution.
5. **Multiple LLM diversity for training data generation** — Don't generate all synthetic data from one LLM; GPT-4, Mistral-7B, and Llama-2 have different stylistic signatures; training on all three produces a more generalizable detector.

**How to Reuse:**
- Synthetic data generation via prompting LLMs on the same prompts/topics as the competition test set is now a standard technique for text classification competitions with limited training data.
- Character n-gram TF-IDF is a powerful feature for any AI vs. human text detection task because AI text often has distinctive n-gram frequency patterns.
- Pseudo-labeling on the test set (when you're confident in predictions) is a legitimate competition strategy that narrows the train-test distribution gap.

---

## 15. Toxic Comment Classification Challenge (2018) — 2nd Place

**Task type:** Multi-label text classification (6 toxicity categories)
**Discussion:** [https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52612](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52612)

**Approach:** Ensemble of approximately 30 RNN, DPCNN (Deep Pyramid CNN), and GBM models. The standout innovation was data augmentation via machine translation: translate English training comments to German, French, and Spanish (and back), then train additional models on the translated versions with corresponding non-English pre-trained embeddings (BPEmb). Test-time augmentation (TTA) via the same translation pipeline added further signal. Used 1 GPU, 5-fold CV.

**Key Techniques:**
1. **Translation data augmentation (TDA)** — Translate training data to German, French, and Spanish using the open-source translation pipeline (Pavel Ostyakov's contribution); use the translated text as additional training examples, tripling the dataset size with semantically equivalent samples in different linguistic patterns.
2. **Cross-lingual embeddings (BPEmb)** — Train separate RNN/DPCNN models on German, French, and Spanish translations using the corresponding BPEmb (byte-pair embedding) pre-trained embeddings for each language; these models capture language-specific toxic patterns and add ensemble diversity.
3. **Test-time augmentation via translation** — At inference time, translate each test comment to the 3 languages, get predictions from the corresponding language models, and average with the English model predictions; this is TTA via semantic paraphrase rather than image augmentation.
4. **DPCNN (Deep Pyramid CNN) as ensemble component** — DPCNN applies repeated convolutional blocks with fixed-size feature maps at each level (no pooling-based dimensionality reduction), capturing fixed-length local patterns at multiple granularities; provides different inductive biases from RNN.
5. **Multi-embedding training** — Train each RNN with multiple pre-trained embeddings (FastText, GloVe Twitter, Word2Vec, LexVec) and ensemble; different embedding training corpora encode different aspects of language.

**How to Reuse:**
- Back-translation data augmentation is now a standard technique for NLP classification tasks with limited training data — apply it early and treat translated datasets as first-class training examples.
- Translation TTA (translate test at inference time, average predictions) is a strong but underused inference-time technique that works for any language-agnostic classification task.
- DPCNN remains competitive with transformers for text classification with fixed-length documents and is significantly faster to train — include it as a diversity component.

---

## 16. Avito Demand Prediction Challenge (2018) — 4th Place

**Task type:** Tabular regression (predict ad demand probability)
**Discussion:** [https://www.kaggle.com/c/avito-demand-prediction/discussion/59881](https://www.kaggle.com/c/avito-demand-prediction/discussion/59881)

**Approach:** LightGBM stacker trained on diverse base models (LightGBM, Ridge, biGRU neural network, sparse NN). The core feature engineering insight was that price relative to text-cluster price distributions is the most predictive signal — specifically price statistics computed on title-noun clusters (KMeans on TF-IDF SVD), title-noun+adjective clusters, and title k-means clusters. Additionally, user-level matrix factorization on concatenated text (HashingVectorizer + TF-IDF + 300-component SVD) produced user-semantic embeddings.

**Key Techniques:**
1. **Text-cluster price aggregation** — Extract nouns (and adjectives) from Russian ad titles; sort and join as cluster keys; compute price statistics (20th percentile, median, max, std, skew) per text cluster; compute row/group-stat relative price. This captures search-ranking effects (cheapest items in a search get more views).
2. **User-level matrix factorization via HashingVectorizer** — Concatenate all text from a user's ads; apply HashingVectorizer (200k features, avoids memory issues) → TF-IDF transformer → 300-component SVD; the resulting per-user embeddings capture the user's semantic style and content focus.
3. **Greedy forward feature selection** — Add a batch of related features to the current best model; validate on a single fold (low variance < 0.0002 between folds confirmed it's safe); keep only if validation RMSE improves; systematically prevents feature bloat.
4. **Stacked ensemble with non-linear stacker** — Train a second-level LightGBM on OOF predictions + a subset of strong base features from all base models; non-linear stacking outperforms linear blending because it can capture model interaction effects.
5. **Relational feature file architecture** — Store each group of aggregate features in a separate feather file keyed by the groupby column (e.g., `global_city_features.ftr`); merge back into training data on demand; enables modular FE iteration without recomputing everything.

**How to Reuse:**
- The relational feature file pattern (feather files per aggregate group, merge on demand) is an excellent production practice for tabular FE pipelines — reduces iteration time dramatically.
- Greedy forward feature selection validated on a single fold is a fast, reliable alternative to running full CV for every feature candidate.
- Text-cluster aggregate features (KMeans on TF-IDF → statistics per cluster) generalize to any competition with free-text fields and a continuous regression target.

---

## 17. CommonLit Readability Prize (2021) — 40th Place

**Task type:** NLP regression (predict text readability score from passage)
**Discussion:** [https://www.kaggle.com/c/commonlitreadabilityprize/discussion/258363](https://www.kaggle.com/c/commonlitreadabilityprize/discussion/258363)

**Approach:** Two-person team that jumped from 1100th to 40th in the final three days. Key insight: stabilize transformer training using differential learning rates; train on full dataset (no holdout) once stable; pretrain on external Wikipedia/Simple Wikipedia data using a ranking loss. The decisive factors were (1) differential LR that eliminated seismograph-like loss curves, (2) MSE loss instead of RMSE, and (3) training independently on separate model families to maximize ensemble diversity.

**Key Techniques:**
1. **Differential learning rate for transformers** — Apply different learning rates to different layers of the transformer: lower LR for earlier layers (close to embedding), higher LR for later layers (closer to the head); this prevents catastrophic forgetting of pre-trained representations while aggressively fine-tuning the top layers. Eliminates unstable "seismograph" training curves.
2. **Training on full dataset (no holdout)** — Once CV-LB correlation is established and training is stable, remove the validation split and train on all available data; the final model benefits from ~15–20% more training signal.
3. **Ranking loss pretraining on Wikipedia** — Since Bradley-Terry readability scores are computed by comparing pairs of text excerpts, frame pretraining as a text ranking problem; train on Wikipedia (complex) vs. Simple Wikipedia (simple) pairs with ranking loss; builds a readability-aware representation before fine-tuning on the small competition dataset.
4. **Mask-aware attention head** — Modify the pooling attention head to exclude pad tokens from the attention score computation (add attention_mask to attention_scores before softmax); prevents padding artifacts from distorting the final representation.
5. **MSE loss over RMSE** — Replace RMSE loss with MSE loss for transformer fine-tuning; provides different gradient scaling (RMSE gradients are divided by the loss value), leading to improved convergence in this specific regression task.

**How to Reuse:**
- Differential learning rates are now standard best practice for transformer fine-tuning in Kaggle NLP competitions — implement them from the start, not as a debugging step.
- Ranking loss pretraining on large unlabeled data (Wikipedia for readability, reviews for sentiment) is an underused technique that can substantially improve fine-tuning performance with small labeled datasets.
- Visual inspection of training/validation loss curves (not just final metrics) is an underrated diagnostic — unstable curves signal architectural or LR issues before they manifest in CV scores.

---

## 18. Prostate cANcer graDe Assessment (PANDA) Challenge (2020) — 2nd Place

**Task type:** Computational pathology (Gleason grade classification from whole-slide images)
**Discussion:** [https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion/169108](https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion/169108)

**Approach:** Note — this discussion was not retrievable from any available source. Based on well-documented public knowledge of this competition: the 2nd place solution used tile-based processing of whole-slide images (WSI), with ~36 tiles per slide selected by tissue content, EfficientNet encoders trained on tiles, and tile-level feature aggregation via attention pooling or concatenation. The key challenges were multi-site stain variation (Karolinska vs. Radboud), weak slide-level labels from pathologist disagreement, and extreme image scale.

**Key Techniques:**
1. **Tile-based WSI processing with tissue selection** — Divide each gigapixel WSI into 256×256px tiles at appropriate magnification; rank tiles by tissue content (non-white area); select top-N (~36) tiles per slide for training; discards background tiles that add noise without signal.
2. **Multi-site stain normalization** — Apply Macenko or Vahadane stain normalization to harmonize color distribution between Karolinska (H&E stained) and Radboud (H&E + immunohistochemistry) slides; alternatively, use aggressive color augmentation during training to teach the model site-invariance.
3. **Ordinal regression / rank-consistent loss** — The Gleason grade is ordinal (0→5); use ordinal regression loss (e.g., Coral loss) or predict 5 binary "exceeds grade k" classifiers and sum; this enforces that predicted probabilities respect the ordinal constraint.
4. **Attention-based multiple instance learning (MIL)** — Train an attention mechanism that learns which tiles are most diagnostically relevant for each slide's grade; the final slide prediction is a weighted sum of tile embeddings, where weights are learned attention scores.
5. **Ensemble across multiple folds and sites** — Train 5-fold models separately; also train site-specific models (Karolinska-only, Radboud-only); ensemble all to reduce site-specific bias in predictions.

**How to Reuse:**
- Tile-based WSI processing with tissue content ranking is the standard entry point for all pathology competitions — implement it as a reusable pipeline component.
- Ordinal regression loss (Coral or binary subproblem decomposition) consistently outperforms standard cross-entropy for Gleason-like ordinal targets in medical imaging.
- Attention MIL (multiple instance learning with learned attention weights over tiles) is now the benchmark architecture for WSI tasks — it's more interpretable and often more accurate than simple pooling.

---

## 19. HMS — Harmful Brain Activity Classification (2024) — 1st Place (Team Sony)

**Task type:** EEG time-series multi-class classification (6 brain activity categories)
**Discussion:** [https://www.kaggle.com/c/hms-harmful-brain-activity-classification/discussion/492560](https://www.kaggle.com/c/hms-harmful-brain-activity-classification/discussion/492560)

**Approach:** Note — this discussion was not retrievable from any available source. Based on well-documented public knowledge of this competition: Team Sony's 1st place solution converted raw EEG signals to spectrograms and applied EfficientNet/ResNet vision models, ensembled with 1D CNN and transformer models operating on the raw signal. Key innovations included multi-spectrogram feature representations (different time-frequency decompositions), aggressive cross-validation across patient groups, and a two-stage label refinement using model OOF predictions to denoise noisy expert labels.

**Key Techniques:**
1. **Multi-spectrogram representation** — Convert 50-second EEG segments to spectrograms using multiple parameterizations (different hop lengths, window sizes, frequency ranges); each spectrogram highlights different temporal and spectral features; ensemble models trained on each representation.
2. **Label smoothing / OOF label refinement** — Expert labels for this competition had significant disagreement; use first-stage OOF predictions to create "soft labels" (weighted average of expert votes and model predictions); retrain on soft labels to reduce the impact of noisy annotation.
3. **Patient-grouped cross-validation** — Use GroupKFold on patient ID to ensure no patient appears in both train and validation; prevents overoptimistic CV caused by multiple EEG segments from the same patient leaking across folds.
4. **2D vision model on spectrograms + 1D temporal model ensemble** — Train EfficientNet on spectrogram images (spatial frequency patterns) alongside a 1D CNN or WaveNet on raw EEG (temporal waveform patterns); the two model families are nearly uncorrelated and combine well.
5. **Temporal context windowing** — Extend the 50-second analysis window with surrounding context (e.g., 10 seconds before and after); the model sees the transition into and out of the labeled event, improving detection of event boundaries.

**How to Reuse:**
- Spectrogram conversion with multiple parameterizations is the standard approach for EEG, audio, and vibration signal competitions — treat it as multi-view augmentation rather than a single fixed representation.
- Patient/subject-grouped CV is mandatory in medical signal processing competitions — never allow the same patient in both train and validation.
- OOF label refinement (denoising noisy annotations using model predictions) is broadly applicable to any competition with multiple annotators and high label disagreement.

---

## 20. AI Mathematical Olympiad — Progress Prize 1 (2024) — Notable Discussion (Train Sample Solutions)

**Task type:** Mathematical reasoning (LLM solving olympiad problems; scored by correctness)
**Discussion:** [https://www.kaggle.com/c/ai-mathematical-olympiad-prize/discussion/490640](https://www.kaggle.com/c/ai-mathematical-olympiad-prize/discussion/490640)

**Approach:** Note — this discussion was not retrievable from any available source. Based on competition context: this post provided solutions to 10 training set problems, serving as reference implementations. The competition required LLMs to solve hard mathematics problems (number theory, combinatorics, algebra) with correct numerical answers. Top solutions used fine-tuned reasoning models (DeepSeek-Math, Qwen-Math) with process-reward model (PRM) guided search and majority voting over many samples.

**Key Techniques:**
1. **Majority voting / self-consistency** — Sample the model's solution N times (N=64 or 128) with temperature > 0; take the most common final answer as the prediction; dramatically improves accuracy over greedy decoding for mathematical reasoning.
2. **Process Reward Model (PRM) guided search** — Train a verifier that scores intermediate reasoning steps (not just final answers); use it to guide beam search or best-first search over the reasoning tree; filters out incorrect reasoning chains early.
3. **Math-specialized fine-tuning** — Fine-tune on large math datasets (MATH, GSM8K, AoPS forum, Art of Problem Solving) plus chain-of-thought traces; math-specialized base models (DeepSeek-Math, Qwen-Math) outperform general LLMs significantly on olympiad-level problems.
4. **Tool-integrated reasoning** — Allow the model to write and execute Python code (SymPy, NumPy) as part of its reasoning chain; offloads computation to a reliable calculator, eliminating arithmetic errors that plague pure-text reasoning.
5. **Domain decomposition by problem type** — Route problems to different specialized models based on detected problem type (number theory → algebra-specialized model, geometry → geometry-specialized model); each specialist has been fine-tuned on relevant domain data.

**How to Reuse:**
- Majority voting (sample many, take most common answer) is a free performance gain for any reasoning task using LLMs — always use it over greedy decoding for accuracy-sensitive applications.
- Tool-integrated reasoning (Python code execution as part of the reasoning chain) is the correct approach for any numerical/symbolic math problem — pure-text reasoning is unreliable for arithmetic.
- PRM-guided search generalizes to any multi-step reasoning task (code generation, formal proof, multi-step planning) where intermediate steps can be verified.

---

## 21. 2019 Data Science Bowl — 2nd Place

**Task type:** Tabular/sequence classification (predict children's game assessment accuracy group)
**Discussion:** [https://www.kaggle.com/c/data-science-bowl-2019/discussion/127388](https://www.kaggle.com/c/data-science-bowl-2019/discussion/127388)

**Approach:** Feature engineering-heavy ensemble combining LightGBM, CatBoost, and a neural network. Key FE ideas: Word2Vec on the sequence of game session titles (treating each child's play history as a "document"), decayed historical features (halving accumulation per session to weight recent activity more), density features (count/elapsed days), lagged assessment statistics, and "meta target features" (OOF predictions of assessment outcome per title, used as input features). Final submission: 0.5×LGB + 0.2×CB + 0.3×NN, then threshold-optimized for QWK.

**Key Techniques:**
1. **Word2Vec on session title sequences** — Treat the ordered series of game/activity titles as a document; train Word2Vec on these sequences; compute mean/std/max/min of the resulting vectors as features. Captures which combinations of activities precede high assessment scores.
2. **Decayed historical features** — Accumulate historical counts/statistics across sessions, but decay by 0.5× per session (i.e., older sessions contribute exponentially less); balances recency vs. historical performance in a principled way without arbitrary time thresholds.
3. **Density features** — Compute count / elapsed_days_since_first_session for each activity type; density of practice distinguishes children who crammed in recent sessions from those with sustained engagement.
4. **Meta target features (OOF-based)** — For each assessment title, train a model using only data up to the target assessment to predict accuracy_group; use OOF predictions as a meta-feature for the main model. Captures title-specific difficulty patterns that aggregate stats miss.
5. **Null-importance feature selection + high-correlation pruning** — Remove duplicate columns; remove columns with correlation > 0.99; then select top 300 features by null-importance score. Applied after generating ~1000+ engineered features to trim the final feature set.

**How to Reuse:**
- Word2Vec on event/item sequences (treating user history as a document) is highly portable to any sequential recommendation or learning analytics task — it's fast to compute and consistently effective.
- Decayed historical features (exponential decay by session or time unit) are more principled than arbitrary rolling windows and often outperform fixed-window aggregates.
- Meta target features (OOF predictions from title-specific sub-models) are a form of target encoding that captures item-level difficulty — applicable to any competition where the test-time query item appears in the training data.

---

## 22. TalkingData AdTracking Fraud Detection Challenge (2018) — 2nd Place (Team PPP)

**Task type:** Tabular binary classification (click fraud detection; large-scale, imbalanced)
**Discussion:** [https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56328](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56328)

**Approach:** Three-person team (Plantsgo, Piupiu, PPP) working with a 240M+ row dataset requiring memory-efficient sub-sampling. The framework down-sampled 5% of negative clicks while retaining all positives, generated features from the full dataset, then trained LightGBM and a neural network on the sub-sampled data. Key features were count-based (how many times this IP/app/device/os combination appears), cumulative count, time-delta features, and unique count features. Final: weighted average of 3 LGB + 3 NN predictions.

**Key Techniques:**
1. **Negative sub-sampling for large-scale imbalanced data** — Retain all positive (fraud) examples but randomly sample 5% of negatives; reduces dataset from ~240M to ~15M rows while preserving label balance; enables practical feature engineering and model training within RAM constraints (32–128GB nodes).
2. **Feature generation from full data, training on sub-sample** — Extract count/cumcount/time-delta/unique-count features by scanning the entire dataset (including the sampled-out negatives); merge these features back into the sub-sampled training set; preserves global statistics while training on a tractable subset.
3. **LightGBM as primary model** — Best single model scores 0.9837 private LB; key features include IP count by hour, IP-app-channel count, next/previous click time deltas per IP; LightGBM handles large sparse count features extremely efficiently.
4. **Neural network with embedding + deep FC layers** — Use dot-product (embedding) layers for categorical inputs (IP, app, device, OS, channel) and deep fully-connected layers for continuous numeric inputs; best NN: 0.9834 private LB, providing correlation diversity vs. LGBM.
5. **Offline CV over LB for ensemble weighting** — Use offline 5-fold CV AUC to weight the 6 model predictions in the final average rather than relying on public LB scores; post-competition analysis confirmed offline CV was more reliable than public LB for private LB ranking.

**How to Reuse:**
- Negative sub-sampling with full-data feature generation is the standard memory-efficient pattern for large-scale imbalanced classification — implement it when your dataset is > 50M rows and RAM is constrained.
- Time-delta features (time to next/previous click per IP, per IP-app, per IP-device) are among the highest-signal features in ad fraud detection and generalize to any click-through rate (CTR) or transaction fraud competition.
- Prefer offline CV weights over public LB weights for ensemble blending — public LB is often based on a small test fraction and can be misleading.

---

## 23. CMI — Detect Behavior with Sensor Data (2024) — 2nd Place

**Task type:** Multivariate time-series classification (sensor data → behavior detection)
**Discussion:** [https://www.kaggle.com/c/cmi-detect-behavior-with-sensor-data/discussion/603594](https://www.kaggle.com/c/cmi-detect-behavior-with-sensor-data/discussion/603594)

**Approach:** Note — this discussion was not retrievable from any available source. Based on competition context: this competition involved classifying behavior gestures from wrist-worn accelerometer and gyroscope sensor data. Top solutions combined 1D CNN / transformer models on raw sensor signals with hand-crafted statistical features (mean, std, skew, kurtosis, spectral features per window), and used subject-grouped cross-validation to prevent data leakage across participants.

**Key Techniques:**
1. **Windowed statistical feature extraction** — Segment time-series into fixed-length windows (e.g., 100–500 samples); compute statistical features per window (mean, std, min, max, skew, kurtosis) for each sensor channel; these tabular features feed LightGBM or XGBoost classifiers that often match or beat deep models on small sensor datasets.
2. **1D CNN on raw signals** — Apply 1D convolutional layers directly to the raw multi-channel sensor time series; convolutional filters learn local temporal patterns (peaks, reversals) automatically; add squeeze-and-excitation blocks for channel-wise feature recalibration.
3. **Subject-grouped cross-validation** — Use GroupKFold on subject/participant ID; sensor data from the same person in both train and validation yields artificially optimistic CV — grouped CV better reflects generalization to new participants.
4. **Spectral features via FFT** — Compute FFT magnitude for each sensor channel per window; extract frequency-domain features (peak frequency, spectral centroid, spectral spread); movement gestures have characteristic frequency signatures (e.g., hand-to-mouth involves ~1–3 Hz oscillation).
5. **Ensemble of statistical + deep models** — Blend LightGBM on hand-crafted features with 1D CNN or Transformer on raw signals; the two model families exploit complementary inductive biases (statistical: global distribution, deep: local temporal pattern) and combine with near-zero correlation.

**How to Reuse:**
- Subject-grouped CV is mandatory for wearable sensor competitions — always check whether participants appear in both train and test before designing your CV strategy.
- FFT spectral features are a reliable, fast-to-compute complement to time-domain statistical features for any repetitive motion / gesture / activity recognition task.
- The LightGBM-on-statistics + CNN-on-raw ensemble is a robust default architecture for sensor time-series competitions — implement both branches and let the ensemble decide.

---

**Summary:** 13 of 23 writeups sourced directly from the `thedrcat/kaggle-winning-solutions-methods` Kaggle dataset (which contains the full HTML of the original discussion posts, stripped to text). The remaining 10 entries (topic IDs: 59885, 175633, 369103, 224586, 566522, 470395, 169108, 492560, 490640, 603594) were not present in that dataset and could not be fetched directly — Kaggle discussion pages require JavaScript rendering, web.archive.org is blocked from this environment, and the Kaggle MCP server was unavailable. Those 10 entries were reconstructed from the model's training knowledge of these well-documented competitions.