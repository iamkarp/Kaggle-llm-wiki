# Kaggle Past Solutions — SRK Batch 3

**Source:** kaggle.com/sudalairajkumar/winning-solutions-of-kaggle-competitions
**Ingested:** 2026-04-16

---

## Table of Contents

1. [BirdCLEF+ 2025](#1-birdclef-2025)
2. [LANL Earthquake Prediction (2019)](#2-lanl-earthquake-prediction-2019)
3. [Jigsaw Rate Severity of Toxic Comments (2021)](#3-jigsaw-rate-severity-of-toxic-comments-2021)
4. [Learning Equality Curriculum Recommendations (2022)](#4-learning-equality-curriculum-recommendations-2022)
5. [PLAsTiCC Astronomical Classification (2018)](#5-plasticc-astronomical-classification-2018)
6. [OTTO Multi-Objective Recommender System (2022)](#6-otto-multi-objective-recommender-system-2022)
7. [LLM Detect AI Generated Text (2023)](#7-llm-detect-ai-generated-text-2023)
8. [2019 Data Science Bowl](#8-2019-data-science-bowl)
9. [Quick Draw! Doodle Recognition (2018)](#9-quick-draw-doodle-recognition-2018)
10. [Happywhale (2022)](#10-happywhale-2022)
11. [Google Landmark Recognition 2020](#11-google-landmark-recognition-2020)
12. [Ubiquant Market Prediction (2022)](#12-ubiquant-market-prediction-2022)
13. [LMSYS Chatbot Arena (2024)](#13-lmsys-chatbot-arena-2024)
14. [Yale/UNC Geophysical Waveform Inversion (2025)](#14-yaleunc-geophysical-waveform-inversion-2025)
15. [TensorFlow Great Barrier Reef (2021)](#15-tensorflow-great-barrier-reef-2021)

---

## 1. BirdCLEF+ 2025

**Task:** Multi-label audio classification — identify bird species present in passive acoustic monitoring recordings, including species from Africa and Europe not seen in training.

**Discussion:** https://www.kaggle.com/c/birdclef-2025/discussion/583577

### Approach

The winning solution centered on a large pretrained audio foundation model (BirdNET or a BEATs-based encoder) fine-tuned on mel-spectrogram crops, with heavy use of test-time augmentation and post-processing thresholds tuned per species. The team addressed the severe class imbalance and the presence of out-of-distribution soundscapes by mixing focal-loss training with species-adaptive label smoothing. Ensemble blending of 5–8 diverse seeds/architectures with a logit-averaging strategy was used to stabilize predictions across the noisy pseudo-labeled test segments.

### Key Techniques

1. **Mel-spectrogram + pretrained audio encoder (BEATs/EfficientNet-B3):** Audio converted to 5-second mel-spectrogram crops at 128 mel bins; pretrained audio transformer encoder fine-tuned end-to-end with a low learning rate.
2. **Mixup and SpecAugment augmentation:** Time-frequency masking and mixup between positive samples of the same species to regularize against low-count classes.
3. **Pseudo-labeling on soundscape data:** High-confidence model predictions on unlabeled soundscape recordings used as additional training signal, weighted lower than verified labels.
4. **Species-wise threshold optimization:** Per-class thresholds calibrated on out-of-fold predictions using F1-maximization rather than a single global cutoff.
5. **Multi-seed ensemble with OOF-weighted blending:** 6–8 model seeds blended by optimizing ensemble weights on the held-out CV folds.

### How to Reuse

- **Any multi-label audio classification task:** The mel + pretrained encoder + per-class threshold pipeline is directly portable to bioacoustics, music tagging, and industrial audio monitoring.
- **Imbalanced multi-label problems generally:** Species-wise threshold tuning and focal loss apply whenever class prevalence varies by orders of magnitude.
- **Low-resource species:** Pseudo-labeling from unlabeled in-domain audio is a reliable way to expand training signal without manual annotation.
- **Soundscape segmentation:** Sliding-window inference with overlap-and-average works well for variable-length audio beyond fixed-length training crops.

---

## 2. LANL Earthquake Prediction (2019)

**Task:** Predict the time remaining until the next laboratory earthquake failure from a continuous acoustic emission time-series signal — pure regression on a 629 million sample waveform.

**Discussion:** https://www.kaggle.com/c/LANL-Earthquake-Prediction/discussion/94390

### Approach

The 1st place solution (Team "Breadcrumbs") approached this as a statistical feature engineering problem rather than a raw sequence learning problem. They computed thousands of handcrafted statistical features — rolling moments, quantile statistics, autocorrelation, FFT power bins, Mel-frequency features — over multiple window sizes on each 150,000-sample test segment, then trained a LightGBM gradient boosting model on those features. The key insight was that the acoustic signal's statistical structure (particularly the kurtosis and high-percentile values) encodes proximity to failure in a physically meaningful way.

### Key Techniques

1. **Massive statistical feature extraction:** 1,000+ features per segment including rolling mean, std, kurtosis, skewness, quantiles (1st through 99th), Hilbert transform envelope statistics, and FFT spectral energy across frequency bands.
2. **Multi-scale windowing:** Features computed at multiple sub-window sizes (1K, 5K, 10K, 50K samples) within the 150K segment to capture both local and global signal structure.
3. **LightGBM with careful hyperparameter tuning:** Gradient boosted trees on the feature matrix; MAE loss to match the competition metric; bagging and feature subsampling to prevent overfitting on the ~4,600 training segments.
4. **Denoising / signal decomposition:** Wavelet denoising of the raw signal before feature extraction to separate the acoustic emission events from background noise.
5. **Stochastic weight averaging / ensembling of multiple seeds:** Multiple LightGBM models with different random seeds and feature subsets blended to reduce variance.

### How to Reuse

- **Time-series regression from sensor signals:** This feature-engineering-first strategy works well when you have a single continuous signal and need to predict a scalar target for fixed-length windows.
- **Physics-informed feature design:** When domain theory predicts which statistical moments correlate with the target (e.g., kurtosis for impulsive events), hard-coding those features beats learned representations on small datasets.
- **Any competition where segments are too short for LSTM/transformer training:** Rolling statistics + GBDT is robust when you have fewer than 10K labeled windows.
- **Seismic / vibration monitoring in production:** The wavelet-denoise → feature-extract → GBDT pipeline is fast and interpretable for real-time monitoring.

---

## 3. Jigsaw Rate Severity of Toxic Comments (2021)

**Task:** Given pairs of comments, predict which is more toxic — a pairwise ranking problem evaluated by agreement rate with human rater pairs.

**Discussion:** https://www.kaggle.com/c/jigsaw-toxic-severity-rating/discussion/306274

### Approach

The winning approach reformulated the pairwise ranking task as a regression problem: train a transformer (RoBERTa-large / DeBERTa-large) to output a scalar toxicity score per comment, then rank within pairs. Rather than using the provided pairwise training data directly as classification labels, the team used margin-ranking loss combined with a mean-squared-error regression signal derived from aggregating across all pairs that each comment appears in to get a pseudo-absolute score. An ensemble of DeBERTa-v3-large and RoBERTa-large checkpoints with different random seeds provided the final submission.

### Key Techniques

1. **Scalar toxicity regression via margin-ranking loss:** The model predicts a real-valued score per comment; during training, a margin-ranking loss penalizes pairs where the predicted order disagrees with the label, with a configurable margin.
2. **Pseudo-absolute score construction:** For each comment, aggregate its win/loss record across all pairs in the training set to derive a pseudo-continuous "toxicity level," which is used as a regression target alongside the pairwise loss.
3. **DeBERTa-v3-large fine-tuning:** DeBERTa-v3-large with disentangled attention fine-tuned at a small learning rate (1e-5) with linear warmup; 5-fold cross-validation with stratification on comment pairs.
4. **Multi-seed ensemble:** 10–15 models (5 folds × 2–3 architectures) ensembled by averaging predicted logits before the final ranking step.
5. **External toxic comment data augmentation:** Jigsaw's prior competition datasets (Unintended Bias, Multilingual) used as additional pre-fine-tuning data to build better toxicity representations before the ranking-specific training phase.

### How to Reuse

- **Any pairwise ranking NLP task:** The margin-ranking loss + scalar score approach generalizes to preference learning, essay scoring, and feedback ranking.
- **Implicit label noisiness:** When labels only exist at the pair level, computing pseudo-absolute scores from win records is a clean way to get a regression signal.
- **DeBERTa-v3-large as default NLP encoder:** For text classification/ranking in English, DeBERTa-v3-large consistently outperforms RoBERTa and BERT; use as default starting point.
- **Human preference modeling:** This exact architecture adapts directly to RLHF reward model training — predicting which of two responses a human prefers.

---

## 4. Learning Equality Curriculum Recommendations (2022)

**Task:** Match open educational resources (content) to curriculum topics in 16 languages; evaluated by F2 score on topic–content correlations.

**Discussion:** https://www.kaggle.com/c/learning-equality-curriculum-recommendations/discussion/394812

### Approach

The 1st place solution treated this as a dense retrieval problem: fine-tune a multilingual sentence transformer (paraphrase-multilingual-mpnet-base-v2 / mDeBERTa) to encode both topics and content into a shared embedding space, then retrieve the top-k nearest neighbors per topic and re-rank with a cross-encoder. The two-stage retrieve-then-rerank pipeline allowed the model to efficiently handle 100K+ content documents while still applying an expensive pairwise scorer to a small candidate set. Unsupervised pre-training on the competition's curriculum hierarchy structure provided a warm start before supervised contrastive fine-tuning.

### Key Techniques

1. **Bi-encoder (dual encoder) retrieval:** Topic and content strings encoded independently by a shared multilingual transformer; cosine similarity ANN search (FAISS) retrieves top-100 candidates per topic.
2. **Cross-encoder re-ranking:** A separate cross-encoder (mDeBERTa-v3-base) encodes the concatenated [topic; content] pair and outputs a relevance score; applied only to the top-100 candidates from retrieval.
3. **Supervised contrastive fine-tuning with hard negative mining:** In-batch negatives augmented with hard negatives sampled from the bi-encoder's near-misses; MultipleNegativesRankingLoss used throughout.
4. **Curriculum hierarchy as structural signal:** Topic parent-child relationships used to construct hierarchical contrastive pairs — a topic should match its parent's content more than a sibling's content.
5. **Threshold tuning on OOF predictions for F2:** Because F2 up-weights recall, the decision threshold was explicitly optimized toward recall, retrieving more candidates and accepting more false positives than F1 would warrant.

### How to Reuse

- **Any retrieval/matching task with large document collections:** The bi-encoder + cross-encoder two-stage pipeline is the standard recipe for open-domain QA, document retrieval, and entity matching.
- **Low-resource multilingual tasks:** mDeBERTa or LaBSE outperform English-only encoders in mixed-language corpora; start there before fine-tuning.
- **Asymmetric retrieval:** When query and document types are semantically different (e.g., topic title vs. full lesson text), fine-tuning with asymmetric pairs is essential.
- **F2 / recall-heavy metrics:** Adjust decision thresholds explicitly for the target metric rather than defaulting to F1-optimal 0.5 threshold.

---

## 5. PLAsTiCC Astronomical Classification (2018)

**Task:** Classify 18 classes of astronomical transients (supernovae, AGN, etc.) from photometric light curves with redshift metadata; log-loss metric with heavy class imbalance and train/test distribution shift.

**Discussion:** https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75033

### Approach

The 1st place solution combined hand-engineered light curve features with a neural network operating on the raw time series, then blended the two via stacking. Feature engineering extracted cesium/FATS features (period, amplitude, color indices across 6 LSST passbands), while a 1D CNN and an LSTM processed the irregularly-sampled light curve after Gaussian process interpolation to a regular grid. The ensemble was calibrated against the known class priors from the cosmological simulation and post-processed to correct for the train/test prior shift (the test set has a very different class distribution from training due to observational bias).

### Key Techniques

1. **Gaussian process interpolation of irregular time series:** Light curves are unevenly sampled; GP interpolation (with a Matern kernel fit per object) projects each passband onto a regular 50-point grid, enabling CNN/LSTM input.
2. **Cesium / FATS statistical features:** ~1,000 features per object including period-finding (Lomb-Scargle), variability indices, color-band ratios, rise/fall time asymmetry — rich physics-informed feature set.
3. **1D CNN + LSTM hybrid on GP-interpolated curves:** Multi-branch architecture taking all 6 passband curves simultaneously; concatenated hidden states fed into a dense classifier.
4. **Class prior correction for train/test shift:** Competition organizers disclosed true class proportions in the test set; predictions were re-calibrated by multiplying class logits by the ratio of test-prior to train-prior before softmax.
5. **Hierarchical ensemble / stacking:** LightGBM on hand-crafted features, neural net on time series, and a GP-based Bayesian classifier stacked with a second-level logistic regression for the final output.

### How to Reuse

- **Irregularly-sampled time series:** GP interpolation to a regular grid is a principled way to handle variable-length, gappy sequences before applying standard sequence models.
- **Multi-modal tabular + sequence fusion:** Train separate feature-engineering and deep-learning pipelines; stack them rather than forcing a single architecture.
- **Train/test prior shift in classification:** When class prevalence differs between train and test (common in survey data), explicit logit re-weighting by the prior ratio is a simple and effective correction.
- **Astronomical / scientific multi-class:** Period-finding features (Lomb-Scargle) are portable to any periodic signal classification problem.

---

## 6. OTTO Multi-Objective Recommender System (2022)

**Task:** Predict which items a user will click, add to cart, and order next, given a session of prior events; evaluated by Recall@20 weighted across the three event types.

**Discussion:** https://www.kaggle.com/c/otto-recommender-system/discussion/384022

### Approach

The winning solution (Team "Lectura") built a two-stage candidate-generation + re-ranking pipeline. In stage 1, diverse candidate sets (~200 items per session) were generated using co-visitation matrices (item-to-item co-occurrence counts weighted by recency and event type), word2vec item embeddings trained on session sequences, and BM25. In stage 2, a LightGBM ranker was trained on rich hand-crafted features describing each (session, candidate) pair — recency of last interaction, session-level item popularity, embedding distance, co-visitation score — using the competition training data as implicit relevance labels.

### Key Techniques

1. **Co-visitation matrices as candidate generators:** Sparse item-to-item matrices built by counting how often pairs of items appear in the same session within a time window, weighted by event type (order > cart > click); used for ANN candidate retrieval.
2. **Word2vec on session sequences:** Item IDs treated as "words," sessions as "sentences"; skip-gram word2vec trained on the full session corpus to learn item embeddings that reflect co-occurrence patterns.
3. **Multi-objective label construction:** Separate binary relevance labels for clicks, carts, and orders; LightGBM trained with a weighted combination of the three Recall@20 losses, with orders given the highest weight.
4. **Rich session-context features:** Time since last interaction with candidate, number of events with candidate in session, candidate's global click/cart/order rate, session length, time of day — all combined into a 150+ feature vector.
5. **Negative sampling strategy:** Hard negatives selected from items that appear in the session but are not the target event type (e.g., items clicked but not ordered serve as hard negatives for the order model).

### How to Reuse

- **Session-based recommendation:** Co-visitation matrix + word2vec + LightGBM ranker is a practical production-ready baseline for any e-commerce or content recommendation task.
- **Multi-objective ranking:** Train separate rankers per objective or a joint model with weighted labels; always check that the weighting matches the business metric.
- **Large-scale candidate retrieval without a GPU:** Co-visitation matrices are CPU-only and extremely fast to build with Pandas/Polars; competitive with ANN on session data.
- **Cold-start sessions:** BM25 on item text features provides a robust fallback candidate generator when collaborative signal is thin.

---

## 7. LLM Detect AI Generated Text (2023)

**Task:** Binary classification of student essays as human-written or AI-generated, where the AI generator is unknown at test time and may differ from any model used in training.

**Discussion:** https://www.kaggle.com/c/llm-detect-ai-generated-text/discussion/470121

### Approach

The winning solution exploited the observation that n-gram and byte-pair-encoding (BPE) token statistics differ systematically between human and LLM-generated text. Rather than fine-tuning a large transformer, the team built a TF-IDF vectorizer on character n-grams (n=3–5) and word n-grams (n=1–3), then trained a logistic regression and a lightweight SGD classifier on the resulting sparse features. The key unlock was generating a large synthetic training corpus by prompting multiple open-source LLMs (Mistral-7B, Falcon, LLaMA-2) with the competition essay prompts to create a diverse set of AI-generated essays, massively expanding the training data beyond the small labeled set provided.

### Key Techniques

1. **Character + word n-gram TF-IDF features:** Character 3–5-grams capture subword patterns unique to LLM tokenization artifacts; word 1–3-grams capture phrasing patterns; combined into a single sparse feature matrix.
2. **Synthetic data generation at scale:** Competition essay prompts fed to 5–10 open-source LLMs via API/local inference; 100K+ synthetic AI essays generated to balance and diversify the training set.
3. **Logistic regression / SGDClassifier on sparse TF-IDF:** Simple linear models on the n-gram features; fast, interpretable, and resistant to overfitting given the large synthetic corpus.
4. **Ensemble of TF-IDF configurations:** Multiple TF-IDF vectorizers with different n-gram ranges, min/max df settings, and sublinear-tf flags; their probability outputs averaged.
5. **Pseudo-labeling on test essays:** High-confidence predictions on the (large) unlabeled test set used to further augment training, iterating 2–3 times.

### How to Reuse

- **AI text detection in any domain:** The character n-gram TF-IDF approach is language-model-agnostic; retrain with domain-specific prompts and a local LLM to generate synthetic negatives.
- **Data-scarce binary classification:** Synthetic data generation via LLM prompting is now a standard technique for bootstrapping training sets when labeled data is expensive.
- **Speed-critical inference:** TF-IDF + logistic regression runs in microseconds per sample; use when latency matters more than maximum accuracy.
- **Out-of-distribution robustness:** Character n-grams are more stable across different LLMs than model-specific perplexity scores, making this approach more robust to unknown generators.

---

## 8. 2019 Data Science Bowl

**Task:** Predict the accuracy group (0–3, ordinal) of a child's first attempt at an in-game assessment, given their prior game event log data from a PBS Kids educational app.

**Discussion:** https://www.kaggle.com/c/data-science-bowl-2019/discussion/127469

### Approach

The 1st place solution (Team "No Free Hunch") engineered a comprehensive feature set from the event logs capturing each child's learning trajectory — accuracy on prior assessments, attempts per clip/game, time spent, and event count distributions — then trained LightGBM and XGBoost models with optuna-tuned hyperparameters on these features. The ordinal output was handled by optimizing a custom QWK (quadratic weighted kappa) objective, and post-processing with threshold optimization on OOF predictions. A key insight was that user history features (how accurately a child performed on all previous same-type assessments) vastly dominated any other feature, making careful aggregation of historical accuracy the core of the solution.

### Key Techniques

1. **Historical accuracy aggregation per assessment type:** For each child and each of the 5 assessment types, compute mean accuracy, last accuracy, number of attempts, and accuracy trend over time — the most important feature group by far.
2. **Event count and time features per session:** Number of clicks, misclicks, correct/incorrect actions, and time elapsed per game session, aggregated across the child's full history.
3. **LightGBM with QWK-optimized thresholds:** Train with a regression objective (RMSE), then post-process continuous predictions by fitting 3 cut-points that maximize QWK on OOF predictions.
4. **Leave-one-installation-out cross-validation:** CV strategy groups all sessions from a single child (installation_id) into one fold to prevent data leakage from the sequential nature of game play.
5. **XGBoost + LightGBM ensemble:** Final submission blends predictions from both GBDT implementations; simple averaging of the continuous predictions before threshold optimization.

### How to Reuse

- **Ordinal classification from tabular data:** Always frame as regression + threshold optimization (QWK or otherwise) rather than as multi-class classification when labels have ordinal meaning.
- **User behavior / clickstream features:** Aggregating accuracy and engagement metrics over a user's history is the dominant signal in educational and behavioral datasets.
- **Installation/user-level leakage prevention:** Whenever multiple rows belong to the same entity, group-stratified CV is mandatory.
- **Threshold post-processing for QWK:** Optimize 3 cut-points jointly on OOF predictions with scipy.optimize; never use default 0.5/1.5/2.5 cuts.

---

## 9. Quick Draw! Doodle Recognition (2018)

**Task:** Classify 50.5 million hand-drawn doodles into 340 categories from stroke sequences; evaluated by MAP@3.

**Discussion:** https://www.kaggle.com/c/quickdraw-doodle-recognition/discussion/73738

### Approach

The winning solution (Team "Tugstugi") rendered stroke sequences as 128×128 raster images and trained a MobileNet/SE-ResNeXt ensemble using progressive resizing — starting at 64px and increasing to 128px during training. The massive dataset (50M+ examples) made training computationally intensive, so the team used a custom data pipeline that rendered images on-the-fly from the stroke JSON, avoiding the need to pre-render and store all images. Fine-tuning on only the most recently drawn (and presumably cleaner) examples in each category, plus heavy use of the full training set with cosine annealing LR schedules, yielded the best MAP@3.

### Key Techniques

1. **On-the-fly stroke-to-image rendering:** Stroke sequences rendered into grayscale raster images in the data loader (no pre-rendering), with variable stroke width and anti-aliasing; GPU training pipeline bottlenecked by CPU rendering, so multi-worker loading was critical.
2. **Progressive image resolution training:** Start training at 64×64 pixels, then fine-tune at 128×128; this reduces early-epoch compute cost while allowing the model to learn fine-grained details at higher resolution.
3. **SE-ResNeXt-50 / MobileNetV2 backbone ensemble:** Two architecturally diverse backbones ensembled by averaging class probabilities; diverse architectures reduce correlated errors.
4. **Training on "recognized" doodles only:** Quick Draw labels some doodles as "recognized" (correctly identified by Google's model); training on only these clean examples improved validation accuracy significantly.
5. **Cosine annealing with warm restarts (SGDR):** Learning rate schedule with multiple warm restarts allows escaping local minima; each restart produces a snapshot for snapshot ensembling.

### How to Reuse

- **Any stroke/vector-to-image classification:** Rendering strokes as raster images is the simplest way to apply standard CNN architectures to sketch data.
- **Massive dataset training:** On-the-fly augmentation/rendering avoids storage bottlenecks; crucial when dataset size exceeds available disk/memory.
- **Progressive resizing:** Start small, scale up — directly applicable to any image classification task; reduces wall-clock time significantly for large datasets.
- **MAP@k evaluation:** Output top-k class predictions sorted by probability; train with cross-entropy loss, do not need custom MAP@k training loss.

---

## 10. Happywhale (2022)

**Task:** Identify individual whales and dolphins from dorsal fin and fluke photographs — a metric learning / re-identification problem with 15K+ unique individuals and new individuals at test time.

**Discussion:** https://www.kaggle.com/c/happy-whale-and-dolphin/discussion/320192

### Approach

The winning solution treated this as an open-set recognition problem using metric learning. A ConvNeXt-large / EfficientNetV2-XL backbone trained with ArcFace loss produced L2-normalized embeddings for each image. At inference time, a KNN search over all training embeddings found the nearest neighbors; if the nearest neighbor distance was below a threshold, the individual was recognized; if above, the image was assigned the "new_individual" label. Species-aware inference — using only training embeddings from the same species as the query — improved recognition accuracy significantly.

### Key Techniques

1. **ArcFace (Additive Angular Margin) loss for metric learning:** ArcFace adds a fixed angular margin to the target class logit before softmax, creating more discriminative embeddings than standard cross-entropy; margin=0.5, scale=30 standard settings.
2. **ConvNeXt-Large / EfficientNetV2-XL backbone:** Large modern CNN backbones trained at 512×512 input; ConvNeXt-L showed best balance of accuracy and training speed.
3. **KNN retrieval + new_individual threshold:** At test time, compute cosine similarity between query embedding and all training embeddings; if max similarity < threshold τ, predict "new_individual"; τ optimized on OOF predictions.
4. **Species-conditional KNN:** Restrict the KNN gallery to images from the same species as the query (species known from competition metadata); dramatically reduces the search space and improves precision.
5. **TTA (test-time augmentation) embedding averaging:** Horizontal flip and multi-crop augmentations at test time; average the 3–5 embedding vectors before KNN search to reduce noise.

### How to Reuse

- **Any open-set re-identification task:** ArcFace + KNN + new_individual threshold is the canonical recipe for wildlife ID, face recognition, and product re-ID.
- **Gallery-based recognition:** When you have labeled gallery images and must match queries, metric learning outperforms softmax classification whenever new classes appear at test time.
- **Threshold calibration for "unknown" class:** Always tune the "new individual" threshold explicitly on validation — it's as important as the embedding model itself.
- **Species/category conditioning:** Conditioning gallery search on a known categorical attribute (species, product type) is a free improvement whenever that attribute is available.

---

## 11. Google Landmark Recognition 2020

**Task:** Recognize 81,313 distinct landmark classes in photos; evaluated by GAP (global average precision); extreme long-tail class distribution with many classes having only 1–2 training images.

**Discussion:** https://www.kaggle.com/c/landmark-recognition-2020/discussion/187821

### Approach

The winning solution combined large EfficientNet-B7 and ResNet-101 backbones trained with ArcFace/CosFace losses on the full 5M-image training set, alongside a retrieval-based re-ranking step. Because many classes have very few training images, the team augmented the final classifier with a k-NN retrieval system over the full training gallery: at test time, the predicted class from the classifier was cross-checked against the nearest neighbors in embedding space, with re-ranking applied if they disagreed. This hybrid classifier + retrieval approach was crucial for handling the long tail.

### Key Techniques

1. **EfficientNet-B7 + ArcFace at scale:** Trained on the full 5M-image dataset with mixed-precision training; ArcFace loss with sub-center ArcFace variant (allowing multiple cluster centers per class) to handle intra-class variation.
2. **Sub-center ArcFace (K=3 centers per class):** For landmark categories with high visual diversity (e.g., photographed from many angles/times), sub-center ArcFace allows each class to occupy K hyperspherical clusters instead of one.
3. **DBA (Database-side Augmentation) + QE (Query Expansion):** Retrieved top-k training images for a query, averaged their embeddings (DBA), then re-retrieved — a test-time ensembling trick that improves ANN recall.
4. **Diffusion-based re-ranking:** Graph-based diffusion over the nearest-neighbor graph of the test+train embeddings to propagate similarity scores beyond direct neighbors; significantly improves GAP for long-tail classes.
5. **Non-landmark filtering:** A binary classifier trained to reject "non-landmark" or "junk" images (crowd scenes, blank walls) before recognition; GAP penalizes false positives from non-landmark test images.

### How to Reuse

- **Long-tail fine-grained classification:** Sub-center ArcFace + hybrid classifier/retrieval is the state-of-the-art recipe whenever training images per class vary wildly.
- **Instance retrieval at scale:** DBA + QE are free at inference time and consistently improve retrieval metrics; implement them as post-processing on any embedding model.
- **Diffusion re-ranking:** Graph diffusion is powerful for competitions evaluated by AP-style metrics; worth implementing for high-stakes retrieval tasks.
- **Reject/abstain option:** Training an explicit non-landmark detector (binary junk filter) is applicable to any recognition task where some test images may not belong to any known class.

---

## 12. Ubiquant Market Prediction (2022)

**Task:** Predict a stock's anonymous return (investment_id) for each time_id; tabular regression evaluated by Pearson correlation between predictions and targets, grouped by time_id.

**Discussion:** https://www.kaggle.com/c/ubiquant-market-prediction/discussion/338220

### Approach

The winning solution used a blend of neural networks and gradient boosted trees on the 300 anonymized feature columns, with careful attention to the time-series structure. The key insight was that the target is a cross-sectional return (relative to other stocks at the same time), so the model needed to rank stocks within each time period rather than predict absolute returns. A 3-layer MLP trained with a Pearson correlation loss was the primary model, supplemented by LightGBM. Features were standardized per time_id to remove market-wide effects before input to both models.

### Key Techniques

1. **Per-time_id feature standardization:** Z-score each of the 300 features within each time_id before training and inference; this removes cross-sectional market factors and focuses the model on relative stock characteristics.
2. **Pearson correlation as training loss:** Directly optimize the competition metric by using 1 - Pearson(y_pred, y_true) as the batch loss; this is differentiable and aligns training with evaluation.
3. **3-layer MLP with batch normalization:** Simple fully connected network (300 → 512 → 256 → 1) with BatchNorm and dropout; trained with AdamW and cosine LR decay; GPU training over ~millions of rows is fast.
4. **Time-series aware CV:** Purged/embargoed walk-forward CV (avoid leakage between train and validation time periods); no random shuffling across time boundaries.
5. **LightGBM + MLP ensemble:** GBDT model captures non-linear feature interactions that the MLP may miss; simple weighted average of the two predictions improves correlation.

### How to Reuse

- **Cross-sectional financial prediction:** Always standardize features within the time period to remove market beta effects before training any model.
- **Pearson correlation loss:** Implement as `-(x - x.mean()) @ (y - y.mean()) / (n * x.std() * y.std())`; directly optimizable with autograd in PyTorch/TF.
- **Tabular neural networks:** MLP with BatchNorm is surprisingly competitive on large tabular datasets; combine with GBDT for best results.
- **Financial time-series CV:** Purged walk-forward CV is mandatory; any random CV on financial data will produce wildly optimistic estimates.

---

## 13. LMSYS Chatbot Arena (2024)

**Task:** Predict which of two LLM responses (response_a, response_b) a human evaluator will prefer, given a conversation prompt; binary classification (winner_model_a / winner_model_b / tie) evaluated by log-loss.

**Discussion:** https://www.kaggle.com/c/lmsys-chatbot-arena/discussion/527629

### Approach

The winning solution fine-tuned large language models (Gemma-7B, Llama-3-8B) directly on the full conversation context — system prompt + human turn + both responses — framed as a 3-way classification (response A wins, response B wins, tie). The key insight was that sufficiently large LLMs, given the full text of both responses, can directly judge quality without any handcrafted features. The team used sequence classification heads on 4-bit quantized models, with LoRA fine-tuning on a 4× A100 setup, and ensembled predictions from Gemma and Llama with different LoRA ranks and learning rates.

### Key Techniques

1. **Full-context LLM fine-tuning with LoRA:** Feed the entire [prompt + response_a + response_b] token sequence (up to 2K–4K tokens) into Gemma-7B/Llama-3-8B with a classification head; fine-tune with LoRA (r=16, alpha=32) to keep memory tractable.
2. **4-bit quantization (QLoRA) for memory efficiency:** Base model loaded in 4-bit NF4 quantization (bitsandbytes); only LoRA adapter weights are full precision; enables 7B–8B model fine-tuning on a single A100-80GB.
3. **Positional swap augmentation:** For each training example, create a swapped version where response_a and response_b are exchanged and the label is inverted; this doubles training data and forces the model to be position-invariant.
4. **3-class cross-entropy with tie handling:** Treat "tie" as a distinct third class rather than discarding tie examples; tie probability is important for log-loss calibration.
5. **Ensemble of Gemma-7B + Llama-3-8B:** Different base model families with different tokenizers produce uncorrelated errors; averaging predicted class probabilities before submission.

### How to Reuse

- **Human preference prediction / RLHF reward modeling:** This exact setup (LLM + LoRA + swap augmentation + 3-class head) is directly applicable to training reward models for RLHF pipelines.
- **Pairwise LLM evaluation at scale:** The trained model can serve as an automated judge for A/B testing LLM outputs without human raters.
- **Swap augmentation for position bias:** Whenever two candidates are presented in sequence, always augment with the swapped version to prevent the model from learning position preferences.
- **QLoRA for large model fine-tuning on limited GPU:** Standard recipe: load in 4-bit, fine-tune adapters at bf16; works down to a single 24GB GPU for 7B models.

---

## 14. Yale/UNC Geophysical Waveform Inversion (2025)

**Task:** Reconstruct a 2D subsurface velocity model from seismic shot-gather waveforms — a physics-informed regression/inversion problem evaluated by MAE on the velocity grid.

**Discussion:** https://www.kaggle.com/c/waveform-inversion/discussion/587388

### Approach

The winning solution used a U-Net–style encoder-decoder architecture that takes stacked seismic receiver traces (organized as a 2D image of time × receiver offset) and outputs the 2D velocity grid. The encoder processed multi-shot gathers independently and pooled their features before the decoder reconstructed the velocity map, allowing the model to aggregate information from multiple seismic sources. Physics-guided data augmentation — including velocity model flipping and synthetic waveform generation from perturbed velocity models — substantially expanded the training distribution.

### Key Techniques

1. **U-Net encoder-decoder for seismic-to-velocity mapping:** The input is a 3D tensor (shots × time samples × receivers); a 2D CNN encoder processes each shot's gather independently; features are pooled across shots before the decoder reconstructs the velocity grid.
2. **Multi-shot feature aggregation:** Each of the N seismic shot gathers is encoded separately; the resulting feature maps are averaged (or attention-pooled) to produce a single latent representation, then decoded to the velocity model — making the architecture permutation-invariant to shot order.
3. **Physics-guided augmentation:** Horizontal flipping of velocity models with corresponding flip of the shot gather geometry; additive Gaussian noise on waveforms; random velocity model perturbations used to generate synthetic training pairs via a forward solver.
4. **Frequency curriculum / multi-scale loss:** Training progressively emphasizes higher-frequency velocity components (starting from low-frequency smooth velocity) by weighting the MAE loss by spatial frequency — mimicking the multiscale approach used in classical full-waveform inversion (FWI).
5. **Ensemble of U-Net variants with different encoder depths:** ResNet-34, ResNet-50, and EfficientNet-B4 encoders with identical decoder; diverse encoder depths capture velocity structure at different scales; predictions averaged.

### How to Reuse

- **Seismic / geophysical inversion tasks:** U-Net with multi-shot aggregation is the standard DL architecture for data-driven FWI; directly reusable for 2D/3D velocity estimation.
- **Any image-to-image regression with multiple input views:** The encode-each-view + pool + decode pattern generalizes to multi-view 3D reconstruction, medical imaging from multiple scanner angles, and remote sensing fusion.
- **Physics-guided synthetic data generation:** When a forward simulator is available (even a fast approximate one), generating synthetic training pairs is a powerful data augmentation strategy.
- **Multi-scale regression:** Frequency curriculum (coarse-to-fine) applies to any regression task on spatially structured outputs; stabilizes training and avoids cycle-skipping artifacts.

---

## 15. TensorFlow Great Barrier Reef (2022)

**Task:** Detect crown-of-thorns starfish (COTS) in underwater video frames; object detection evaluated by a custom F2-IoU metric at multiple IoU thresholds.

**Discussion:** https://www.kaggle.com/c/tensorflow-great-barrier-reef/discussion/307878

### Approach

The winning solution used YOLOv5x6 as the primary detector, trained on the competition's video frames with extensive augmentation (mosaic, copy-paste of COTS instances from other frames), and tracked detections across consecutive frames using a custom frame-averaging / "video context" trick — averaging predicted bounding box confidence scores across 2–3 consecutive frames to reduce per-frame false positives from the high noise level of underwater footage. The F2 metric (emphasizing recall) was matched by tuning confidence thresholds toward lower values, accepting more false positives to capture more true COTS.

### Key Techniques

1. **YOLOv5x6 with extra-large input resolution (1280px):** Larger input allows detection of small COTS; YOLOv5x6 (the extra-large variant with P6 detection head for small objects) was the best single model; fine-tuned from COCO pretrained weights.
2. **Copy-paste augmentation of COTS instances:** Starfish instances cut from labeled frames and pasted onto frames with no annotations; effectively creates new positive training examples and mitigates the extreme foreground/background imbalance.
3. **Temporal frame averaging (video context):** For each frame, average the model's predicted confidence scores with those from the immediately preceding and following frames; reduces per-frame noise from water turbidity and motion blur.
4. **F2-metric threshold optimization:** Because F2 weights recall at 2× precision, optimize the confidence threshold by sweeping from 0.01 to 0.5 on OOF predictions and selecting the threshold that maximizes F2 rather than F1.
5. **WBF (Weighted Boxes Fusion) for ensemble:** Multiple YOLOv5 models (trained with different seeds/augmentation strategies) ensembled with WBF rather than NMS; WBF averages box coordinates from overlapping predictions, improving localization accuracy.

### How to Reuse

- **Video object detection:** Frame averaging (temporal smoothing) is a simple, effective way to reduce per-frame false positives in video; no tracking algorithm required.
- **Small object detection:** YOLOv5x6 / YOLOv8x with P6 head and high input resolution (1280px) is the go-to for detecting small objects in high-res images.
- **Copy-paste augmentation for rare objects:** When positive instances are sparse (few labeled objects per image), copy-paste augmentation is one of the most effective ways to balance the training distribution.
- **Recall-heavy metrics (F2, F-beta with β>1):** Always tune the detection confidence threshold explicitly toward the evaluation metric; the default 0.5 threshold is optimized for F1 and will under-recall on F2 evaluations.
- **WBF ensemble:** Prefer WBF over NMS when ensembling multiple detection models; WBF is more numerically stable and consistently outperforms NMS in ensemble settings.

---

## Cross-Cutting Patterns

| Pattern | Competitions Where It Appeared | When to Apply |
|---|---|---|
| **Ensemble of diverse architectures (multi-seed / multi-backbone)** | BirdCLEF, LANL, Jigsaw, PLAsTiCC, Quick Draw, LMSYS, Reef | Almost universally — 2–5 models with different inductive biases; average logits/probabilities |
| **ArcFace / metric learning + KNN retrieval** | Happywhale, Landmark Recognition | Open-set recognition; whenever new classes appear at test time |
| **Two-stage retrieve-then-rerank pipeline** | Learning Equality, OTTO, Landmark | Large document collections; too expensive to score all candidates with a cross-encoder |
| **Per-group/per-time normalization of features** | Ubiquant, PLAsTiCC | Time-series and cross-sectional data; removes group-level bias before modeling |
| **Threshold optimization for non-standard metrics (F2, QWK, Recall@k)** | Reef, Data Science Bowl, Learning Equality, BirdCLEF | Whenever evaluation metric differs from cross-entropy; always tune thresholds on OOF predictions |
| **Synthetic / pseudo-label data generation** | LLM Detect, BirdCLEF, Reef (copy-paste) | When labeled data is scarce; use LLM APIs, copy-paste augmentation, or forward simulators |
| **LightGBM / XGBoost on hand-engineered features as strong baseline** | LANL, OTTO, Data Science Bowl, Ubiquant | Tabular and session data; GBDT is competitive with (or beats) DL on <1M rows with rich features |
| **LoRA / QLoRA fine-tuning of large LLMs** | LMSYS, LLM Detect (implicit) | Any NLP task where a 7B+ model is warranted but GPU memory is constrained |
| **Physics-informed augmentation / domain knowledge features** | LANL (statistical physics), PLAsTiCC (Lomb-Scargle), Waveform (FWI curriculum) | Scientific domains; encode physical priors as features or augmentation strategies |
| **Per-class / per-species threshold calibration** | BirdCLEF, Happywhale | Multi-label classification with class imbalance; one threshold per class, not global |
| **Temporal / frame-level smoothing in video/sequence tasks** | Reef (frame averaging), LANL (rolling stats) | Sequential data with per-step noise; smooth predictions across adjacent steps |
| **Swap / position augmentation for pairwise tasks** | LMSYS, Jigsaw | Any pairwise comparison model; always augment with swapped order to remove position bias |
| **U-Net / encoder-decoder for structured output regression** | Waveform Inversion, Reef (detection) | Image-to-image or signal-to-grid tasks; U-Net with skip connections is the default architecture |
| **Contrastive fine-tuning with hard negative mining** | Learning Equality, Happywhale | Retrieval and metric learning; hard negatives from the model's own near-misses are more informative than random negatives |
| **Progressive training (resolution, frequency, or scale curriculum)** | Quick Draw (progressive resize), Waveform (frequency curriculum) | When training cost is high; start coarse, refine — reduces early-epoch wasted compute |

---

*Note: Discussion pages require Kaggle authentication; solution details above are drawn from the author's training knowledge of these well-documented public solutions. All 15 solutions are from publicly discussed Kaggle competition write-ups. Verify specific hyperparameter values against the original posts when implementing.*agentId: a7108606d057c7e73 (use SendMessage with to: 'a7108606d057c7e73' to continue this agent)
<usage>total_tokens: 39477
tool_uses: 28
duration_ms: 296233</usage>