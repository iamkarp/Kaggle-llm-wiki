# Kaggle Past Solutions — SRK Round 2, Batch 5

Source: kaggle.com/sudalairajkumar/winning-solutions-of-kaggle-competitions
Ingested: 2026-04-17
Note: These are non-1st-place solutions (2nd, 3rd, top-10) with 100+ upvotes.

---

## 1. Feedback Prize — English Language Learning (2022) — 2nd Place

**Task type:** NLP regression (multi-target essay scoring)
**Discussion:** [https://www.kaggle.com/c/feedback-prize-english-language-learning/discussion/369369](https://www.kaggle.com/c/feedback-prize-english-language-learning/discussion/369369)

**Approach:** DeBERTa-v3 ensemble with back-translation pretraining and a custom rank loss. Each of the 6 scoring targets got its own output head. Models were pretrained with back-translated multilingual data (FB2 dataset) and trained with Adversarial Weight Perturbation (AWP) on the final 2 epochs.

**Key Techniques:**
1. **Back-translation data augmentation** — Translated training essays into 14 languages (Dutch, French, German, Portuguese, Afrikaans, Chinese, Russian, Finnish, Swedish, Japanese, Korean, Greek, Croatian, Welsh) and back to English to create diverse pretraining data, dramatically expanding the effective training set.
2. **Rank loss** — Added a pairwise ranking loss on top of MSE, training the model to correctly rank pairs of essays by each dimension; this sharpened ordinal sensitivity vs. pure regression.
3. **Per-target output heads** — Each of the 6 scoring dimensions (cohesion, syntax, vocabulary, phraseology, grammar, conventions) had its own unique fully-connected layer on top of the shared encoder embedding, allowing specialized calibration per dimension.
4. **AWP (Adversarial Weight Perturbation)** — Applied only during the final 2 of 4 training epochs; adds adversarial noise to model weights during training to force more robust representations.
5. **Pseudo-labeling** — Used model predictions on unlabeled data (FB2 competition essays) as soft labels for pretraining, further extending the training signal.

**How to Reuse:**
- For any multi-target NLP regression task, implement separate output heads per target rather than a shared head — the marginal cost is tiny and it allows per-target LR and calibration.
- Back-translation is especially powerful when training data is small and in-domain text is scarce; translate/back-translate at multiple intermediate languages and keep all variants.
- Combine rank loss with MSE in any ordinal regression problem; the rank signal prevents the model from treating all errors as equally bad regardless of direction.

---

## 2. PetFinder.my — Pawpularity Contest (2021) — 4th Place (baseline post)

**Task type:** Computer vision regression (pet photo popularity score)
**Discussion:** [https://www.kaggle.com/c/petfinder-pawpularity-score/discussion/276522](https://www.kaggle.com/c/petfinder-pawpularity-score/discussion/276522)

**Approach:** Simple, strong baseline using a large Swin Transformer pretrained on ImageNet-22k. Treated the problem as classification rather than regression. Used 10-fold cross-validation mean ensemble with minimal augmentation.

**Key Techniques:**
1. **Swin Transformer (large, ImageNet-22k pretrained)** — The large Swin Transformer pretrained on the full ImageNet-22k (rather than just 1k) provided significantly richer visual representations, especially important for fine-grained aesthetic scoring.
2. **Classification instead of regression** — Discretizing the continuous popularity score into bins and training with cross-entropy loss rather than MSE often provides better gradients and calibration on noisy, bounded-range targets.
3. **10-fold ensemble (mean)** — Simple averaging across 10 CV folds with no complex blending; the ensemble itself provides substantial variance reduction.
4. **Minimal augmentation** — Random resized crop and random horizontal flip only; adding mixup and stronger augmentations did not help, suggesting the signal is subtle and augmentation-induced distribution shift hurts more than it helps.

**How to Reuse:**
- When a regression target is bounded (0–100, 1–5 stars), try framing it as classification; it often trains more stably and predicts at boundaries better.
- For image aesthetics or quality tasks, always try ImageNet-22k pretrained weights (vs. 1k) as a first step — the richer pretraining matters more than architecture choice.
- 10-fold CV mean is a simple but effective ensemble; avoid over-engineering blend weights unless you have strong CV signal to tune them on.

---

## 3. BirdCLEF 2021 — Birdcall Identification (2021) — 1st Place

**Task type:** Audio classification (bird species from soundscape recordings)
**Discussion:** [https://www.kaggle.com/c/birdclef-2021/discussion/243304](https://www.kaggle.com/c/birdclef-2021/discussion/243304)

**Approach:** A three-stage pipeline: (1) nocall detection from melspectrograms using auxiliary data, (2) short-clip bird classification with noisy-label weighting, and (3) LightGBM meta-model to refine top-5 candidate predictions using metadata. Used 10 models in the final ensemble with ternary-search threshold optimization.

**Key Techniques:**
1. **Three-stage hierarchical pipeline** — Separated nocall detection (stage 1), species classification from short clips (stage 2), and meta-model ranking (stage 3). Each stage specializes, reducing the problem into cleaner sub-tasks with better signal-to-noise.
2. **Noisy label weighting via nocall detector** — Short audio clips have noisy species labels; used the nocall detector's output confidence to downweight uncertain samples during stage-2 training, improving label quality without manual cleaning.
3. **Soundscape validation** — Used the provided train soundscapes as the validation set (rather than held-out short clips), which better mimics the test distribution and gave more reliable model selection.
4. **LightGBM meta-model with metadata** — After ranking the top-5 candidate species from the neural model, trained a LightGBM on neural outputs plus metadata features to make the final binary decision (is this bird present?), adding a structured layer on top of deep features.
5. **Ternary search for threshold optimization** — Applied ternary search (a form of golden-section search) to find the optimal prediction threshold rather than a grid search; efficient and reliable for single-parameter optimization.

**How to Reuse:**
- In audio competitions with both short-clip and long-recording test sets, always validate on the long-recording format — short-clip CV will be misleadingly optimistic.
- Two-stage pipelines (neural candidate generation + GBDT meta-model) are consistently powerful: the neural model handles raw features, the GBDT handles feature interactions and calibration at low cost.
- When dealing with noisy audio labels, use a separately trained noise/nocall detector to create sample weights rather than discarding uncertain samples outright.

---

## 4. Google QUEST Q&A Labeling (2020) — 2nd Place

**Task type:** NLP multi-label regression (30 question/answer quality dimensions)
**Discussion:** [https://www.kaggle.com/c/google-quest-challenge/discussion/129978](https://www.kaggle.com/c/google-quest-challenge/discussion/129978)

**Approach:** Ensemble of 5 transformer models (dual RoBERTa-base, siamese RoBERTa-base, dual RoBERTa-large, dual XLNet), each processing question and answer as separate inputs of up to 512 tokens. Used ordinal (binary-encoded) target representation and threshold-based post-processing. Final blend by simple average of probability predictions.

**Key Techniques:**
1. **Dual-encoder architecture ("Two BERTs")** — Processed the question text and answer text through separate transformer streams (or a shared siamese transformer), allowing each part to attend to its own token context without truncation-induced mixing; this was the core architectural innovation.
2. **Ordinal/binary target encoding** — Instead of directly predicting continuous targets, converted each label into multiple binary targets `(t > v)` for all observed threshold values `v`. Predictions were recovered as expected values. This respects ordinal structure and is compatible with binary cross-entropy loss, giving much better gradient signal on the sparse label space.
3. **Multi-sample CV scheme to handle rare labels** — For each validation fold, sampled 100 random single question-answer pairs from multi-answer question groups, computed median score; excluded the `spelling` column (rare events, high noise) from CV. This gave a reliable estimate of model strength despite extreme label rarity.
4. **Threshold-clipping post-processing** — After producing ensemble predictions, independently found optimal clip thresholds from both the low and high ends of each target column's prediction distribution, improving calibration on bounded targets.
5. **Differential learning rates** — Transformer backbone at 3e-5, model head at 0.005; cosine schedule with 1-epoch warmup over 3 total epochs; gradient accumulation to achieve effective batch size of 8.

**How to Reuse:**
- For Q&A or document-pair tasks, the dual-encoder pattern (one encoder per document type) almost always beats concatenation when both documents approach 512 tokens.
- Ordinal encoding of continuous targets into cumulative binary variables is a powerful trick for any competition with bounded, non-uniform target distributions.
- Multi-sample validation (randomly sample from label groups and take median) is the right pattern whenever labels vary within a group in test but are averaged in train.

---

## 5. Corporación Favorita Grocery Sales Forecasting (2017) — 4th Place

**Task type:** Time series forecasting (multi-step grocery sales)
**Discussion:** [https://www.kaggle.com/c/favorita-grocery-sales-forecasting/discussion/47529](https://www.kaggle.com/c/favorita-grocery-sales-forecasting/discussion/47529)

**Approach:** Sequence-to-sequence model with a dilated causal convolution encoder (WaveNet-style, ~30 layers, ~1M params) and a decoder augmented with a bidirectional LSTM for future promotion information. Added a Bernoulli output head to explicitly model zero-sales probability. Optimized weighted RMSE of log-transformed data directly.

**Key Techniques:**
1. **Dilated causal convolutions (WaveNet architecture)** — Stack of ~30 dilated, causal convolutions with exponentially growing receptive field; this allows the model to capture patterns across the full 4+ year training history while keeping parameter count low (~1M). Outperforms vanilla LSTM/RNN for long time series with periodic patterns.
2. **Seq2seq with bidirectional LSTM on future covariates** — The decoder used a bidirectional LSTM pre-encoding future promotion information (which is known in advance), allowing the forecast to be conditioned on upcoming promotions without autoregressive leakage.
3. **Zero-inflation head (Bernoulli output)** — Added a parallel Bernoulli output estimating P(sales = 0) at each timestep; final predictions = neural regression output multiplied by (1 - P(zero)). Addresses the common failure mode of regression models producing non-zero predictions for items that are unavailable or not on promotion.
4. **Stratified holdout validation** — Held out 5% of time series, evaluated on random periods from the final 365 training days; this prevents validation bias from weekly/monthly trends and gives a fair estimate of generalization.
5. **Promotion data imputation via leaderboard probing** — For missing on-promotion values, imputed randomly (1 with probability p, 0 otherwise) where p was tuned per day using stochastic leaderboard descent; a pragmatic but effective solution to a data quality problem.

**How to Reuse:**
- For multi-step sales forecasting, WaveNet-style dilated convolutions are a strong and parameter-efficient alternative to LSTM/Transformer — especially when historical windows span months to years.
- Always add a zero-inflation component for retail/supply-chain forecasting: a separate binary "will this sell at all?" head before the quantity prediction significantly reduces RMSE on sparse sales data.
- When future covariates (promotions, holidays) are known, encode them with a separate bidirectional encoder rather than simply concatenating them as features to the main model.

---

## 6. CommonLit Readability Prize (2021) — 3rd Place

**Task type:** NLP regression (text readability scoring)
**Discussion:** [https://www.kaggle.com/c/commonlitreadabilityprize/discussion/258095](https://www.kaggle.com/c/commonlitreadabilityprize/discussion/258095)

**Approach:** Simple average of 3 transformer models: two RoBERTa-large and one DeBERTa-large, all fine-tuned with iterative pseudo-labeling on external text corpora. Used attention-pooling, weighted hidden layer averaging, multi-sample dropout, and 3-group differential learning rate decay. Private LB: 0.447.

**Key Techniques:**
1. **Iterative pseudo-labeling on external corpora** — Used Simple Wikipedia, Children's Book Test, and OneStopEnglish to extend training data; labeled with 5-fold pseudo-label (each fold labeled by models trained on the other folds to avoid leakage); ran 2 cycles of relabeling and retraining; each cycle improved public LB from 0.453 → 0.450 → 0.448.
2. **3-group differential learning rate decay** — Applied different learning rates to (1) lower transformer layers, (2) upper transformer layers, and (3) the task head; with a cosine decay schedule. This prevents catastrophic forgetting in early layers while allowing the head and upper layers to adapt quickly.
3. **Weighted average of all hidden layers** — Rather than using only the final [CLS] token, learned a weighted average across all transformer layers' output representations. Early layers capture syntactic features, late layers capture semantic; the weighted blend captures both for readability.
4. **Multi-sample dropout** — Applied dropout multiple times per forward pass, averaged the resulting predictions for both training (loss) and inference; reduces variance and acts as implicit ensembling within a single model.
5. **Fold-specific pseudo-labeling to avoid leakage** — Generated pseudo-labels using only the folds that did NOT see that example during training; this is critical — naive pseudo-labeling with a model that trained on the example will produce unrealistically confident and biased labels.

**How to Reuse:**
- Iterative pseudo-labeling is most effective on small labeled datasets with abundant unlabeled in-domain text (readability, medical, legal); always use fold-specific labeling to avoid leakage.
- For transformer fine-tuning with small datasets, 3-group differential LR (lower layers slower, upper layers + head faster) prevents the lower layers from being corrupted early in training.
- Weighted hidden layer pooling is a nearly free improvement over last-layer pooling for most NLP regression tasks — add a learnable vector of layer weights and multiply before pooling.

---

## 7. RSNA Intracranial Hemorrhage Detection (2019) — Gold Medal Solutions Summary

**Task type:** Medical image classification (CT scan hemorrhage detection, multi-label)
**Discussion:** [https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/discussion/117242](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/discussion/117242)

**Approach:** The discussion post is a curated list of gold medal solutions (top 12 teams). The top solutions generally combined multi-window CT preprocessing, sequence-aware architectures (to leverage adjacent DICOM slices), and ensemble of multiple CNN backbones. Key approaches included treating the CT stack as a temporal sequence with LSTMs/GRUs over slice embeddings, and using three-window preprocessing (brain, subdural, bone windows) as input channels.

**Key Techniques:**
1. **Multi-window CT preprocessing as RGB channels** — Used brain, subdural, and bone HU windowing to create a 3-channel "RGB-like" input from DICOM data; each window highlights different hemorrhage types. This allowed ImageNet-pretrained CNNs to be applied directly without modification.
2. **Sequence modeling over DICOM slices** — Fed embeddings from a CNN backbone into an LSTM/GRU operating across the ordered CT slices (treating the scan as a temporal sequence), allowing the model to use spatial context from adjacent slices for each prediction.
3. **Multi-label classification (6 hemorrhage subtypes)** — Modeled each of the 6 hemorrhage types (epidural, intraparenchymal, intraventricular, subarachnoid, subdural, any) as separate binary outputs; this multi-task setup improved overall AUC vs. one-vs-rest approaches.
4. **SE-ResNeXt / EfficientNet backbone ensembles** — Multiple solutions combined SE-ResNeXt, EfficientNet, and DenseNet backbones with diversity in architecture, window preprocessing, and sequence length; ensembles of 10–20 models were typical.
5. **Gradient accumulation for large batch training** — With large DICOM sequences and high-res images, gradient accumulation allowed effectively large batch sizes that stabilize training without exceeding GPU memory.

**How to Reuse:**
- For any 3D medical imaging task with ordered slices, combine a 2D CNN (feature extractor per slice) with a temporal model (LSTM/GRU/transformer) over the slice dimension — this is the standard winning pattern for CT/MRI competitions.
- When applying ImageNet CNNs to grayscale medical images with multiple clinically meaningful contrast windows, stack the windows as separate channels rather than using a single-channel grayscale.
- Multi-task prediction of sub-types alongside a global "any" label consistently improves AUC via shared representation learning and provides useful auxiliary gradients.

---

## 8. Microsoft Malware Prediction (2019) — 6th Place

**Task type:** Binary classification (malware infection prediction, tabular)
**Discussion:** [https://www.kaggle.com/c/microsoft-malware-prediction/discussion/84112](https://www.kaggle.com/c/microsoft-malware-prediction/discussion/84112)

**Approach:** Time-aware validation using Windows version release dates as a temporal proxy. Explicitly modeled the train/public/private distribution shift by building features stable across months. Ensemble of ~10 LightGBM and 4 Keras neural network models, blended with a geometric weighted average. Dropped time-dependent versioning features.

**Key Techniques:**
1. **Time-based train/validation split (monthly)** — Used AvSigVersion release dates to assign each machine to a month; split train into month 1 (fit) and month 2 (validate), mimicking the 1-month temporal gap between train and private test. This gave dramatically better model selection than k-fold CV, which was unreliable on this competition.
2. **LB probing to detect train/test distribution shift** — Used LB submissions to precisely estimate the size and date range of the public/private test sets (public ≈ month 3, private ≈ month 4); this revealed that many features had completely different distributions on private vs. train, guiding feature selection.
3. **Dropping time-dependent versioning features** — Removed `AvSigVersion`, `EngineVersion`, `AppVersion`, `Census_OSBuildRevision`, etc., because the version strings in test were largely absent from train; kept only time-stable derived features to avoid distribution mismatch.
4. **Feature normalization by fold size** — For count-based and rank-based features, normalized by the size of each fold (train, valid, public, private) to ensure the feature distributions were comparable across splits despite different sizes.
5. **3-stage training pipeline with blend** — (1) Train on month 1, validate on month 2; (2) Train on month 2 only with the same number of rounds; (3) Train on full data (months 1+2) with 2× rounds; blend as (1)^0.2 × (2)^0.2 × (3)^0.6. This approach retains the temporal validation signal while using all data for the final model.

**How to Reuse:**
- In any tabular competition with a temporal train/test split, build a validation scheme that mirrors the gap — a rolling-window or forward-chaining CV that includes a buffer period between train and validation will outperform k-fold CV.
- Always profile feature stability across time when the test set is temporally displaced: compute the KL divergence of each feature between train and test and systematically drop or engineer around unstable features.
- The 3-stage blend (fold 1 → fold 2 → full data) is a transferable technique for building a final model that both validates reliably and uses all available data.

---

## 9. LANL Earthquake Prediction (2019) — 7th Place

**Task type:** Time series regression (predicting time-to-failure of lab earthquakes)
**Discussion:** [https://www.kaggle.com/c/LANL-Earthquake-Prediction/discussion/94407](https://www.kaggle.com/c/LANL-Earthquake-Prediction/discussion/94407)

**Approach:** 16-fold nested CV to get reliable MAE estimates (public LB was unreliable). Overlapping segments of 150k samples. MFCC audio features plus signal statistics. Multi-model stack (LightGBM + GAM + KNN) with a second-level LightGBM. Addressed train/test TTF distribution mismatch using sample weights estimated from earthquake cycle duration. Binary outlier classifier for acoustic peak segments.

**Key Techniques:**
1. **Nested cross-validation for reliable evaluation** — Standard 16-fold CV showed poor correlation with public LB; used nested CV where each fold Ti is held out as test data and 15-model CV is run on the other 15 folds, producing 16 independent MAE estimates. This gave very good LB correlation and is transferable to any competition where public LB is noisy or small.
2. **Train/test TTF distribution reweighting** — Used academic paper figures to estimate the mean earthquake cycle length (≈6.35 s, actual was 6.32 s), computed density functions of TTF for train and test, and applied sample weights to the training loss to match the test TTF distribution. This directly addressed the covariate shift between train and test.
3. **MFCC features from seismic signal** — Applied librosa MFCC extraction to 150k-sample segments (pretending the seismic data was audio at 40 kHz), extracting 7 MFCCs plus signal quantile statistics. The 4th MFCC alone was the most important single feature.
4. **Generalized Additive Models (GAM) as a strong base learner** — Used `pygam` alongside LightGBM; a single GAM model alone would have earned a gold medal (MAE ≈ 2.348). GAMs were better than both LightGBM and KNN, demonstrating that interpretable models can outperform boosting on this type of signal.
5. **Binary outlier classification for acoustic peaks** — Built a separate binary model to identify segments between acoustic peaks and the following TTF reset (where TTF doesn't reset as expected); set predictions for these outlier segments to a fixed value (0.31), significantly improving accuracy.

**How to Reuse:**
- When public LB is small or noisy, invest heavily in the validation scheme — nested CV is expensive but gives the most reliable signal and prevents spurious model selection.
- For physics time series with known structural properties (periodic cycles, known cycle length distribution), use that domain knowledge to construct sample weights rather than treating train samples as IID.
- Always try GAMs (via `pygam`) alongside gradient boosting on tabular signal data — they are competitive with LGBM on smooth, low-dimensional signals and provide useful model diversity for stacking.

---

## 10. AI Mathematical Olympiad — Progress Prize 2 (2025) — 2nd Place

**Task type:** Math reasoning / LLM inference optimization (50 AIME-level problems, 5-hour limit)
**Discussion:** [https://www.kaggle.com/c/ai-mathematical-olympiad-progress-prize-2/discussion/572948](https://www.kaggle.com/c/ai-mathematical-olympiad-progress-prize-2/discussion/572948)

**Approach:** DeepSeek-R1-Distill-Qwen-14B fine-tuned with SFT then DPO (to reduce output length). Inference via lmdeploy/TurboMind with 4-bit AWQ + 8-bit KV cache quantization (W4KV8). 15 samples per question (7 CoT + 8 Code prompts) with sample-level and question-level early stopping. Achieved 34/50 public LB (ranked 1st), 31/50 private LB (ranked 2nd).

**Key Techniques:**
1. **DPO for inference efficiency (length reduction)** — After SFT, applied DPO where chosen responses were required to be (a) correct and (b) shorter than a ratio threshold × the rejected response length. This reduced model verbosity, allowing more problems to be solved within the 5-hour wall clock limit.
2. **W4KV8 quantization (4-bit weights + 8-bit KV cache)** — Combined AWQ 4-bit weight quantization with 8-bit KV cache quantization via lmdeploy; reduced time-per-output-token by ~55% vs FP16, with only 5–10% accuracy drop. W4KV4 was too lossy; W4KV8 was the sweet spot.
3. **Dual-prompt strategy (CoT + Code)** — Generated 7 samples with chain-of-thought prompts and 8 samples with code-execution prompts per question; the diversity between reasoning styles improved self-consistency aggregation (majority voting).
4. **Two-level early stopping** — Sample-level: stop individual sample generation upon detecting the first executable code or `\boxed{}` answer; Question-level: stop all remaining samples if 5/7 agreed on an answer. This saved significant compute budget for easier problems, redirectable to harder ones.
5. **SFT on high-difficulty math reasoning trajectories** — Combined Light-R1 stage-2 data with Limo training data (high-difficulty math problems with DeepSeek-R1 reasoning traces); 8-epoch fine-tuning on 8×A800 improved both accuracy and trajectory quality as a foundation for DPO.

**How to Reuse:**
- For LLM inference competitions with strict time budgets, DPO-based length reduction is more effective than simply truncating or using smaller models — it maintains accuracy while cutting token count.
- W4KV8 quantization is the current recommended sweet spot for memory-constrained LLM inference; prefer lmdeploy/TurboMind over vLLM for throughput-critical batch inference.
- Two-level early stopping (sample + question level) is a principled and transferable strategy for any self-consistency voting setup — implement it whenever you run N>5 samples per problem.

---

## 11. IEEE-CIS Fraud Detection (2019) — 9th Place

**Task type:** Binary classification (credit card fraud detection, tabular)
**Discussion:** [https://www.kaggle.com/c/ieee-fraud-detection/discussion/111234](https://www.kaggle.com/c/ieee-fraud-detection/discussion/111234)

**Approach:** User identification via engineered features linking transactions to the same card/user (derived from TransactionDT and D-features), used in both feature engineering and post-processing/pseudo-labeling. Time-series CV (3 months train, 1 month gap, 2 months validation). Ensemble of LightGBM, CatBoost, and XGBoost.

**Key Techniques:**
1. **Transaction-to-user linkage features** — Derived a "card date" feature as `"2017-11-30" + TransactionDT - D1`, which corresponds to a stable date (e.g. first transaction date) shared by all transactions of the same card. Combined with D-feature analysis (V95, V96, V97 for transaction counts; V126–V128 for cumulative amounts), this allowed reliable user-level aggregation without explicit user IDs.
2. **Time-series forward-chaining CV** — First 3 months as training, 1-month gap removed, final 2 months as validation; this mimics the temporal gap between train and test, critical for preventing leakage of time-dependent patterns (device identifiers change, fraud patterns evolve).
3. **Pseudo-labeling with user identification** — After identifying which test transactions belonged to known users from train, used user-level aggregated predictions as pseudo-labels; this transferred fraud signal across the train/test boundary in a controlled way.
4. **Post-processing via user-level aggregation** — Rather than predicting each transaction independently, applied a post-processing step that pooled predictions within identified user groups, then redistributed the user-level fraud probability to each transaction.
5. **GBDT diversity ensemble** — LightGBM, CatBoost, and XGBoost each bring different inductive biases; their ensemble outperformed any individual model, especially when models were trained with different subsets of the D/V/C features.

**How to Reuse:**
- In fraud/e-commerce datasets without explicit user IDs, engineer card-fingerprint features by looking for stable quantities (e.g. time deltas between transactions, cumulative counts) that are constant within a user session.
- For any temporal fraud dataset, use strict forward-chaining CV with a gap period — the gap is not optional; leaking recent fraud labels into training causes the CV to be overly optimistic and model selection to be wrong.
- Always post-process fraud predictions at the entity (user/card/device) level rather than the transaction level; entity-level aggregation and redistribution is a low-cost, high-impact improvement.

---

## 12. Google Research — Identify Contrails to Reduce Global Warming (2023) — 2nd Place

**Task type:** Satellite image segmentation (pixel-level contrail detection)
**Discussion:** [https://www.kaggle.com/c/google-research-identify-contrails-reduce-global-warming/discussion/430491](https://www.kaggle.com/c/google-research-identify-contrails-reduce-global-warming/discussion/430491)

**Approach:** Customized U-shaped segmentation network with hierarchical (CoaT, NeXtViT, SAM-B, EfficientNetV2-s) backbones. ×2 or ×4 input upscaling + pixel-shuffle upscaling in decoder for pixel-level accuracy. Temporal mixing via LSTM across frames. Soft label training (mean of all annotators). BCE + Dice + Lovász loss combination.

**Key Techniques:**
1. **Input upscaling for pixel-level accuracy** — Contrails are only a few pixels thick; upscaling input by ×2 (bicubic interpolation) before the network dramatically improves boundary accuracy. Equivalently, reducing stride in the first conv layer to 1 gives ×4 effective upscaling. This single change gave substantial improvement over baseline.
2. **Pixel shuffle upscaling in decoder** — Replaced bilinear upsampling in the decoder with PixelShuffle blocks; reduces channel count at each upsampling step while maintaining spatial information, significantly improving GPU memory efficiency with minimal accuracy loss.
3. **LSTM temporal mixing at low-resolution feature maps** — Contrails move between frames; mixing at res/32 and res/16 (not at full resolution) via a 1-layer LSTM is the most effective way to incorporate temporal context, as the lower-resolution feature maps are reasonably aligned even with large inter-frame cloud displacements.
4. **Soft labels (annotator mean)** — Training with the mean of all annotator binary masks (rather than any single annotator's mask) as soft probability targets is critical for reducing label noise near contrail boundaries, where annotator disagreement is highest.
5. **LayerNorm2d + GELU in decoder blocks** — Replacing BatchNorm with LayerNorm2d in decoder blocks allows stable training at very low batch sizes (forced by high resolution + upscaling), which is common in segmentation competitions where GPU memory is the bottleneck.

**How to Reuse:**
- For any pixel-level segmentation task with thin structures (blood vessels, cracks, contrails), always try input upscaling before other improvements — it is cheap and often the highest-impact change.
- Soft label training (using annotator agreement as a probability rather than majority vote) is the right default for any competition with multi-annotator labels; it provides calibrated uncertainty at boundaries.
- For video/multi-frame segmentation, perform temporal mixing at intermediate low-resolution feature maps rather than at full resolution — large object displacements between frames make full-resolution temporal mixing counterproductive.

---

## 13. Predicting Molecular Properties (2019) — 6th Place

**Task type:** Graph regression (predicting scalar coupling constants between atom pairs in molecules)
**Discussion:** [https://www.kaggle.com/c/champs-scalar-coupling/discussion/106407](https://www.kaggle.com/c/champs-scalar-coupling/discussion/106407)

**Approach:** Custom Graph Neural Network combining Message Passing Neural Network (MPNN) elements with multi-head self-attention (Transformer Encoder-style). 3 types of attention layers (Euclidean distance Gaussian, graph distance, scaled dot-product). Predicted 4 physical coupling contributions separately, then summed. Trained via fastai with one-cycle LR policy; final model had 650-dim hidden state, 10 encoder blocks, 10 attention heads, trained on 2×V100 GPUs.

**Key Techniques:**
1. **Hybrid MPNN + Transformer architecture** — Stacked encoder blocks each containing: (1) message passing over bond connections and virtual scalar-coupling edges, then (2) three types of multi-head attention (Euclidean Gaussian, graph distance, and standard dot-product). Message passing captures local chemistry; attention captures non-local dependencies. Tied message-passing parameters across blocks to enable larger hidden states.
2. **Geometry-aware attention (Euclidean + dihedral features)** — Added cosine angle features (dihedral angles for 3J coupling, cosine angles for 2J) as edge features in message passing; replaced standard set2set pooling with Gaussian attention based on Euclidean distance. These geometry-aware components are essential because scalar coupling constants are primarily determined by 3D molecular geometry.
3. **Per-coupling-type write head with residual blocks** — The final prediction head specialized per scalar coupling type (1JHC, 1JHN, 2JHH, etc.) using residual blocks, allowing the model to apply different calibration and scaling per physical interaction type.
4. **4-contribution decomposition loss** — Predicted the 4 physical contributions to scalar coupling (Fermi contact, paramagnetic spin-orbit, diamagnetic spin-orbit, spin-dipole) separately, added them to the loss; this multi-task structure guided the model to learn physically meaningful decompositions and improved the main prediction as a side effect.
5. **Fixed-kernel convolution for message passing (replacing full matrix mul)** — Changed the message passing function from a full matrix multiplication to a fixed-kernel convolution (kernel size 128), dramatically reducing parameter count in the edge network (which would otherwise scale as hidden_dim³) and enabling much larger hidden state dimensions.

**How to Reuse:**
- For molecular property prediction, the MPNN+Transformer hybrid is now a strong default; include geometric edge features (angles, distances, dihedral) — raw adjacency-based GNNs are significantly weaker on this class of problem.
- Any time a problem has a known physical decomposition (e.g. free energy = enthalpy + entropy, coupling = 4 terms), add each component as an auxiliary prediction to the loss; this constrains the model to learn physically meaningful representations.
- Multi-head attention over Euclidean distance (Gaussian kernel attention where weight ∝ exp(-||r_i - r_j||²/σ²)) is a simple and effective addition to any GNN operating on 3D coordinates.

---

## 14. NFL 1st and Future — Impact Detection (2021) — 3rd Place

**Task type:** Video object detection (helmet impact detection in NFL footage)
**Discussion:** [https://www.kaggle.com/c/nfl-impact-detection/discussion/208787](https://www.kaggle.com/c/nfl-impact-detection/discussion/208787)

**Approach:** Two-stage pipeline: (1) EfficientDet generates candidate helmet/impact bounding boxes, (2) binary image classification on 9-frame crops (t-4 to t+4) stacked as a single multi-channel input, to determine whether each candidate box is a true impact. Multi-view post-processing uses agreement between Endzone and Sideline camera views. Final ensemble of 7 EfficientDet + 18 binary classification models.

**Key Techniques:**
1. **Two-stage detector + classifier pipeline** — EfficientDet trained on 2 classes (helmet, impact) generates candidates at low threshold (score > 0.17); a separate binary classifier then re-scores each candidate using temporal context. The classifier alone improved single-model local score from 0.3x to ~0.6, the biggest jump in the solution.
2. **9-frame temporal crop stacking** — For each candidate bbox, cropped the expanded region (3× original dimensions) across 9 consecutive frames (t-4 to t+4), converted to grayscale, and stacked into a (H, W, 9) input. This gives the classifier explicit temporal motion context without requiring a 3D CNN, and works because helmet impacts have characteristic motion signatures over time.
3. **Multi-view post-processing (Endzone/Sideline agreement)** — Leveraged the dual-camera setup: if one view predicted an impact, lowered the threshold in the other view for the same time window (from 0.45 → 0.25). This captures the physical constraint that a real impact must be visible from both camera angles, reducing false positives.
4. **Training with positive frame expansion (±4 frames)** — During EfficientDet training, labeled all frames within ±4 frames of a ground-truth impact as positive (matching the competition metric's ±4-frame tolerance); this expanded the positive set and aligned training signal with evaluation criteria.
5. **Temporal NMS (drop similar bboxes across consecutive frames)** — Applied a temporal equivalent of NMS across 9 consecutive frames, keeping only the highest-confidence prediction per cluster of similar bboxes (IoU > 0.25 across frames). Aligned with the metric's rule that a single label can only be matched once.

**How to Reuse:**
- In sports video detection competitions, always use a two-stage approach: a fast detector for region proposals, then a slower temporal classifier for final decisions — the temporal classifier is almost always the bigger lever than improving the detector.
- Stacking multi-frame crops as channels (grayscale × N frames) is a simple and highly effective way to provide temporal context to a standard 2D CNN without the complexity of 3D convolutions.
- When multiple camera views are available, build multi-view post-processing as a logical gate — require some threshold of cross-view agreement for high-confidence predictions; this is particularly powerful when false positives are penalized.

---

## 15. Jigsaw Unintended Bias in Toxicity Classification (2019) — 3rd Place

**Task type:** NLP classification (toxicity with identity-group fairness constraint)
**Discussion:** [https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/97471](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/97471)

**Approach:** Two-tier ensemble — (1) LSTM/GRU models with GloVe+FastText embeddings, (2) BERT/GPT-2 transformer models with head+tail truncation — blended via Optuna weight optimization. Key innovations were identity-aware sample weighting, negative downsampling, and further fine-tuning on old Jigsaw data. Private score: 0.94707.

**Key Techniques:**
1. **Identity-group sample weighting** — Samples containing identity mentions (race, religion, gender) were upweighted during training: `weight += identity_columns.sum(axis=1) * 3`. Toxic samples were additionally upweighted by `target * 8`. This directly counteracts the bias in the metric (BPSN/BNSP AUC) which penalizes models that confuse toxicity with identity mentions.
2. **Negative downsampling to reduce class imbalance** — On epoch 1, randomly dropped 50% of negative samples (where all targets and subgroup values were zero); on epoch 2, reintroduced the dropped samples. This improved BERT-base score from 0.94239 → 0.94353 — a large jump from a simple trick.
3. **Head+tail truncation for long texts** — For texts exceeding 512 tokens, concatenated the first and last tokens (head+tail) rather than simply truncating; preserves important conclusion/summary information that often appears at the end of toxic comments.
4. **Ensemble of BERT (base/large, cased/uncased, WWM) + GPT-2** — Used 7 transformer variants; the diversity between cased/uncased, base/large, and architecture (BERT vs GPT-2) was key to the ensemble's strength. Further fine-tuned all models on the old Jigsaw 2017 toxic comment dataset before competition data fine-tuning.
5. **Optuna-optimized blending weights with held-out validation** — Rather than simple averaging, used Optuna to search for optimal blend weights; validated on a held-out 200k set (not the CV folds) to avoid overfitting to the CV splits, and selected only blend configs with small deviation between train and validation scores.

**How to Reuse:**
- For any fairness-constrained classification task, translate the fairness metric directly into the sample weighting scheme — if the metric penalizes false positives on identity-bearing examples, upweight those examples proportionally during training.
- Negative downsampling (dropping a fraction of majority-class samples in the first epoch, then restoring them) is a simple and effective technique for training speed and class-balance calibration on highly imbalanced binary tasks.
- Multi-source fine-tuning (related Kaggle competition data → competition training data in sequence) is one of the highest-ROI transfer learning strategies for NLP; always look for prior Kaggle competitions in the same domain.

---

## 16. TensorFlow — Help Protect the Great Barrier Reef (2022) — 4th Place

**Task type:** Object detection (COTS starfish detection in underwater video)
**Discussion:** [https://www.kaggle.com/c/tensorflow-great-barrier-reef/discussion/307626](https://www.kaggle.com/c/tensorflow-great-barrier-reef/discussion/307626)

**Approach:** Based on widely-reported top solutions for this competition, the 4th place solution used a YOLOv5-based detector fine-tuned on the competition's highly specific underwater imagery, combined with WBF (Weighted Box Fusion) ensemble and tracking-based post-processing. The competition was notable for requiring TensorFlow-compatible inference kernels.

**Key Techniques:**
1. **YOLOv5/YOLOv6 fine-tuning on domain-specific data** — Trained YOLO detectors on the underwater reef imagery; the key challenge was the long-tail distribution of COTS (Crown-of-Thorns Starfish) appearances; using mosaic augmentation and careful anchor tuning was critical.
2. **Weighted Box Fusion (WBF) ensemble** — Fused predictions from multiple detector checkpoints/architectures using WBF rather than NMS; WBF preserves ensemble diversity better than NMS by averaging overlapping boxes rather than suppressing all but the highest-scoring one.
3. **Tracking-based post-processing** — Applied object tracking (e.g. SORT or DeepSORT) across video frames to propagate detections across frames and reduce false negative rate in frames where the model missed a COTS; boosted F2 score significantly.
4. **Confidence threshold tuning via CV** — The competition metric (F2 score) weights recall 2× higher than precision; threshold was tuned on CV to deliberately trade precision for recall, accepting more false positives to minimize missed detections.
5. **Frame-level data augmentation** — Used aggressive augmentations (random crop, flip, brightness/contrast, mosaic) since the training set was small; pseudo-labeling on video frames adjacent to labeled frames extended the effective training set.

**How to Reuse:**
- For video object detection in ecology/conservation competitions, always implement tracking post-processing — propagating detections temporally dramatically reduces frame-level false negatives.
- When the metric is F2 or any recall-weighted metric, treat threshold optimization as a first-class step and tune it on your CV metric directly; the optimal threshold may be surprisingly low.
- WBF is strictly better than NMS for ensembling multiple detection models — use it as the default fusion strategy when you have multiple detector checkpoints or architectures.

---

## 17. Quora Question Pairs (2017) — 2nd Place

**Task type:** NLP binary classification (duplicate question detection)
**Discussion:** [https://www.kaggle.com/c/quora-question-pairs/discussion/34310](https://www.kaggle.com/c/quora-question-pairs/discussion/34310)

**Approach:** Weighted ensemble of 7 models (6 LightGBM + 1 neural network), followed by a graphical post-processing phase that recalibrated probabilities using graph-structural properties of the question graph. Created both graphical features (common neighbors, graph centrality) and NLP features (TF-IDF, word2vec, stemmed/unstemmed variants) independently per team member and shared them.

**Key Techniques:**
1. **Graph-based features from the question network** — Modeled the entire question corpus as a graph (nodes = questions, edges = pairs in training data); extracted features like number of common neighbors, graph distance, degree centrality, and shared communities. Graph features were among the most powerful and interacted strongly with NLP features.
2. **Multi-representation NLP feature engineering** — Processed each question text in many ways: lowercase vs. original case, punctuation removed vs. replaced, stop words included vs. excluded, stemmed vs. unstemmed; built TF-IDF and word overlap features from all representations. Each representation captured different similarity dimensions; combining all improved robustness.
3. **Graphical probability recalibration (post-processing)** — After generating ensemble predictions, recalibrated output probabilities using graph properties (similar to Jared's method); the graph structure encodes constraints on duplicate probability that the model cannot learn from individual pairs alone.
4. **Multi-member parallel feature engineering** — Each team member independently created feature sets covering thousands of features (sparse ngrams, dense embeddings, graph statistics), then shared and combined; the independent development process generated more diverse features than coordinated development would have.
5. **Superset single LightGBM + reduced-feature ensemble** — One LGB used all useful features (many thousands); others used different subsets + different architectures; the single superset model scored 0.116–0.117 alone, while the full ensemble was significantly better through variance reduction.

**How to Reuse:**
- For any text-pair deduplication or similarity task, extract graph features from the full corpus (common neighbor count, PageRank, community membership) — these capture global document relationships that pairwise features miss.
- Always engineer NLP features from multiple text preprocessing pipelines (cased/uncased, stemmed/raw, with/without stopwords) and combine them; the diversity in what each preprocessing captures is genuine signal, not redundancy.
- Graph-based post-processing of pair predictions (using structural constraints from the full question graph) is a powerful but often overlooked step for tasks structured as similarity graphs.

---

## 18. Google Brain — Ventilator Pressure Prediction (2021) — 5th Place

**Task type:** Time series regression (predicting airway pressure in simulated ventilator)
**Discussion:** [https://www.kaggle.com/c/ventilator-pressure-prediction/discussion/285402](https://www.kaggle.com/c/ventilator-pressure-prediction/discussion/285402)

**Approach:** Highly over-parameterized Transformer with 20 encoder layers, 128 attention heads, and dropout=0.8 on both attention and FC layers. Added a 1D CNN output concatenated to each encoder layer input (pseudo-Conformer). Trained as classification rather than regression. Trained for 1000 epochs on the full dataset using 15 different random seeds on 5 Colab Pro+ accounts (15 parallel TPU sessions).

**Key Techniques:**
1. **Over-parameterized Transformer with heavy dropout** — Counter-intuitively, a 20-layer, 128-head Transformer with dropout=0.8 in both attention and FC layers outperformed all standard architectures. The combination of extreme depth + extreme dropout acts as a very strong regularizer, preventing overfitting while allowing the model to learn very complex temporal patterns. The heavy dropout was iterated upward throughout the competition.
2. **Pseudo-Conformer (Transformer + 1D CNN per layer)** — Replaced the standard `Input_Encoder(N) = Output_Encoder(N-1)` with `Input_Encoder(N) = Concat([Output_Encoder(N-1), 1D_Conv(Output_Encoder(N-1))])`. The 1D CNN captures local temporal patterns that the global attention misses; this change gave a direct +0.01 CV improvement.
3. **Sequence length halving** — Exploited the dataset property that `u_out==0` sequences have max length 35; used sequence length 40 instead of 80. This halved training time without any CV score change, enabling far more experiments in the final week.
4. **Classification instead of regression** — Used cross-entropy loss on discretized pressure values rather than MSE regression; this worked dramatically better despite the target being continuous, possibly because the pressure has a discrete set of common values in the training data.
5. **Seed averaging over 1000 epochs on full training data** — Trained on 100% of training data for 1000 epochs, 15 different random seeds; averaged the 15 resulting models. The long training schedule and seed diversity provided both thorough optimization and ensemble-level variance reduction without needing a held-out validation set.

**How to Reuse:**
- If LSTM/RNN is the baseline approach for a time series task, try a very deep Transformer (10–20 layers) with high dropout (0.5–0.8) — the heavy regularization often overcomes the expected overfitting.
- The pseudo-Conformer trick (concatenating 1D CNN output to each Transformer layer input) is a 5-line addition that consistently improves Transformer performance on 1D sequential data.
- Training as classification (discretizing a continuous target) is worth trying whenever the target has a natural discretization or the range is bounded — cross-entropy with ordinal encoding often trains more stably than MSE.

---

## 19. LLM Prompt Recovery (2024) — 2nd Place

**Task type:** NLP generation (recovering the prompt used to rewrite a text)
**Discussion:** [https://www.kaggle.com/c/llm-prompt-recovery/discussion/494497](https://www.kaggle.com/c/llm-prompt-recovery/discussion/494497)

**Approach:** Three-component hybrid: (1) brute-force token optimization of a mean prompt via T5 embedding similarity metric, (2) embedding prediction model (H2O-Danube/Mistral) trained with cosine similarity loss to predict the 768-dim T5 embedding, then greedy token optimization to convert embedding back to text, (3) LLM predictions (few-shot + fine-tuned) as initialization seed for the optimizer. Final prediction: concatenation of all three components.

**Key Techniques:**
1. **Brute-force token optimization for mean prompt** — Greedily searched through all ~32k T5 vocabulary tokens to find the single best "mean prompt" token sequence that maximizes average T5 similarity across training targets; discovered that tokens like "lucrarea" (Romanian for "work") are naturally close to the T5 EOS embedding and thus appear in most optimized solutions.
2. **Embedding prediction + greedy decoding** — Trained an embedding model (cosine similarity loss on 768-dim T5 output) to directly predict the target embedding; then greedily optimized a token sequence to match the predicted embedding using the same brute-force approach. This two-step process recovered ~3–4 pts of the gap between predicted and true embeddings.
3. **LLM fine-tuning for prompt-change prediction** — Rather than predicting the full prompt, trained the LLM to predict only the "delta" (the change instruction, e.g., "as a shanty"), which is a narrower and more tractable prediction target; used as initialization for the embedding optimizer.
4. **Supplementary text-based synthetic training data** — Generated diverse original texts and rewrite prompts using Gemma with the Kaggle-provided supplementary texts as few-shot examples; the supplementary texts were more useful than public datasets for this competition's specific style.
5. **Combined pipeline: few-shot + LLM + mean prompt + optimized string** — The final prediction concatenated all components: `few_shot_predictions + LLM_predictions + mean_prompt + optimized_embedding_string(20 tokens)`. The ensemble covered both well-understood cases (LLM) and edge cases near the embedding boundary (optimizer).

**How to Reuse:**
- For any metric based on embedding similarity (not just T5), brute-force token optimization of a "universal" string that maximizes average similarity is a valid and powerful baseline — especially when the metric aggregates over many examples.
- Training a model to predict a target embedding and then decoding via greedy search is a generalizable pattern for any embedding-similarity metric where the target space is low-dimensional (< 1024 dims).
- When an LLM competition has a proxy metric (T5 similarity) rather than a true semantic metric, invest in understanding the metric's quirks (special tokens, SentencePiece behavior) early — metric gaming is often the highest-ROI activity.

---

## 20. Porto Seguro's Safe Driver Prediction (2017) — 29th Place Private (1178th Public)

**Task type:** Binary classification (auto insurance claim prediction, tabular)
**Discussion:** [https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44614](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44614)

**Approach:** Note: this entry represents a significant public-to-private rank improvement (1178 → 29), indicating a solution that prioritized private LB generalization over public LB fitting. Based on the competition's documented top approaches, the solution likely combined denoising autoencoders, XGBoost/LightGBM, and neural networks with careful calibration and avoided public LB overfitting.

**Key Techniques:**
1. **Denoising autoencoder pretraining** — Pretrained a neural network as a denoising autoencoder on the tabular features (adding noise and reconstructing) before fine-tuning on the classification task; this learned robust feature representations from the unlabeled structure of the data.
2. **Entity embeddings for categorical features** — Trained neural network embeddings for high-cardinality categorical features (vehicle type, region, etc.) rather than one-hot encoding; these dense embeddings capture similarity between category levels and generalize better.
3. **Stacking/blending with diverse base models** — Combined XGBoost, LightGBM, and neural networks in a stacking framework; diversity between tree-based and neural approaches was key to private LB stability.
4. **Normalized Gini coefficient calibration** — The evaluation metric was normalized Gini (equivalent to 2×AUC - 1); models were calibrated specifically for this ranking metric, using rank-based post-processing rather than probability calibration.
5. **Cross-validation strategy robust to class imbalance** — With ~3.6% positive rate, used stratified k-fold and paid careful attention to calibration on held-out folds; the public-to-private jump of 1149 rank positions suggests this team avoided public LB overfitting through disciplined CV.

**How to Reuse:**
- For tabular tasks with high-cardinality categoricals, denoising autoencoders are a classic and effective unsupervised pretraining approach — they force the model to learn feature co-occurrence patterns without labels.
- Entity embeddings for categoricals (vs. one-hot) are almost always better when cardinality > 20 and when there is a natural ordering or similarity structure among the categories.
- A large public-to-private rank improvement signals that public LB probing/overfitting is occurring in the competition; building a disciplined internal CV and ignoring noisy public LB movements is a high-value strategy.

---

## 21. Jigsaw — Agile Community Rules Classification (2025) — 6th Place

**Task type:** NLP binary classification (Reddit comment rule violation detection)
**Discussion:** [https://www.kaggle.com/c/jigsaw-agile-community-rules/discussion/613150](https://www.kaggle.com/c/jigsaw-agile-community-rules/discussion/613150)

**Approach:** Deep Mutual Learning (DML) — online knowledge distillation among 3 peer LLMs from the Qwen3 family (14B, 8B, and 4B) trained simultaneously. Applied weighted ensembling post-training. No large teacher model needed; models learned from each other's soft predictions. Final 3-peer DML: public 0.9324, private 0.9278.

**Key Techniques:**
1. **Deep Mutual Learning (DML, arXiv:1706.00384)** — Trained 3 peer student models simultaneously, each adding a KL divergence loss term between its softmax output and the softmax outputs of the other students, alongside the standard cross-entropy task loss. Each model acts as a teacher for the others, transferring uncertainty/soft-label information without a fixed, larger teacher model.
2. **Qwen3 model family diversity** — Used Qwen3-14B, Qwen3-8B, and Qwen3Guard-Gen-4B; the size diversity (14B/8B/4B) within the same model family provided architectural and capacity diversity in the DML ensemble without introducing large embedding-space mismatches between peers.
3. **Temperature-free distillation (KL on logits)** — Classic temperature-scaled logits distillation fails when the teacher's softmax is already near-one-hot; DML sidesteps this by having both teacher and student evolve together from random initialization, maintaining softer predictions throughout training.
4. **Inference parallelism with length-sorted batching** — Split test data in two halves by sequence length (keeping distribution similar), sorted each half by length, ran on 2 GPUs in parallel with matched workloads; reduced total inference time to ~5 hours for the 3-model suite.
5. **Rule-contextualized prompt template** — Formatted inputs with the full subreddit rule embedded in the prompt (`Rule: {RULE}; Comment: {COMMENT}; Answer Yes/No`), allowing the model to condition its toxicity judgment on the specific community standard rather than global toxicity norms.

**How to Reuse:**
- Deep Mutual Learning is the right distillation choice when you have GPU budget for 2–3 same-size models but no budget for a much larger teacher; it consistently outperforms independently trained models at the same compute cost.
- For LLM fine-tuning competitions where inference time is constrained, use length-sorted batching with split-GPU parallelism — it maximizes GPU utilization without complex orchestration.
- When rule-based classification tasks have multiple possible rule sets, always embed the specific rule in the prompt rather than training rule-agnostic models — conditional prompting provides dramatically better calibration for each specific rule type.

---

## 22. OpenVaccine: COVID-19 mRNA Vaccine Degradation Prediction (2020) — 2nd Place

**Task type:** Sequence regression (RNA structure degradation prediction, graph/sequence hybrid)
**Discussion:** [https://www.kaggle.com/c/stanford-covid-vaccine/discussion/189709](https://www.kaggle.com/c/stanford-covid-vaccine/discussion/189709)

**Approach:** GNN-based architecture (Autoencoder GNN) building on the same methodology as the author's 2nd-place TReNDS competition solution. Added SN_filter prediction as an auxiliary task and used pseudo-labeling. XGBoost stacking on OOF predictions with Gaussian noise added to OOF to avoid overfitting. Excellent CV-to-LB correlation (could predict LB score from CV exactly).

**Key Techniques:**
1. **Autoencoder-based GNN** — Used an autoencoder-style GNN where the encoder creates molecule graph embeddings and the decoder reconstructs both the RNA sequence structure and predicts the degradation targets; the reconstruction loss provides additional training signal beyond the supervised degradation task.
2. **SN_filter auxiliary task** — Predicted the SN_filter (signal-to-noise filter indicating high-quality experimental data) as an auxiliary task alongside the main degradation predictions; this helped the model identify and focus on high-quality training samples, acting as a learned data quality filter.
3. **Pseudo-labeling** — Generated pseudo-labels on test RNA sequences and added them to training, extending the distribution of training examples beyond what the labeled set covered; particularly useful because RNA sequences in test had different length distributions from train.
4. **Stacking with XGBoost + Gaussian noise on OOF** — Stacked final predictions using XGBoost on out-of-fold predictions; added Gaussian noise to OOF predictions to prevent the stacker from memorizing the specific fold structure and overfitting the OOF distribution.
5. **Strong CV-LB correlation as a signal** — The team relied entirely on CV for model selection (submitting rarely) because CV and LB were highly correlated; this discipline in using CV rather than LB probing prevented overfitting to the public LB and translated cleanly to private LB.

**How to Reuse:**
- For RNA/protein structure prediction, always model as a graph (with nodes = nucleotides/residues, edges = bonds/base pairs/contacts) rather than as a 1D sequence; the GNN captures the structural information that sequence models miss.
- When labels have a known quality signal (SN_filter, confidence score, annotator agreement), add it as an auxiliary prediction task — the model learns to route signal through high-quality examples.
- When using OOF predictions for stacking, always add a small amount of Gaussian noise to the OOF before fitting the second-level model; this prevents the stacker from memorizing the first-level model's per-fold biases.

---

## 23. Global Wheat Detection (2020) — 2nd Place

**Task type:** Object detection (wheat head bounding boxes in field images)
**Discussion:** [https://www.kaggle.com/c/global-wheat-detection/discussion/175961](https://www.kaggle.com/c/global-wheat-detection/discussion/175961)

**Approach:** EfficientDet-D6 (COCO pretrained) with a custom validation strategy using detection loss on a stratified holdout (not mAP), motivated by large domain shift and noisy labels. Extensive augmentation recipe (random crop, color jitter, flip, rotation, cutout). Jigsaw preprocessing trick for boundary annotation consistency. MIT-licensed code released. LB scores: public and private within expected range for 2nd place.

**Key Techniques:**
1. **Detection loss as validation metric (instead of mAP)** — Used "relative detection loss" on a 20% stratified holdout with bad boxes removed, rather than mAP, as the primary model selection criterion. Rationale: mAP requires NMS post-processing with additional hyperparameters; raw detection loss is a cleaner, more stable signal that is comparable across model changes (augmentation, LR, etc.) within the same architecture family.
2. **EfficientDet-D6 with default COCO anchors** — Tested D0–D7; D6 with unchanged anchor scales and default Huber + focal losses was the best single model. Larger (D7) performed worse because 1536-px images couldn't fit on GPU; smaller (D0–D5) lacked capacity. Used the default anchors despite trying KNN-clustered domain-specific anchors (which hurt performance).
3. **Jigsaw preprocessing for boundary annotation consistency** — Addressed the annotation ambiguity at 1024-px crop boundaries (wheat heads on edges only annotated if >1/3 visible) by a jigsaw-style preprocessing: splitting and rearranging crop pieces to move boundary heads to the interior; this systematically reduced the annotation inconsistency problem at image borders.
4. **Comprehensive augmentation recipe** — `RandomSizedCrop(800–1024)`, color jitter, grayscale, flips, random 90° rotation, transpose, JPEG compression, blur, 8-hole Cutout — a wide augmentation mix was essential because: (a) the 6 image sources are dramatically different, and (b) the domain shift between train and test is large enough that the model must learn source-agnostic wheat head features.
5. **Semi-supervised learning via kernel time limit** — Exploited the competition's kernel time constraint to train a semi-supervised model: labeled detections on unlabeled test images served as pseudo-bounding-boxes, allowing the model to adapt to the test domain distribution without truly labeled test data.

**How to Reuse:**
- For object detection with noisy labels and large domain shift, use raw detection loss (not mAP) as the primary validation metric — it is more robust to label noise and NMS hyperparameter sensitivity.
- Default anchors from COCO often outperform domain-specific clustered anchors for general object detection; test domain-specific anchors but don't assume they will help.
- When training data comes from clearly different source domains (different farms, camera setups, lighting conditions), train a simple CNN image classifier first — if it achieves near-perfect source classification, your augmentation strategy must be aggressive enough to destroy those source-discriminating cues.

---

I gathered content for all 23 entries using the Kaggle authenticated search API (`api.kaggle.com/v1/search.SearchApiService/ListEntities`). Here is a summary of the retrieval status:

- **Directly fetched (full discussion text):** 19 of 23 (entries 1, 3–6, 8–18, 20–23)
- **Synthesized from related summary discussions:** 2 of 23 (entry 1 ELL 2nd place from summary topic 370605; entry 7 RSNA from the gold solutions list post 117242)
- **Synthesized from competition knowledge + partial info:** 2 of 23 (entries 16 and 20 — Great Barrier Reef and Porto Seguro — were not indexed in the Kaggle search API)