# Kaggle Past Solutions — SRK Round 2, Batch 6

Source: kaggle.com/sudalairajkumar/winning-solutions-of-kaggle-competitions
Ingested: 2026-04-17
Note: These are non-1st-place solutions (2nd, 3rd, top-10) with 100+ upvotes. Writeup content sourced from thedrcat/kaggle-winning-solutions-methods dataset; entries marked [DATASET NOT AVAILABLE] rely on curated knowledge from public post-competition discussions.

---

## 1. NFL 1st and Future — Impact Detection (2021) — 2nd Place

**Task type:** Video object detection / temporal event detection
**Discussion:** [https://www.kaggle.com/c/nfl-impact-detection/discussion/208979](https://www.kaggle.com/c/nfl-impact-detection/discussion/208979)
**Note:** The specific 2nd-place writeup (208979) is not in the solutions dataset; content drawn from the adjacent 3rd-place writeup (208787) for the same competition, which describes the same class of approach.

**Approach:** Two-stage pipeline treating the task as "Multi-view Video Event Detection" rather than simple object detection. Stage 1: EfficientDet trained on helmet and impact classes generates candidate bounding boxes. Stage 2: a binary image classifier re-scores each candidate by stacking 9 consecutive grayscale frames (t−4 to t+4) into a single multi-channel input. Post-processing uses multi-view consistency (sideline vs. endzone cameras) to adjust thresholds and NMS-style deduplication across nearby frames.

**Key Techniques:**
1. **EfficientDet two-class detector** — trained on both "helmet" and "impact" classes; only "impact" boxes with score > 0.17 are kept as candidates, using all positive frames plus 50% of negative frames during training.
2. **Temporal stacking for binary classification** — 9 consecutive grayscale crops (h×w×9 tensor) fed to a CNN binary classifier; this boosted single-model local score from 0.3x to ~0.6.
3. **Multi-view post-processing** — when one camera view predicts an impact, the score threshold for the opposite camera within ±1 frame is lowered (0.25 vs. 0.45), exploiting the dual-camera setup.
4. **Frame-range deduplication** — the metric treats predictions within ±4 frames as a single TP; a dropping step removes redundant boxes across this window to avoid splitting one event into multiple predictions.
5. **Ensemble of EfficientDet variants** — yolov5l6 + yolov5x6 with TTA (horizontal flip only) for the detector stage; binary classifiers ensembled across folds.

**How to Reuse:**
- When a competition involves video or sequential images, stacking temporal frames as channels of a standard CNN is a cheap and effective way to incorporate motion context without building a full video model.
- In multi-camera or multi-view settings, cross-view consistency is a free signal for post-processing: if two views agree, lower the threshold; if they disagree, be conservative.
- For event-detection metrics that allow a ±N-frame window, always build NMS/dedup logic that respects that window to avoid wasting TP capacity on repeated detections.

---

## 2. Jigsaw Multilingual Toxic Comment Classification (2020) — 4th Place

**Task type:** Multilingual NLP / binary text classification
**Discussion:** [https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion/160980](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion/160980)

**Approach:** Blend of several XLM-RoBERTa-large models with a 3-step progressive language fine-tuning strategy. Starting from 7 languages, models were fine-tuned to 3 languages, then to a single language (tr, it, es). Because this stepwise fine-tuning distorts global predictions, a per-language shift post-processing step corrected calibration. CV used a 6-fold mix of Group-3fold (per language) and simple 3-fold, giving a reliable proxy for the multilingual public/private LB.

**Key Techniques:**
1. **3-step progressive language fine-tuning** — train on 7 languages → fine-tune on 3 → fine-tune on 1 target language; each step brings the model closer to the monolingual distribution and significantly improved LB.
2. **Dynamic batch padding by sorted length** — inputs padded to the longest sequence in the batch (not a fixed 512); additionally, batches sorted by sequence length so all items in a batch have near-equal length, eliminating most padding entirely. Yielded ~2× training speedup.
3. **Max+mean pooling of hidden states vs. CLS token** — the classification head tested both; both worked, and models with different pooling heads contributed diversity to the ensemble.
4. **Negative downsampling** — dataset is ~5× class-imbalanced; using as many negatives as positives reduces dataset size ~5× and speeds training ~5× with little quality loss.
5. **Per-language prediction shift post-processing** — after the stepwise fine-tuning, global predictions drift; a linear per-language shift (fitted on the small validation set) recalibrates them back to the global scale.

**How to Reuse:**
- In any multilingual competition, progressive fine-tuning from broad (many languages) to narrow (target language) is a strong regularization strategy and consistently beats training on all languages simultaneously.
- Dynamic batch padding sorted by sequence length is universally applicable to any sequence model and costs nothing to implement; use `SmartBatchingDataset` or sort-by-length collate functions.
- Rank-based ensembling (weighted sum of rank percentiles across models) is robust to scale differences between model outputs — especially useful when ensembling models with different pooling heads or architectures.

---

## 3. PLAsTiCC Astronomical Classification (2018) — 4th Place

**Task type:** Time-series multiclass classification (astronomical light curves)
**Discussion:** [https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75011](https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75011)

**Approach:** Blend of LightGBM, neural network, and stacking models. Key innovations were ratio features (inter-passband ratios), Bazin light-curve fitting for parametric features, adversarial-validation-based sample weighting, and a log-ensemble (averaging log-predictions rather than predictions). Stacking on confusion-matrix-style outputs of LGB gave a 0.04 score boost.

**Key Techniques:**
1. **Bazin light-curve fitting** — fits a parametric function to each light curve, capturing rise/fall time, amplitude, and phase offset as compact features. A highly domain-specific but high-impact feature that standard statistical aggregations miss.
2. **Inter-passband ratio features** — rather than using raw per-passband features, dividing each passband value by the sum across all passbands; creates scale-invariant features that capture the spectral shape of the object.
3. **Sample weighting via adversarial validation** — train/test sets are drawn from very different distributions. Using `hostgal_photoz` and `ddf` for sample weights (fit via adversarial validation) improved score ~0.02.
4. **Stacking on confusion matrix predictions** — train a Logistic Regression on top of the LGB model's class probability outputs; treats model predictions as meta-features. Gave +0.04 improvement.
5. **Log-ensemble** — instead of averaging predictions P, average log(P) and exponentiate; more appropriate when the metric is log-loss-based and predictions are on a log scale.

**How to Reuse:**
- For time-series classification, always evaluate whether domain-specific parametric curve-fitting (like Bazin for light curves, or similar for finance/IoT) produces richer features than generic statistical aggregations.
- Log-ensemble is a simple one-line change that can improve ensembles when the task is log-loss optimization; test it whenever blending probability outputs.
- Stacking on the model's own probability outputs (as a confusion matrix) is a cheap and effective meta-learning step; a simple LogisticRegression or Ridge on OOF probabilities often beats handcrafted blending weights.

---

## 4. Happywhale — Whale and Dolphin Identification (2022) — Top-3 (3rd Place area)

**Task type:** Fine-grained image retrieval / metric learning (whale/dolphin individual ID)
**Discussion:** [https://www.kaggle.com/c/happy-whale-and-dolphin/discussion/304504](https://www.kaggle.com/c/happy-whale-and-dolphin/discussion/304504)
**Note:** The specific discussion (304504) is listed in the SRK batch but not in the solutions dataset; the adjacent 3rd-place writeup (319789) from the same competition describes the overall top-3 approach.

**Approach:** Two-branch metric learning system: one branch trained on the full body image, another trained only on the backfin (detected via YOLOv5). EfficientNet-B7 backbones with ArcFace and CurricularFace loss functions. At inference, embeddings from both branches (8 × 512 = 4096 dimensions) are concatenated, normalized per-segment, and the nearest neighbours determine the predicted identity. Pseudo-labeling on the test set and Bayesian optimization for embedding weights were key final touches.

**Key Techniques:**
1. **Part-based detection + re-ID ensemble** — YOLOv5 detects the backfin; a separate model trained only on backfin crops specializes in cases where body-level features are ambiguous. Combining body and part embeddings improves recall for individuals with only partial visibility.
2. **ArcFace + CurricularFace loss** — two metric learning losses providing different training dynamics; CurricularFace focuses harder examples later in training. Using both in separate models adds diversity for ensembling.
3. **Embedding concatenation for retrieval** — instead of averaging part and body embeddings, concatenate them (4096-dim) so the nearest-neighbor search can weight both spaces. Normalize within each 512-dim block before concatenating to prevent scale dominance.
4. **Bayesian optimization for ensemble weights** — optimize embedding weights (how much to weight each 512-dim block) with Optuna/Bayesian search; avoids manual tuning and transfers to private LB.
5. **New-ID replacement post-processing** — explicitly model the "new individual not in training" case; based on confidence thresholds, replace low-confidence identities with `new_individual` to improve MAP@5.

**How to Reuse:**
- In any individual-ID / re-identification competition, train separate models on full body and discriminative parts (face, fin, plate); concatenated embeddings consistently outperform single-view models.
- ArcFace is the baseline metric learning loss for open-set recognition; CurricularFace and other curriculum-based variants add value when combined with ArcFace in an ensemble.
- For retrieval-style competitions, the `new_individual` (or equivalent OOD) prediction deserves its own threshold tuning — it's often worth more MAP@5 points than marginal improvements to known-class accuracy.

---

## 5. Cornell Birdcall Identification (2020) — 2nd Place

**Task type:** Audio classification (bird species from field recordings)
**Discussion:** [https://www.kaggle.com/c/birdsong-recognition/discussion/183269](https://www.kaggle.com/c/birdsong-recognition/discussion/183269)

**Approach:** Convert audio to mel spectrograms (saved to disk for speed), then treat as image classification. Heavy domain-specific augmentation: mixing 1–3 files, adding non-bird ambient sounds, adjusting upper frequency response, power-scaling spectrograms for contrast. Final ensemble of 6 CNN models (EfficientNet-B0, ResNet50, DenseNet121) with predictions squared before averaging. Manually cleaned 20,000 training files to remove long silent segments.

**Key Techniques:**
1. **Manual data cleaning** — went through 20,000 training audio files and removed long segments with no target bird. Cleaner training data proved more valuable than model complexity.
2. **Domain-specific spectrogram augmentation** — power-scaling the spectrogram image (raising to 0.5–3.0) changes effective contrast; value < 1 brings noise closer to signal, value > 1 suppresses quiet sounds. Combined with adding realistic ambient sounds (rain, conversation, traffic) forces the model to learn bird-specific features rather than acoustic environment.
3. **Upper-frequency roll-off augmentation** — with probability 0.5, attenuates high frequencies to simulate the physical effect of distance (high frequencies fade faster). Bridges train/test distribution gap where test birds may be farther away.
4. **Soft label for background species** — foreground bird label = 1.0, background species (present but not target) = 0.3 using BCEWithLogitsLoss. Reduces overconfident false negatives on co-occurring species.
5. **Power-mean ensemble** — predictions squared, averaged, then square-rooted (geometric mean in probability space); slightly outperforms arithmetic averaging and is more conservative on uncertain predictions.

**How to Reuse:**
- For any bioacoustics / audio competition, invest heavily in data cleaning before modeling — removing non-target segments beats most model improvements.
- Domain-specific augmentations that simulate physical phenomena (distance attenuation, ambient noise) are among the most reliable augmentations for audio-to-image pipelines.
- Soft labels for co-occurring but non-target classes are a general technique for any multi-label audio or image problem where "background" categories are annotated imprecisely.

---

## 6. RSNA 2023 Abdominal Trauma Detection (2023) — 2nd Place

**Task type:** Medical image segmentation + multi-label classification (CT scans)
**Discussion:** [https://www.kaggle.com/c/rsna-2023-abdominal-trauma-detection/discussion/447453](https://www.kaggle.com/c/rsna-2023-abdominal-trauma-detection/discussion/447453)
**Note:** This competition is not in the thedrcat solutions dataset (post-dates it). Summary from curated post-competition knowledge.

**Approach:** Multi-stage pipeline: (1) 3D segmentation of abdominal organs (liver, spleen, kidneys, bowel) using SegFormer or nnU-Net on axial CT slices, then (2) per-organ 2D CNN classifiers using the cropped organ ROI as input. The organ-level predictions feed into a patient-level aggregation model. Ensembled across multiple CNN backbones (EfficientNet, ConvNeXt) at different input resolutions.

**Key Techniques:**
1. **Organ segmentation → ROI cropping** — segment organs first, then crop tightly to each organ for the downstream injury classifier; eliminates irrelevant anatomy and dramatically reduces the classification problem's spatial complexity.
2. **3D-to-2D slice projection** — because CT volumes are 3D but labeled at the patient level, extract 2D axial slices from segmented organ volumes and classify each slice independently; aggregate slice predictions (max, mean) to patient level.
3. **Multi-label loss per organ** — treat each organ (liver, spleen, left kidney, right kidney, bowel) as a separate binary classification head with its own loss; enables the model to specialize per organ rather than learn one joint representation.
4. **Patient-level aggregation with sequence models** — use LSTM or attention over the sequence of per-slice predictions to capture spatial context before producing the final patient-level label.
5. **Pseudo-labeling on unlabeled scans** — use a confident ensemble to generate soft labels on the larger unlabeled test set, then retrain to improve generalization across scanner types and patient demographics.

**How to Reuse:**
- The segment-then-classify pipeline is the dominant approach in any structured medical imaging task — always prototype organ/lesion detection before committing to end-to-end models.
- For 3D volumetric data with 2D labels, the slice-level approach with max-pooling aggregation is a reliable baseline that avoids the memory costs of full 3D models.
- Multi-head organ-specific classifiers nearly always outperform a single joint classifier in radiology tasks — specialization matters when pathology manifestations differ across anatomical structures.

---

## 7. CommonLit Readability Prize (2021) — 4th Place

**Task type:** NLP regression (text readability scoring)
**Discussion:** [https://www.kaggle.com/c/commonlitreadabilityprize/discussion/258148](https://www.kaggle.com/c/commonlitreadabilityprize/discussion/258148)

**Approach:** Solo gold medal using DeBERTa-large as the primary model, with a custom AttentionBlock head applied to the sequence's last hidden state. Multi-seed CV (5 folds × 5 seeds = 25 runs) for reliable CV estimation on a small, noisy dataset. Stochastic Weight Averaging (SWA) applied after epoch 3. Final ensemble via RidgeCV, BayesianRidgeRegression, and the Netflix method (from 2009 BigChaos prize).

**Key Techniques:**
1. **Multi-seed averaging for small noisy datasets** — run 5 seeds per fold configuration; average the 25 resulting model predictions. On small (~2,800 samples) datasets with noisy targets, this dramatically stabilizes CV estimates and reduces luck dependence.
2. **AttentionBlock on sequence last hidden state** — instead of pooling or using only the CLS token, apply a learned attention mechanism over all token hidden states; allows the model to weight which tokens matter most for readability regression.
3. **Stochastic Weight Averaging (SWA)** — start SWA after epoch 3 of 6; averages checkpoints in weight space rather than prediction space, improving generalization on small datasets without additional data.
4. **Textstat readability features concatenated to transformer output** — `flesch_reading_ease` and `smog_index` appended to the AttentionBlock output before the regression head; tiny but repeatable CV improvement.
5. **Netflix ensemble method** — the 2009 BigChaos prize blending approach (linear combination with LOO cross-validation for weight estimation); consistently outperformed simple averaging and was more stable than greedy model addition.

**How to Reuse:**
- For small-dataset NLP tasks, multi-seed averaging (not just multi-fold) is the most reliable way to reduce variance — budget compute for at least 3–5 seeds per experiment.
- SWA is a free improvement for any fine-tuning task on small datasets; requires only a scheduler change after warm-up and costs no additional forward passes.
- When ensembling regressors, fit ensemble weights with RidgeCV or BayesianRidge on OOF predictions — these shrink coefficients appropriately and prevent overfitting the ensemble to a small validation set.

---

## 8. LLM Prompt Recovery (2024) — Top area (subjective competition learnings post)

**Task type:** Text generation / LLM prompt inference (recover the prompt used to transform text)
**Discussion:** [https://www.kaggle.com/c/llm-prompt-recovery/discussion/483916](https://www.kaggle.com/c/llm-prompt-recovery/discussion/483916)
**Note:** The title describes this as a "learnings" post rather than a strict placement writeup. The competition is not in the thedrcat dataset. Summary from curated post-competition knowledge.

**Approach:** The competition metric (sharpened cosine similarity of Gemma embeddings) had unusual properties that dominated strategy. Top teams discovered that a single universal prompt ("What is the essence of the text?") scored extremely highly due to high cosine similarity in embedding space, making the metric nearly impossible to optimize beyond this baseline. Real signal came from fine-tuned Mistral / LLaMA models predicting the instruction category, followed by retrieval from a prompt dictionary.

**Key Techniques:**
1. **Metric gaming awareness** — the sharpened cosine similarity metric heavily rewarded semantically generic prompts; understanding the metric's geometry (centroid of prompt embedding space) before building models is critical in LLM-output evaluation competitions.
2. **Embedding-space centroid analysis** — computing the mean embedding of a large set of prompt candidates reveals the "universal baseline"; any model output closer to this centroid than specific predictions scores well.
3. **Fine-tuned LLM for prompt category classification** — fine-tune Mistral-7B or similar on (original text, transformed text) → prompt category pairs; classification into prompt types (summarize, rewrite, make formal, etc.) outperforms open-ended generation.
4. **Retrieval-augmented prompt prediction** — maintain a dictionary of seen prompts; retrieve the most embedding-similar prompt from the dictionary rather than generating free-form — more stable and less prone to hallucination.
5. **Sharpening exponent tuning** — the metric uses a sharpening exponent on cosine similarity; tuning the power applied to model output probabilities to match the metric's sharpening was a key calibration step.

**How to Reuse:**
- In any competition using embedding-based similarity as a metric, first compute the centroid of the training label embedding space — if one generic output scores near-optimally, the metric is weak and strategy shifts to being "less wrong" than random variation.
- For prompt-recovery or instruction-inference tasks, frame it as classification (which instruction category) rather than generation; classification over a fixed taxonomy generalizes better.
- Always analyze the metric geometry before modeling; in LLM competitions especially, the metric can be dominated by simple baselines that no ML model can beat.

---

## 9. LMSYS — Chatbot Arena Human Preference Predictions (2024) — 2nd Place

**Task type:** NLP / LLM output preference prediction (pairwise ranking)
**Discussion:** [https://www.kaggle.com/c/lmsys-chatbot-arena/discussion/527685](https://www.kaggle.com/c/lmsys-chatbot-arena/discussion/527685)
**Note:** Not in the thedrcat solutions dataset. Summary from curated post-competition knowledge.

**Approach:** Fine-tuned Gemma-7B and LLaMA-3-8B on the pairwise preference task (given prompt + response A + response B, predict which is preferred). Long context handling via chunking and partial truncation. The winning insight was that larger models fine-tuned on the full conversation context significantly outperformed smaller models or feature-based approaches.

**Key Techniques:**
1. **Full-conversation fine-tuning of 7B+ LLMs** — passing the complete (prompt + response A + response B) string to Gemma-7B or LLaMA-3-8B with standard cross-entropy fine-tuning on the preference label; larger models had substantially better calibration and accuracy than smaller ones.
2. **Truncation strategy for long conversations** — when conversations exceed context window, truncate from the middle (preserve prompt start and response end) rather than the end; preserves the most discriminative parts of each response.
3. **LoRA / QLoRA for efficient fine-tuning** — use rank-16 to rank-64 LoRA adapters; enables fine-tuning 7B models on single GPUs with negligible quality loss vs. full fine-tuning.
4. **Ensemble of multiple LLM backbones** — blend Gemma-7B, LLaMA-3-8B, and Mistral-7B fine-tuned checkpoints; diversity across base models improves calibration on edge cases.
5. **Length and formatting features** — append simple metadata (response length ratio, markdown formatting presence) as auxiliary features to the LLM's classification head; marginal but consistent improvement.

**How to Reuse:**
- For any pairwise preference prediction task, fine-tuning a 7B LLM end-to-end on (A, B, preference) triples is the baseline that beats all feature engineering; prioritize larger models over more features.
- Truncation from the middle (keep start + end, drop middle) is broadly applicable to any long-context NLP task when you cannot increase context window size.
- LoRA fine-tuning at rank 16–32 is sufficient for most classification tasks on top of 7B models; use QLoRA (4-bit) if GPU memory is constrained.

---

## 10. ASHRAE — Great Energy Predictor III (2019) — 3rd Place (Public 1st → Private 6th)

**Task type:** Time-series regression (building energy consumption forecasting)
**Discussion:** [https://www.kaggle.com/c/ashrae-energy-prediction/discussion/122796](https://www.kaggle.com/c/ashrae-energy-prediction/discussion/122796)
**Note:** The target discussion (122796, title "497th place shakedown: PUBLIC LB 1st solution") describes the approach that was PUBLIC LB 1st but finished 6th privately. The adjacent 3rd-place writeup (124984) from the same competition is used here.

**Approach:** Blended over 30 diverse experiments using simple average after filtering by Pearson correlation to public LB scores. Key insight: a script that eliminated simultaneous zeros across all meters at the same site/period (likely meter outages) was the main differentiating preprocessing step. Models included Keras CNN (Karpathy/aerdem4 style), LightGBM, and CatBoost trained on meter-level subsets. No fancy CV; relied on public LB for model selection.

**Key Techniques:**
1. **Simultaneous-zero elimination preprocessing** — scan for periods where all meters at a site report zero simultaneously; these are meter outages, not true zero consumption. Removing them from training significantly reduced noise. This single preprocessing step gave a consistent LB lift.
2. **Meter-level model splitting** — train separate decision tree models per meter type (electricity, chilled water, steam, hot water) rather than one global model; each meter has fundamentally different seasonal patterns and correlates with different weather variables.
3. **Solar horizontal radiation feature** — derived from site latitude, day-of-year, and hour-of-day using a simplified solar geometry formula; particularly useful for electricity metered buildings where solar gain drives HVAC load.
4. **Temperature lag features** — lagged versions of outdoor temperature (previous hours/days) as features; thermal mass means current energy use reflects past temperature more than present.
5. **Blend 30+ experiments by Pearson correlation filter** — rather than selecting the "best" single model, average all models whose predictions correlate with the public LB above a threshold; diversity-preserving ensemble that avoids overfitting to any single CV split.

**How to Reuse:**
- In energy/utilities forecasting, always check for systematic zero periods that correspond to outages or meter failures — they are a major source of noise that CV cannot detect.
- Splitting a global model into per-segment models (per meter type, per building category, per geography) is a reliable pattern for structured tabular time-series data with heterogeneous segments.
- When CV and public LB disagree, blending all plausible experiments weighted by their public LB Pearson correlation is a safer strategy than chasing either metric alone.

---

## 11. CZII — CryoET Object Identification (2024–2025) — 1st Place (labeled as segmentation + ensemble)

**Task type:** 3D biological image segmentation / object detection (cryo-electron tomography)
**Discussion:** [https://www.kaggle.com/c/czii-cryo-et-object-identification/discussion/561510](https://www.kaggle.com/c/czii-cryo-et-object-identification/discussion/561510)
**Note:** Not in the thedrcat solutions dataset. Summary from curated post-competition knowledge. The CSV lists rank_in_comp=2 and title says "1st place solution" — may be a label mismatch in the SRK CSV.

**Approach:** Hybrid segmentation pipeline: partial U-Net for semantic segmentation of particle types on 3D tomogram volumes, followed by instance identification via connected-component analysis or 3D NMS. Ensembled predictions across multiple U-Net variants trained with different augmentation strategies (flips, rotations, noise). Post-processing accounted for the particle size priors for each class.

**Key Techniques:**
1. **Partial U-Net architecture** — a U-Net variant with fewer skip connections or partial channel attention; reduces memory footprint for 3D volumetric inputs while retaining segmentation quality for the small, dense particle classes in cryo-ET data.
2. **3D patch-based training** — train on sub-volumes (patches) of the full tomogram rather than the full volume; enables larger batch sizes and better data augmentation for the rare particle classes.
3. **Class-specific post-processing thresholds** — each particle class (ribosome, fatty acid synthase, etc.) has known size ranges; apply class-specific minimum-volume and confidence thresholds during connected-component extraction to reduce false positives.
4. **Multi-scale ensemble** — train U-Nets at different voxel resolutions; merge predictions from coarse (for large particles) and fine (for small particles) scales at inference.
5. **Augmentation with physical constraints** — random rotations respecting the anisotropic resolution of cryo-ET data (z-axis typically lower resolution); noise injection matching the contrast transfer function artifacts common in cryo-ET.

**How to Reuse:**
- For 3D biomedical segmentation, always train on patches rather than full volumes; even with modern GPUs, full-volume training at reasonable resolution is impractical for most cryo-ET or CT datasets.
- Use known domain priors (particle size, shape) as hard constraints in post-processing; this is faster and more reliable than learning these constraints end-to-end.
- Multi-scale ensemble (train at 2–3 voxel resolutions, merge) is a standard and reliable boost for 3D segmentation tasks.

---

## 12. American Express — Default Prediction (2022) — 11th Place

**Task type:** Tabular classification (credit default prediction from time-series customer data)
**Discussion:** [https://www.kaggle.com/c/amex-default-prediction/discussion/347786](https://www.kaggle.com/c/amex-default-prediction/discussion/347786)

**Approach:** LightGBM with DART boosting on a rich set of aggregated time-series features. The most impactful innovation was "meta features" — OOF predictions from a model trained on the raw per-statement data, then aggregated by time period and used as features for the final model. Adversarial validation + null importance were used for feature selection (4,300 → 1,300 features). Ensemble of 3 LightGBM models with different feature subsets via weighted rank averaging.

**Key Techniques:**
1. **Meta features from OOF predictions** — train a model on raw time-series rows (before aggregation by customer), generate OOF predictions, then aggregate those predictions by time period (last 3 months, last 6 months) and use as features for the final model. Lifted single-model score from 0.799 to 0.800 public LB.
2. **Time-period-aware aggregations** — instead of aggregating all statements for a customer, compute separate min/max/mean/std for the last 3 months and last 6 months; captures behavioral changes over time that static aggregations miss.
3. **Rate and diff features** — `last_value − last_3month_mean`, `last_3month_mean / last_6month_mean`; simple ratio/difference features between time-period aggregates capture trend and momentum signals.
4. **Adversarial feature selection** — train a classifier to distinguish train from test; features with high importance in this adversarial classifier (R_1, D59, S_11, B_29) are leaking distributional information and should be dropped from the main model.
5. **Null importance feature selection** — compare feature importance with real target vs. shuffled target; keep only features whose real importance exceeds mean shuffled importance. Reduces 4,300 → 1,300 features with negligible quality loss.

**How to Reuse:**
- Meta features (OOF predictions from a simpler model, aggregated over time) are a highly transferable technique for any time-series aggregation task — especially when raw temporal data is available but final labels are at a coarser level.
- Always run null importance feature selection before hyperparameter tuning; it removes irrelevant noise features cheaply and speeds up every subsequent experiment.
- Adversarial validation for feature selection (drop features that help distinguish train from test) is distinct from adversarial validation for sample weighting — both are worth doing, independently.

---

## 13. Home Credit Default Risk (2018) — 2nd Place (Team: ikiri_DS, 12 members)

**Task type:** Tabular classification (loan default risk)
**Discussion:** [https://www.kaggle.com/c/home-credit-default-risk/discussion/64596](https://www.kaggle.com/c/home-credit-default-risk/discussion/64596)
**Note:** Target discussion 64596 not in dataset; using adjacent 2nd-place writeup 64722 for the same competition.

**Approach:** Large 12-person team with heavy division of labor across feature engineering, model diversity, and blending. The feature pipeline included ~1TB of candidate features with brute-force search, genetic programming, PCA/UMAP/t-SNE/LDA dimension reductions, and domain-specific interest rate features. Models ranged from LightGBM/CatBoost to CNNs, RNNs, neural networks, and a denoising autoencoder (DAE). Final blend used direct AUC maximization via modified Powell algorithm for 2nd-level weighting.

**Key Techniques:**
1. **Brute-force feature search from ~1TB feature pool** — generate an enormous number of candidate features (aggregations, interactions, ratios across all tables), then select via importance-based filtering; scale compensates for lack of domain knowledge.
2. **Dimension reduction as features** — PCA, UMAP, t-SNE, and LDA embeddings of the original feature set used as additional features for tree models; captures non-linear structure that tree models might miss in raw feature space.
3. **Denoising autoencoder (DAE) for tabular data** — based on porto seguro 1st place approach; pre-train a DAE on all available data (train+test) to learn robust embeddings, then use these as features. Particularly effective for high-cardinality, noisy tabular data.
4. **Interest rate feature engineering** — reverse-engineer implied interest rates from loan amount, repayment schedule, and duration in the bureau table; a domain-specific feature with high predictive signal for creditworthiness.
5. **Direct AUC maximization for blending** — use modified Powell optimization algorithm to find blend weights that directly maximize validation AUC rather than fitting a linear model; more appropriate when the metric is AUC (non-differentiable).

**How to Reuse:**
- For credit/tabular competitions with multiple related tables, always derive domain-specific financial ratios (implied interest rates, utilization rates, delinquency patterns) — these outperform generic aggregations because they encode economic meaning.
- DAE pre-training on train+test combined is a consistently strong technique for tabular data; treat it as a learned embedding layer analogous to pre-trained word embeddings in NLP.
- When ensembling at the final stage, optimize blend weights directly for the competition metric (AUC, F1, etc.) using Nelder-Mead or Powell rather than fitting a linear model on OOF predictions.

---

## 14. OTTO — Multi-Objective Recommender System (2023) — 5th Place

**Task type:** E-commerce session-based recommendation (clicks, carts, orders)
**Discussion:** [https://www.kaggle.com/c/otto-recommender-system/discussion/382802](https://www.kaggle.com/c/otto-recommender-system/discussion/382802)

**Approach:** Two-stage recommender: (1) candidate generation via co-visitation matrices implemented in Numba for speed, (2) LightGBM ranking model on rich session-level features. Inference accelerated with Treelite. Key insight: normalizing co-visitation weights by item frequency (like TF-IDF for items) dramatically improved candidate quality. Single model performed nearly identically to ensemble on private LB.

**Key Techniques:**
1. **Co-visitation matrix with action-type decomposition** — instead of one global co-visit matrix, build separate matrices for (any→any), (click→cart), (cart→order), (order→order), etc.; each relationship type captures different purchase-funnel dynamics.
2. **Frequency-normalized co-visit weights** — weight of (item A, item B) = count(A,B) / frequency(A); analogous to TF-IDF where common items are down-weighted. This shifts focus to statistically surprising co-occurrences rather than popularity effects.
3. **Recency weighting in candidates** — weight item pairs by their distance in the session (fewer steps between = stronger signal); `weight(aid1, aidk) = (k−1) / frequency(aid1)` gives more weight to immediate transitions.
4. **Optuna hyperparameter optimization for candidate retrieval** — tune the weights of each co-visit matrix, recency discount, and frequency normalization jointly using Optuna on a validation holdout; avoids manual grid search across dozens of retrieval parameters.
5. **Treelite for inference speed** — compile the final LightGBM model to Treelite for 3–5× faster CPU inference; essential for ranking 80–120 candidates per session across millions of sessions within the Kaggle notebook time limit.

**How to Reuse:**
- For any session-based recommendation task, co-visitation matrices split by action type (view, add-to-cart, purchase) consistently outperform single global matrices; action type encodes purchase intent.
- Frequency normalization of co-visit counts is the recommender equivalent of TF-IDF — apply it by default to avoid popularity bias dominating the candidate list.
- Use Treelite (or ONNX Runtime for neural models) for any competition where inference time is constrained; 3–5× speedup is typical and enables larger candidate sets.

---

## 15. Mechanisms of Action (MoA) Prediction (2020) — 5th Place

**Task type:** Tabular multi-label classification (drug mechanism of action from gene expression)
**Discussion:** [https://www.kaggle.com/c/lish-moa/discussion/200533](https://www.kaggle.com/c/lish-moa/discussion/200533)

**Approach:** Discriminate between "seen" drugs (in training) and "unseen" drugs (not in training) using metric learning (L2 Softmax → cosine similarity), then blend models trained with two different CV strategies — MultilabelStratifiedKFold (better for seen drugs) and MultilabelStratifiedGroupKFold (better for unseen) — weighted by the similarity score. Features: raw gene/cell, PCA, RankGauss normalization, PolynomialFeatures, and simple statistics.

**Key Techniques:**
1. **Seen/unseen drug discrimination via metric learning** — train an L2-normalized softmax classifier to produce embeddings; use cosine similarity between test sample and training drug embeddings to estimate how "familiar" each test sample is. Weight ensemble blending by this similarity.
2. **RankGauss normalization** — transform each feature column by rank, then map to a Gaussian distribution via inverse CDF; makes features more Gaussian, which benefits neural networks. Widely adopted in tabular competitions after this competition.
3. **MultilabelStratifiedGroupKFold** — CV that groups all observations for the same drug (drug_id) into the same fold; prevents leakage from the same drug appearing in train and validation when the test set contains unseen drugs.
4. **Dual CV strategy blending** — train one set of models on standard multilabel-stratified folds (good for seen drugs), another on group-stratified folds (good for unseen); blend based on metric learning similarity score at inference.
5. **Drug ID reconstruction via clustering** — when official drug IDs were not yet released, reconstructed them heuristically using clustering + manual adjustment; enabled reliable GroupKFold CV weeks before the official IDs were published.

**How to Reuse:**
- RankGauss is a universal tabular feature transformation for neural networks — apply it by default before training any NN on tabular data with heterogeneous feature distributions.
- Whenever test set has OOD samples (unseen groups, new categories), build a meta-model that detects OOD-ness and routes predictions to appropriate specialized models rather than using one global model.
- MultilabelStratifiedGroupKFold is the correct CV for any multi-label task where entities (drugs, patients, users) appear multiple times — use it to prevent group leakage.

---

## 16. RSNA Intracranial Hemorrhage Detection (2019) — 2nd Place (3rd Place area)

**Task type:** Medical image classification (CT scan multi-label hemorrhage type detection)
**Discussion:** [https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/discussion/117223](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/discussion/117223)
**Note:** Target discussion 117223 not in dataset; using the adjacent 2nd-place writeup 117228 from the same competition.

**Approach:** Single image classifier (ResNeXt101) trained on 5-fold split, then embeddings extracted from the pre-logit (GAP) layer and fed into an LSTM that models the sequence of CT slices for each patient. Trained 15 LSTMs (3 folds × 5 epochs, taking a separate LSTM checkpoint per epoch). Delta embeddings (current − previous, current − next) concatenated to the LSTM input to provide the model with explicit slice-to-slice change signals.

**Key Techniques:**
1. **CNN-LSTM two-stage architecture** — a CNN classifies individual slices; the CNN's penultimate (GAP) layer embeddings are extracted and fed sequentially to an LSTM, which models the volumetric context across the scan. This avoided the memory cost of full 3D models while capturing sequence-level information.
2. **Delta embeddings for LSTM input** — concatenate (current embedding), (current − previous embedding), and (current − next embedding) as the LSTM's input at each step; explicitly provides the model with the "difference" signal between adjacent slices, which is informative for detecting lesion boundaries.
3. **DICOM windowing preprocessing** — apply multi-window normalization (brain, blood, soft-tissue windows) to raw DICOM Hounsfield values before converting to images; produces multi-channel inputs that highlight different tissue densities.
4. **Per-epoch LSTM training on fixed CNN embeddings** — extract CNN embeddings once per fold, then train 5 separate LSTMs (one per CNN epoch); average all 15 LSTM predictions at inference. This is a cheap form of ensemble that exploits the diversity of CNN checkpoints from different training stages.
5. **Sequence padding with dummy embeddings** — pad shorter sequences (fewer slices) to a fixed length using a dummy zero embedding; the LSTM loss is masked to ignore the padded positions, avoiding gradient corruption from padding.

**How to Reuse:**
- The CNN-LSTM two-stage pattern is directly transferable to any sequential medical imaging task (time-series MRI, video endoscopy, longitudinal scans) — extract slice embeddings with a 2D CNN, model sequence with LSTM.
- Delta embeddings (difference between adjacent frames/slices) are a general-purpose technique for any sequential model where change detection is more informative than absolute values.
- Per-epoch ensemble (take checkpoints at multiple training epochs, average their predictions) is a free ensemble that requires zero additional training; always save checkpoints and average the last 3–5 epochs.

---

## 17. APTOS 2019 Blindness Detection (2019) — 7th Place

**Task type:** Medical image classification / regression (diabetic retinopathy severity grading)
**Discussion:** [https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108058](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108058)

**Approach:** Ordinal regression formulation with a 4-element sigmoid output summed to produce a score in [0, 4]. Models: SeResNeXt50, SeResNeXt101, InceptionV4. Pre-trained on 2015 Kaggle DR data + Idrid + Messidor datasets, then fine-tuned, then pseudo-labeled on test set + Messidor 2. Heavy Albumentations augmentation pipeline. Loss function progression: soft cross-entropy → focal+kappa → ordinal Cauchy loss.

**Key Techniques:**
1. **Ordinal regression as summation of sigmoid outputs** — predict a 4-element tensor with sigmoid activations, sum to get the final score in [0, 4]; avoids thresholding post-processing and allows MSE-like losses while preserving ordinal structure. More robust than standard classification for ordered categories.
2. **Multi-dataset pre-training sequence** — pre-train on 2015 DR dataset → validate on current competition + Idrid → train on competition + Idrid + Messidor → pseudo-label Messidor 2 → fine-tune. Each step uses a progressively smaller but more relevant dataset, reducing distribution shift.
3. **Cauchy loss for ordinal regression** — a robust loss function (similar to Huber but with heavier tails) less sensitive to outlier gradings; outperformed MSE and Huber (smooth L1) in final experiments.
4. **Heavy Albumentations augmentation** — ShiftScaleRotate, OpticalDistortion, RandomSizedCrop; also custom augmentations for retina images (zero-top-and-bottom to remove black borders). Augmentation was the primary regularization strategy with no dropout.
5. **Global Average Pooling with simpler heads** — tested Coord-Conv + FPN, Coord-Conv + LSTM pooling heads; all performed equally to or worse than standard GAP. Simpler is better for medical image grading — invest time in loss function and data rather than head architecture.

**How to Reuse:**
- For any ordinal label task (severity grading, quality scores), the summation-of-sigmoids formulation naturally enforces ordinality without post-hoc threshold tuning — use it as a default before trying classification.
- Multi-dataset pre-training (related public datasets → competition data → pseudo-labeled test) is a reliable recipe for small medical imaging competitions; the order matters — start broad, finish specific.
- Cauchy loss (or its equivalents: pseudo-Huber, Welsch loss) is worth testing for any regression with noisy labels; it down-weights large residuals more aggressively than L2, which is appropriate when annotation quality is variable.

---

## 18. Predict Student Performance from Game Play (2023) — 7th Place

**Task type:** Tabular / time-series classification (predict 18 binary question outcomes from gameplay logs)
**Discussion:** [https://www.kaggle.com/c/predict-student-performance-from-game-play/discussion/420119](https://www.kaggle.com/c/predict-student-performance-from-game-play/discussion/420119)
**Note:** Not in thedrcat solutions dataset. Summary from curated post-competition knowledge. The title notes "Efficiency 1st" — indicating a focus on feature engineering over complex models.

**Approach:** Feature-heavy LightGBM approach with minimal compute ("efficiency first"). Rather than sequence models, engineered hundreds of aggregation features from the gameplay log (click counts, time spent, error rates per game level) per student per question-group. Separate models per question cluster. The key was creating question-specific features that captured the relevant gameplay behaviors for each of the 18 binary questions.

**Key Techniques:**
1. **Question-group-specific feature engineering** — each of the 18 questions is answered after a specific set of levels; engineer features only from the gameplay logs relevant to those levels, rather than global features across all levels. This dramatically reduces noise.
2. **Per-question binary model** — train 18 separate LightGBM models (or 18 output heads), one per question; rather than one joint model that must simultaneously predict all questions with potentially conflicting feature importances.
3. **Aggregate statistics from event sequences** — from raw clickstream events, compute: total time on level, number of incorrect attempts, ratio of correct-to-total actions, unique room visits. These simple aggregations proved more robust than sequence models for this competition size.
4. **Level-transition features** — features capturing whether the student revisited earlier levels (backtracking), skipped levels, or followed the intended linear path; correlates with comprehension and predicts correctness on later questions.
5. **Calibration of binary probabilities** — apply isotonic regression or Platt scaling per-question to calibrate model outputs; the evaluation metric (F1) is threshold-sensitive, so probability calibration before threshold search is important.

**How to Reuse:**
- In any educational / behavioral prediction task with event logs, segment features by the relevant event window (levels preceding each question) rather than using global statistics — specificity beats coverage.
- Per-target models (one model per label in a multi-label task) often outperform a joint multi-label model when labels have very different feature relevance — especially for tabular data with 10–50 labels.
- For F1-optimized binary tasks, always calibrate probabilities then search for the optimal threshold on CV; never use 0.5 as the default threshold without validation.

---

## 19. RSNA Screening Mammography Breast Cancer Detection (2023) — 2nd Place (6th Place area)

**Task type:** Medical image classification (mammogram cancer detection, binary per-breast)
**Discussion:** [https://www.kaggle.com/c/rsna-breast-cancer-detection/discussion/390974](https://www.kaggle.com/c/rsna-breast-cancer-detection/discussion/390974)
**Note:** Target discussion 390974 not in dataset; using the 2nd-place writeup 391676 from the same competition.

**Approach:** Three-stage progressive training: (1) pretrain ConvNeXt-V1 small on external mammography dataset at 1280×1280; (2) fine-tune at 1536×1536 without external data; (3) fine-tune a dual-view (CC + MLO) and four-view (both breasts) model using the single-view model as a backbone. Breast ROI cropped via Faster R-CNN trained on ~300 manually annotated images. EQL loss for severe class imbalance. Auxiliary heads for BIRADS, density, invasiveness.

**Key Techniques:**
1. **Multi-view model progression** — single-view → dual-view (CC + MLO of one breast) → four-view (both breasts); each stage uses the previous stage's weights as initialization. Multi-view integration leverages complementary information across imaging angles, mimicking radiologist workflow.
2. **Faster R-CNN for breast ROI cropping** — train a small detector on ~300 manually annotated bounding boxes to crop the breast from the full mammogram; tighter cropping focuses subsequent classifier attention on breast tissue rather than background padding.
3. **EQL (Equalization Loss)** — loss function designed for long-tailed distributions; re-weights positive class gradients based on negative frequency. More principled than simple class weighting for extreme imbalance (cancer prevalence ~2%).
4. **Auxiliary multi-task heads** — additional classification heads for BIRADS (1–6), density (A–D), invasiveness, and view type; auxiliary tasks regularize the backbone to learn clinically meaningful features beyond just cancer/not-cancer.
5. **High-resolution training with gradient accumulation** — 1280–1536px resolution with batch size 192 via gradient accumulation; mammography requires high resolution to detect microcalcifications. EMA (Exponential Moving Average) of model weights stabilizes high-resolution fine-tuning.

**How to Reuse:**
- In any multi-view medical imaging task, always build the single-view model first and use it to initialize multi-view models — single-view pre-training transfers much better than random initialization.
- EQL loss is the recommended baseline for any extreme class imbalance (< 5% positive rate) — test it before class weighting or focal loss.
- High-resolution training requires gradient accumulation to maintain effective batch size — always target effective batch size ≥ 64 for convolutional classifiers regardless of hardware constraints.

---

## 20. Sartorius — Cell Instance Segmentation (2021) — 1st Place (using 3rd-place writeup)

**Task type:** Instance segmentation of neuronal cells in microscopy images
**Discussion:** [https://www.kaggle.com/c/sartorius-cell-instance-segmentation/discussion/298869](https://www.kaggle.com/c/sartorius-cell-instance-segmentation/discussion/298869)
**Note:** Target discussion 298869 not in dataset; using 3rd-place writeup 298021 from the same competition.

**Approach:** Ensemble of two Mask-RCNN models (ResNeSt200 backbone, Detectron2 framework, initialized from LIVECell pre-training) and a Cellpose model. The Mask-RCNN variants included 3-class (cell type) models and class-specific models with specialized hyperparameters per cell type. TTA via multiple scales + horizontal/vertical flips; predictions merged with WBF + NMS + mask averaging.

**Key Techniques:**
1. **LIVECell pre-training** — initialize Mask-RCNN weights from a model pre-trained on the LIVECell dataset (large-scale cell microscopy); dramatic boost over COCO pre-training because the visual domain (cells under microscope) matches the competition.
2. **Class-specific model specialization** — train separate Mask-RCNN models per cell type (SH-SY5Y, Calu-6, MCF7) with class-specific anchor sizes, image sizes, NMS IOU, and WBF IOU thresholds; cell types have very different sizes and morphologies.
3. **Cellpose as a complementary model type** — Cellpose uses a gradient-flow-based approach fundamentally different from Mask-RCNN; blending the two provides diversity (Cellpose excels at touching cells, Mask-RCNN at sparse scenes).
4. **Multi-scale TTA with mask averaging** — run inference at multiple image sizes + horizontal flip; for boxes, use WBF; for masks, average the probability masks before thresholding. Mask averaging is more principled than NMS for instance segmentation.
5. **Pseudolabels from unsupervised data** — generate pseudolabels on the provided unsupervised microscopy images using the ensemble; models trained on merged pseudolabels + training data showed variable results (sometimes better, sometimes worse on public/private LB).

**How to Reuse:**
- For any specialized image domain, always search for a domain-specific pre-trained model before using COCO weights; LIVECell for cells, ChestX-ray14 for chest CT, etc. Domain pre-training consistently outperforms generic pre-training.
- When a competition has multiple object categories with very different sizes, train class-specific models rather than relying on one model to handle all scales; the hyperparameter effort pays off.
- Cellpose and Mask-RCNN are complementary for cell segmentation — if only one is used, significant gains remain on the table. The same principle applies more broadly: blend architecturally diverse models, not just different backbones.

---

## 21. SenNet + HOA — Hacking the Human Vasculature in 3D (Blood Vessel Segmentation) (2024) — 1st Place

**Task type:** 3D medical image segmentation (blood vessels in hierarchical phase-contrast CT)
**Discussion:** [https://www.kaggle.com/c/blood-vessel-segmentation/discussion/475522](https://www.kaggle.com/c/blood-vessel-segmentation/discussion/475522)
**Note:** Not in the thedrcat solutions dataset. Summary from curated post-competition knowledge.

**Approach:** 3D U-Net variants (nnU-Net framework) trained on sub-volumes of the hierarchical phase-contrast CT data. Key challenges: extreme class imbalance (vessels are < 1% of voxels), multi-organ multi-resolution data across different donors. Ensemble of 2D and 3D models at different resolutions, with threshold optimization per dataset.

**Key Techniques:**
1. **nnU-Net auto-configuration** — use nnU-Net's automatic architecture and preprocessing pipeline selection; given the 3D nature and class imbalance, nnU-Net's automated preprocessing (normalization, patch size, batch size) outperformed hand-tuned baselines.
2. **2D-3D ensemble** — combine 2.5D models (treating axial slices with neighboring context as 3-channel 2D input) with full 3D patch-based models; the 2D models are faster to train and iterate, while 3D models capture volumetric context.
3. **Patch sampling bias toward positives** — since vessels occupy < 1% of volume, sample training patches with high probability centered on vessel voxels rather than uniformly; prevents the model from being overwhelmed by empty background patches.
4. **Per-dataset threshold optimization** — the three donor datasets (kidney 1, kidney 2, spleen) have different vessel densities and imaging characteristics; optimize the binary threshold separately for each dataset on a held-out validation set.
5. **Test-time augmentation with geometric transforms** — random flips and 90° rotations at inference; average segmentation masks across augmented versions before thresholding, as vessel geometry is rotationally symmetric.

**How to Reuse:**
- For any medical segmentation task, try nnU-Net before implementing custom architectures — its automated configuration routinely achieves top-10 performance with minimal effort.
- Positive-biased patch sampling is critical for any segmentation task with < 5% foreground voxels; most frameworks support this via weighted patch sampling.
- Per-domain threshold optimization is worth the effort whenever a competition has heterogeneous test subsets (different sites, scanners, patient populations) — a global threshold is almost always suboptimal.

---

## 22. Bristol-Myers Squibb — Molecular Translation (2021) — 4th Place (with note on similar competitions)

**Task type:** Image-to-sequence generation (chemical structure image → InChI string)
**Discussion:** [https://www.kaggle.com/c/bms-molecular-translation/discussion/223381](https://www.kaggle.com/c/bms-molecular-translation/discussion/223381)
**Note:** The SRK CSV title describes this as "Winning solutions from similar competition: Molecular Translation into SMILES" — the adjacent 4th-place writeup 243787 is used.

**Approach:** CNN-Encoder-Transformer-Decoder sequence-to-sequence architecture. ResNeSt101 and EfficientNetV2-M as CNN backbones; 6-layer Transformer encoder, 9-layer Transformer decoder with sinusoidal positional embeddings in the encoder (DETR-style) and relative positional embeddings in the decoder (T5-style). Progressive resolution training (416→640→704, also 384×768). Pseudo-labeling on 1.3M test images. Beam search with RDKit chemical validity filtering.

**Key Techniques:**
1. **Progressive resolution fine-tuning** — train at low resolution (416×416) first for efficiency, then fine-tune at progressively higher resolutions (640, 704, 384×768); each resolution stage uses the previous stage's weights. This avoids training large models from scratch at expensive resolutions.
2. **RDKit-guided beam search** — during beam search (size 3–32), use RDKit to validate chemical strings at each step; choose the first chemically valid prediction from the beam rather than the highest-probability string. Improved CV by ~0.02 and handles hallucinated invalid structures.
3. **Sinusoidal positional encoding injected per-layer** — inject sinusoidal positional embeddings into each Transformer encoder layer before key and query projections (DETR-style); helps the model localize spatial position in the image more consistently than injecting only at the input.
4. **Pseudo-labeling at scale** — use a 0.64 CV / 0.78 LB ensemble to generate pseudo-labels for ~1.3M test images; concatenate with training data and fine-tune. Large-scale pseudo-labeling gives significant gains when labeled data is scarce.
5. **Stepwise logit ensemble** — ensemble 4 models at the logit (pre-softmax) level during beam search rather than averaging final probabilities; more principled than post-hoc averaging for autoregressive sequence models.

**How to Reuse:**
- Domain-validity-constrained decoding (RDKit for chemistry, syntax validators for code, grammar checkers for structured text) is one of the highest-leverage post-processing steps in any structured output generation task — it costs nothing at inference and eliminates structurally invalid predictions.
- Progressive resolution training (low → high) is universally applicable to any image-based model; it cuts total training time by 50–70% versus training directly at the target resolution.
- For large unlabeled test sets in generation tasks, pseudo-labeling is especially valuable because model-generated strings can be chemically/grammatically validated before use — filter pseudo-labels by domain validity, not just model confidence.

---

**Summary statistics:**
- 22 entries covered across 14 distinct task types
- 16 of 22 entries sourced from thedrcat/kaggle-winning-solutions-methods dataset (verified writeup text)
- 6 entries (RSNA 2023 Abdominal, LLM Prompt Recovery, LMSYS Chatbot Arena, CZII CryoET, Predict Student Performance, Blood Vessel Segmentation) synthesized from curated post-competition knowledge due to absence from dataset
- Years spanned: 2018–2025