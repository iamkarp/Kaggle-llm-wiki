# Kaggle Past Solutions — SRK Round 2, Batch 2

Source: kaggle.com/sudalairajkumar/winning-solutions-of-kaggle-competitions
Ingested: 2026-04-17
Note: These are non-1st-place solutions (2nd, 3rd, top-10) with 100+ upvotes. Content sourced from thedrcat/kaggle-winning-solutions-methods and tahaalselwii/kaggle-winning-solutions-digest datasets; for entries not in those datasets, content drawn from well-documented public write-ups in the ML community.

---

## 1. Child Mind Institute — Detect Sleep States (2023) — Listed as "1st place" but ranked 2nd in CSV

**Task type:** Time series segmentation — detect sleep onset and wakeup events from wrist accelerometer data
**Discussion:** https://www.kaggle.com/c/child-mind-institute-detect-sleep-states/discussion/459715

**Approach:** A hybrid 2-level pipeline. Level 1 is a CNN → Residual GRU → CNN architecture that produces per-minute predictions of sleep/wake state. Level 2 aggregates those per-minute features and runs LightGBM, CatBoost, CNN, and Transformer models to sharpen temporal precision. Key post-processing steps include decaying target labels, periodicity filtering, confidence score recalibration, and optimized peak detection. Final score: 0.852 event-detection AP.

**Key Techniques:**
1. **CNN-GRU-CNN hybrid backbone** — CNN encodes local accelerometer patterns, residual GRU captures long-range temporal dependencies, second CNN refines local context from GRU output; more stable than pure RNN on long sequences.
2. **Minute-based positional embedding** — encodes time-of-day at minute granularity as a learnable embedding; allows the model to exploit circadian rhythm regularity without hard-coding a threshold.
3. **Decaying target labels** — labels near annotated onset/wakeup events are assigned soft targets that decay with distance, smoothing the training signal and avoiding overconfident predictions near ambiguous boundaries.
4. **Two-level stacking with tree models** — the neural backbone's per-minute probabilities become features for LightGBM/CatBoost at the second level, allowing gradient boosting to correct systematic biases in the neural output.
5. **Confidence-recalibrated peak detection** — instead of a fixed threshold, peak prominence and estimated confidence scores jointly gate which candidate events survive post-processing, reducing false positives.

**How to Reuse:**
- For time series event detection tasks, stacking a neural sequence model's soft predictions into a second-level gradient booster consistently outperforms using either alone.
- Minute-of-day or hour-of-day embeddings are low-cost features for any domain with circadian or weekly periodicity (health, energy, finance).
- Soft/decaying labels near boundaries are especially valuable when ground-truth annotations have inherent ambiguity (annotator disagreement, edge cases).

---

## 2. Feedback Prize — Evaluating Student Writing (2021) — 6th Place

**Task type:** NLP token classification — identify and classify argumentative discourse elements (Claim, Evidence, Lead, etc.) in student essays
**Discussion:** https://www.kaggle.com/c/feedback-prize-2021/discussion/313424

**Approach:** Reformulated the task as a text-based object detection problem inspired by YOLO. Rather than standard BIO tagging, the solution used a custom token classification head with three parallel outputs: objectness score (does this token start a span?), regression to span start/end indices, and discourse type classification. Word-level logits were aggregated using RoIAlign. Models: DeBERTa-large and DeBERTa-xlarge. Post-processing via simple NMS. Achieved 0.732 F1.

**Key Techniques:**
1. **YOLO-style text span detector** — reformulates NER as object detection; each token predicts objectness + span bounds + class, allowing overlapping spans and variable-length segment detection without BIO constraint limitations.
2. **RoIAlign for word-level aggregation** — borrows from object detection to pool subword token representations into word-level features, preserving spatial precision of span boundaries.
3. **Masked token augmentation** — randomly masks tokens during training to improve robustness; functions like word dropout for transformers and was found to help on long essay inputs.
4. **Custom positive sampling strategy** — during training, positives (span starts) are heavily upsampled relative to background tokens to prevent the class imbalance from dominating the loss.
5. **NMS post-processing** — non-maximum suppression removes redundant overlapping span predictions, a direct transfer from computer vision that works well when spans can overlap.

**How to Reuse:**
- Reframe NER/span extraction tasks as object detection when spans can nest or overlap; the YOLO-style head gives finer-grained control than BIO.
- RoIAlign-style pooling is useful whenever you need word-level features from subword tokenizers — handles tokenizer alignment cleanly.
- NMS is broadly applicable post-processing for any task where the model generates multiple candidate spans or bounding boxes per region.

---

## 3. ASHRAE — Great Energy Predictor III (2019) — 1st Place (listed 2nd in CSV)

**Task type:** Regression — predict energy consumption (chilled water, electric, hot water, steam) for 1,448 buildings over 12 months
**Discussion:** https://www.kaggle.com/c/ashrae-energy-prediction/discussion/124709

**Approach:** Team Isamu & Matt combined two independent solution pipelines and averaged their predictions. Both pipelines were LightGBM-based, with extensive data cleaning (removing outlier readings, correcting timestamp alignment issues, and fixing a site 0 timezone bug that was the biggest single data quality issue). Features were calendar-based (hour, day, month, weekday, holiday flags), building metadata (square footage, primary use type, floor count, age), and lagged weather variables. The final ensemble of 16 LightGBM models (different seeds + data splits) achieved 0.930 RMSLE.

**Key Techniques:**
1. **Data cleaning as the primary lever** — identified and removed meter reading anomalies (zero-reading streaks, implausible spikes); corrected the site 0 timestamp timezone error; removing bad data improved CV more than most model changes.
2. **LightGBM with cyclic calendar features** — encoding hour, weekday, and month as cyclical features (sin/cos) rather than integers; separate models per meter type to avoid leakage across energy sources.
3. **Building metadata features** — square footage, primary use category, floor count, year built, and their interactions with calendar features; allowed the model to learn usage patterns per building class.
4. **Weather lag and rolling averages** — lagged temperature, humidity, wind speed by 1–72 hours; rolling means over 24h and 168h (1 week) windows capture thermal inertia of buildings.
5. **Pipeline ensemble via simple averaging** — two independently built pipelines averaged at prediction level; diversity from different feature engineering choices rather than different models provides robust generalization.

**How to Reuse:**
- For energy/utility forecasting always audit data quality before modeling; timezone errors and meter anomalies routinely dwarf algorithmic improvements.
- Separate models per target type (electric vs. steam vs. chilled water) often beat a single model with target as a feature, especially when usage patterns differ structurally.
- Rolling weather lags (24h, 168h) are a general-purpose feature for any problem with physical inertia (temperature, pollution, traffic).

---

## 4. Cassava Leaf Disease Classification (2020) — 3rd Place

**Task type:** Image multiclass classification — 5 classes of cassava leaf disease + healthy
**Discussion:** https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/221150

**Approach:** Ensemble of three Vision Transformer (ViT) models: `vit_base_patch16_384` (best single model, 384×384 input), and two variants of `vit_base_patch16_224` operating at 448×448 by tiling the input into four overlapping crops and attention-layer-weighted averaging the results. All models used 5× TTA. Final weighted average across all three models: 0.9028 private LB. The author deliberately avoided large EfficientNet variants that showed overfitting signs on this noisy dataset.

**Key Techniques:**
1. **Vision Transformer (ViT) over CNNs** — ViT outperformed EfficientNet, SE-ResNeXt on this dataset despite being newer; patch-based global attention captured long-range disease patterns across leaves better than local convolutions.
2. **Tile-and-weight aggregation for resolution mismatch** — to run a 224-patch ViT on 448×448 images, input is split into four 224×224 crops; each crop's attention weights determine how predictions are combined, preserving fine-grained spatial information.
3. **Label smoothing (alpha=0.01)** — light label smoothing on noisy ground-truth labels improved private LB slightly; cassava competition labels were known to be noisy with ~15% error rate.
4. **Avoiding complexity as anti-overfitting** — deliberately dropped large-image EfficientNets because CV improvements didn't transfer to LB, indicating overfitting; restraint on model capacity was a competitive advantage.
5. **Multi-year data (2019 + 2020)** — incorporating 2019 competition data as additional training images; combined dataset regularized the model against the noisy 2020 labels.

**How to Reuse:**
- When CNN and ViT both work, try ViT first for fine-grained classification where global context matters (disease patterns, product defects spanning the image).
- Tile-and-aggregate is a practical trick for running fixed-resolution models at higher resolution without retraining.
- On noisy-label datasets, simpler models with light label smoothing often beat complex models that memorize noise.

---

## 5. Avito Demand Prediction Challenge (2018) — 2nd Place

**Task type:** Regression — predict demand (probability of a deal) for online classified ads; features include text, images, metadata, and pricing
**Discussion:** https://www.kaggle.com/c/avito-demand-prediction/discussion/59871

**Approach:** A multi-modal ensemble combining gradient boosting on tabular features, NLP on ad text, and CNN image features. Text was encoded with TF-IDF and FastText embeddings; images passed through a pretrained ResNet or VGG to extract embedding vectors. A deep neural network combining all modalities was trained end-to-end, and its predictions were blended with LightGBM trained on hand-crafted tabular + aggregated text/image features. Price-level aggregations and target-encoded category statistics were critical.

**Key Techniques:**
1. **Multi-modal neural fusion** — concatenated image embeddings (pretrained CNN), text embeddings (FastText + TF-IDF), and tabular features into a single MLP; trained jointly to allow cross-modal interactions.
2. **Target encoding with groupby statistics** — mean deal probability by category, city, user, and their intersections; leak-free target encoding with out-of-fold estimates prevents target leakage on train.
3. **Price log-transformation and binning** — price spans many orders of magnitude; log(1+price) and quantile binning per category stabilized LightGBM feature splits and reduced outlier influence.
4. **FastText word embeddings for Russian text** — pretrained FastText on Russian Common Crawl gave better coverage of misspellings and colloquial ad language than character n-grams alone.
5. **LightGBM + NN blend** — tree models and neural network capture different patterns (trees: interactions and thresholds; NNs: smooth representations); blending at prediction level consistently outperformed either alone.

**How to Reuse:**
- For marketplace/e-commerce tasks with image + text + tabular data, always build a multi-modal path even if individual modality scores are modest; the combination frequently surprises.
- Target-encoding at multiple group levels (category, city, user) provides powerful signals with low feature engineering cost.
- Train separate LightGBM and NN models, then blend; the diversity almost always beats a single best model even if the NN is weaker on its own.

---

## 6. CommonLit Readability Prize (2021) — 2nd Place

**Task type:** NLP regression — predict reading difficulty of text passages (continuous score, evaluated on RMSE)
**Discussion:** https://www.kaggle.com/c/commonlitreadabilityprize/discussion/258328

**Approach:** An ensemble of 19 diverse language models with hand-tuned weights (some weights negative). Models included RoBERTa variants, DeBERTa-large/xlarge/xxlarge, BART-large, Electra-large, Funnel Transformer, GPT2-medium, ALBERT-xxlarge, T5-large, and classical ML heads (SVR, Ridge) on top of RoBERTa embeddings. Weights were initialized via Nelder-Mead optimization on CV, then hand-tuned against public LB. Post-processing applied different multiplicative coefficients based on the predicted score range. Best CV: 0.4449, Private LB: 0.447.

**Key Techniques:**
1. **Negative-weight ensembling** — allowing ensemble weights to go negative (some models act as "correction" rather than "addition") captures error patterns not visible from individual model performance; optimized via Nelder-Mead with no sum-to-1 constraint.
2. **MLM continued pretraining** — one RoBERTa-base model was further pretrained on the competition's text corpus with masked language modeling before fine-tuning; domain adaptation improved that model's CV by ~0.015.
3. **Training without dropout** — disabling dropout for all models except the classical-head variants let models fit more tightly to the small training set (~2,800 examples); regularization came from ensemble diversity instead.
4. **Post-processing by score range** — predictions in different quantile ranges were multiplied by different empirically tuned coefficients; exploits the observation that model calibration differs at high vs. low readability.
5. **GPT2 as a diversity model** — including GPT2-medium (an autoregressive LM) alongside encoder-based transformers added genuine architectural diversity; its weight (0.17) was among the highest in the final ensemble.

**How to Reuse:**
- On small NLP datasets, train without dropout and rely on ensemble diversity for regularization rather than within-model regularization.
- Nelder-Mead ensemble weight optimization with no sum-to-1 constraint is worth trying; negative weights often reveal that a weak model is anti-correlated with the ensemble error.
- Continued MLM pretraining on the competition's own text (even a few thousand examples) gives meaningful improvements; it's cheap relative to full pretraining.

---

## 7. Riiid Answer Correctness Prediction (2020) — 3rd Place

**Task type:** Sequential binary classification — predict whether a student will answer a question correctly given their full interaction history; real-time inference required
**Discussion:** https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/209585

**Approach:** Ensemble of two Transformer models with customized attention. Architecture: encoder-decoder with 3+3 or 4+4 layers, no LayerNorm, T-Fixup initialization, d_model=512, trained on sequences of 500 interactions. Continuous and categorical features embedded jointly. Strict causal masking prevented future data leakage. Key engineered features: time deltas between questions, repeated-question tracking, timestamp periodicity (hour-of-day). Real-time inference handled by "The Blindfolded Gunslinger" — a dynamic time-aware ensemble method adapting weights under API time limits. Final: 0.818 AUC.

**Key Techniques:**
1. **T-Fixup initialization without LayerNorm** — replaces LayerNorm with careful weight initialization that keeps gradients stable; allowed stable training of deep transformers on sequential interaction data without the normalization overhead.
2. **Causal masking for knowledge tracing** — strict attention masking ensures the model only sees past interactions, preventing future leakage in the sequential prediction task; implemented with careful handling of the task_container_id grouping.
3. **Time-delta and periodicity features** — logarithm of elapsed time between interactions captures the forgetting curve; hour-of-day periodicity features capture study session patterns; both are low-cost but high-impact.
4. **Repeated question tracking** — whether and how many times a student has seen this exact question before, and their accuracy on prior attempts; directly captures the testing effect (retrieval practice).
5. **Dynamic ensemble under time constraint ("Blindfolded Gunslinger")** — monitors remaining API time during inference and adjusts ensemble composition dynamically; a practical engineering solution unique to real-time Kaggle competitions.

**How to Reuse:**
- T-Fixup initialization is a drop-in LayerNorm replacement for any transformer; worth trying when LN causes instability or on non-standard sequence lengths.
- For knowledge tracing / recommendation / sequential prediction: elapsed time since last event and repetition count are near-universal high-value features.
- Design inference code to be time-budget aware from the start if the competition has API time limits; retrofitting is painful.

---

## 8. 2019 Data Science Bowl (2019) — 3rd Place

**Task type:** Tabular/sequential regression — predict children's game assessment accuracy group from game event logs
**Discussion:** https://www.kaggle.com/c/data-science-bowl-2019/discussion/127891

**Approach:** Single Transformer model (no position embedding) applied to sequences of game sessions per installation_id. Raw log data was aggregated by game_session into crosstab counts of event codes, event IDs, game accuracy, and max round. Categorical features were embedded independently, concatenated, and projected via a linear layer; continuous features were embedded directly. The key finding: standard position embedding (BERT/ALBERT/GPT2-style) hurt CV score — because session order within an installation is not as important as session content composition. Private LB: 0.564.

**Key Techniques:**
1. **Removing positional embeddings from Transformer** — standard position embeddings degraded performance; the model performs better as a set-aggregator over game sessions rather than a sequence model, since order within an installation matters less than composition.
2. **Session-level aggregation via crosstab** — aggregating raw event logs by game_session using cross-tabulation of event codes and IDs compresses variable-length sessions into fixed-size count vectors, making them usable as Transformer input tokens.
3. **Separate categorical embeddings** — each categorical column (event_code, title, world, etc.) gets its own embedding table; all embeddings concatenated and projected to hidden_size/2; allows the model to learn independent representations before interaction.
4. **Training-time and test-time augmentation** — applies augmentation during training (random session reordering) and TTA at inference; for a set-based model, permutation augmentation is valid and provides regularization.
5. **Architectural first-principles search** — author tested BERT, ALBERT, GPT2, and found they all underperformed due to position bias; rather than blindly following state-of-the-art, the solution was tailored to the data structure.

**How to Reuse:**
- For tabular sequence tasks where order doesn't matter (basket-of-behaviors, set of sessions), a Transformer without position embeddings is a natural fit and often beats order-assuming RNNs.
- Crosstab aggregation is a fast, effective way to convert variable-length event logs into fixed-size feature vectors for any session-based problem.
- Test which position encoding assumption actually fits your data; position embeddings are not always helpful and can hurt when the data is permutation-invariant.

---

## 9. U.S. Patent Phrase to Phrase Matching (2022) — 2nd Place

**Task type:** NLP regression — predict semantic similarity between patent anchor phrases and target phrases within a CPC code context; evaluated by Pearson correlation
**Discussion:** https://www.kaggle.com/c/us-patent-phrase-to-phrase-matching/discussion/332234

**Approach:** Exploited the strong correlation between all targets sharing the same anchor. Stage 1: trained DeBERTa-v3-large and BERT-for-patents models with grouped target lists added to context. Stage 2: used out-of-fold scores from Stage 1 to add quantitative context (e.g., "target1 23, target2 47…" with scores ×100 as tokens). Used FGM adversarial training, EMA (decay=0.999), and knowledge distillation (soft labels from ensemble OOF). CV strategy: StratifiedGroupKFold to prevent anchor leakage. Final LB: 0.8775 Pearson correlation.

**Key Techniques:**
1. **Target grouping as context augmentation** — grouping all targets under the same anchor and appending them to the input context gives the model crucial correlation signals: if target A is highly similar to anchor, other semantically related targets should score high too.
2. **OOF score injection as token** — in Stage 2, multiplying OOF similarity scores by 100 and appending them as text tokens ("target1 47") creates a self-referential context; the model learns to calibrate predictions using the score distribution of related phrases.
3. **FGM adversarial training** — Fast Gradient Method perturbations on input embeddings during training (eps=0.1) improved single-model CV by 0.002–0.005; standard regularizer for NLP fine-tuning that's easy to add.
4. **Knowledge distillation from ensemble** — using ensemble OOF soft labels to train individual models; a single distilled model matches ensemble performance, saving inference time while maintaining diversity for re-distillation.
5. **BERT-for-patents domain adaptation** — using a model pretrained on patent text gave better baseline representations for this highly specialized legal/technical vocabulary.

**How to Reuse:**
- When multiple examples share a common "anchor" or "group," injecting all group members as context is a powerful and underused technique for NLP similarity tasks.
- OOF score injection at Stage 2 is a general trick for multi-stage NLP pipelines — first predict, then use predictions as features in the second stage.
- BERT-for-patents (or other domain-specific models from HuggingFace) should always be tried when domain vocabulary is specialized; fine-tuned generalist models rarely beat domain pretraining.

---

## 10. M5 Forecasting — Accuracy (2020) — 4th Place

**Task type:** Hierarchical time series forecasting — predict 28-day unit sales for 30,490 Walmart products across 10 stores; metric: WRMSSE
**Discussion:** https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/163216

**Approach:** A deliberately simple, practical solution: single LightGBM model per store per week-horizon (4 week-horizon models × 10 stores = 40 models total), each predicting 7 days of sales. Objective: Tweedie regression. Five holdout periods for validation. No recursive features (to avoid error accumulation), no post-processing or leakage-exploiting multipliers, no stacking. Used general calendar, price, and time-series features applicable to any retail forecasting task. Result: 4th place despite the simplicity — a compelling case for practical engineering over overfit optimization.

**Key Techniques:**
1. **Tweedie loss for count data** — Tweedie regression handles zero-inflated count data (many products have zero sales on many days) better than MSE; appropriate objective for retail demand where zeros are real, not missing.
2. **Separate models per store and week horizon** — training 4 separate models (F1–F7, F8–F14, F15–F21, F22–F28) per store avoids recursive forecasting error accumulation and lets each model specialize in near vs. far horizons.
3. **Avoiding recursive features** — features derived from future predictions (recursive) accumulate errors over the forecast horizon; using only lag features computed from actual history makes the pipeline robust and deployable.
4. **Five holdout validation periods** — M5's WRMSSE varies dramatically across time periods; evaluating on 5 different historical windows gives a more stable CV signal than a single holdout.
5. **Trusting CV over leaderboard metric optimization** — deliberately did not tune for WRMSSE; instead optimized for stable CV, which prevented overfitting to competition-specific quirks.

**How to Reuse:**
- Use Tweedie or Poisson loss for any intermittent/zero-heavy demand forecasting; MSE penalizes zeros incorrectly.
- Separate models per forecast horizon consistently outperform single recursive multi-step models for horizons beyond 7 days.
- Validate on multiple holdout windows (not just one) for time series — seasonal effects, promotions, and regime changes make single holdouts unreliable.

---

## 11. Bengali.AI Handwritten Grapheme Classification (2019) — 2nd Place

**Task type:** Image multiclass classification (3 simultaneous outputs: grapheme root, vowel diacritic, consonant diacritic)
**Discussion:** https://www.kaggle.com/c/bengaliai-cv19/discussion/135966

**Approach:** Started with 3-head models (root, consonant, vowel) but pivoted to predicting individual graphemes to simplify the loss. Used FMix augmentation (superior to CutMix for this task) that realistically blends 2–3 images using Fourier-space masks. Built a post-processing pipeline averaging grapheme component probabilities to boost recall on rare consonant diacritics (classes C=3 and C=6). Final solution: 7 SE-ResNeXt50/101 models blended with component-wise weighted strategies. Private LB: 0.9689.

**Key Techniques:**
1. **FMix augmentation over CutMix** — FMix creates smooth, Fourier-space masks rather than rectangular cutouts; produces more realistic blended images for handwriting, where strokes are continuous rather than block-shaped.
2. **Component-wise probability averaging** — instead of predicting the full grapheme composite directly, averaging probabilities of each component (root, vowel diacritic, consonant diacritic) separately and recombining; improves recall on rare component classes.
3. **Individual grapheme prediction head** — switching from a joint 3-output head to independent single-task models simplifies gradient flow and loss balancing; each component model can specialize.
4. **Image size scaling (224×224)** — found that small image size was sufficient; larger images didn't improve enough to justify compute; selecting the right resolution is as important as architecture.
5. **7-model SE-ResNeXt ensemble** — SE-ResNeXt provides channel-wise attention (Squeeze-and-Excitation) that helps focus on stroke-relevant features; ensembling 7 models with different seeds captured variance.

**How to Reuse:**
- Try FMix as an alternative to CutMix whenever the objects of interest have smooth boundaries or continuous spatial structure (handwriting, medical images).
- For multi-label tasks with rare classes, averaging per-component probabilities independently (rather than joint prediction) often rescues recall on low-frequency labels.
- Component-wise model specialization (one model per output) reduces gradient interference between dissimilar sub-tasks.

---

## 12. Optiver Realized Volatility Prediction (2021) — "Previous Competitions Winning Solutions" reference post

**Task type:** Financial time series regression — predict realized volatility of stocks over 10-minute windows using order book and trade data; metric: RMSPE
**Discussion:** https://www.kaggle.com/c/optiver-realized-volatility-prediction/discussion/249523

**Approach:** This discussion post linked to winning solutions from related prior competitions rather than presenting a new solution. The top approaches in the actual competition used: (1st) Nearest Neighbor feature engineering on 600+ features with LightGBM + 1D-CNN + MLP ensemble; (3rd) LightGBM + ANN stacking with time_id nearest neighbor features and target transformation. The key competitive insight across all top solutions was the "time_id nearest neighbor" — finding similar market microstructure states across time and using their volatilities as features.

**Key Techniques:**
1. **Nearest neighbor features across time_id** — for each stock's 10-minute window, find the k most similar windows (by order book shape) across all time_ids and use their realized volatilities as features; directly encodes "similar market conditions → similar volatility."
2. **600+ feature engineering on order book** — WAP (weighted average price), log returns, spread, bid-ask imbalance, trade size distributions, rolling means/stds across multiple time horizons; comprehensive feature set more important than model complexity.
3. **t-SNE for time ordering reconstruction** — the competition shuffled time_ids to prevent leakage; top solution reconstructed the true order using t-SNE on order book features, enabling proper time-series CV and covariate shift detection.
4. **Target transformation (log + stabilization)** — volatility is right-skewed; log transformation stabilizes variance and makes LightGBM's tree splits more effective; combined with stock-level normalization.
5. **LightGBM + 1D-CNN + MLP ensemble** — three architecturally diverse models capture complementary patterns: trees for threshold interactions, 1D-CNN for local temporal patterns, MLP for smooth feature combinations.

**How to Reuse:**
- Nearest neighbor features (find similar historical situations and use their outcomes as features) are powerful for any time series or financial forecasting problem; especially effective when the target has clear regime-dependent behavior.
- Order book features (WAP, bid-ask imbalance, order flow imbalance) are the standard feature set for any microstructure volatility or price impact problem.
- When competition shuffles time to prevent leakage, try reconstructing order via feature-space similarity (t-SNE, UMAP) — enables proper time-series CV.

---

## 13. Home Credit Default Risk (2018) — Open Solution Journal (LB 0.806)

**Task type:** Binary classification — predict probability of loan default for applicants with limited credit history; metric: AUC
**Discussion:** https://www.kaggle.com/c/home-credit-default-risk/discussion/57175

**Approach:** An open-source collaborative solution (Neptune.ai team) that documented their end-to-end pipeline publicly throughout the competition. The pipeline combined aggressive feature engineering across all tables (application, bureau, credit card balance, installments, previous applications), null importance-based feature selection, and LightGBM. Key: an open Jupyter notebook workflow where community improvements were incorporated. Final public LB: 0.806.

**Key Techniques:**
1. **Multi-table aggregation feature engineering** — joining and aggregating across all 7 related tables (bureau, credit card, installments, POS cash, previous applications) with statistics (mean, max, min, sum, count) at each aggregation level; the breadth of cross-table features was the main performance driver.
2. **Null importance feature selection** — training LightGBM with permuted targets to measure each feature's importance under the null hypothesis; features with observed importance below null distribution are discarded; reduces dimensionality without losing predictive signal.
3. **Recursive feature depth** — aggregating aggregations (e.g., bureau balance stats → bureau stats → application) creates second-order features that capture credit history patterns not visible at first-order aggregation.
4. **LightGBM with grouped bagging** — training multiple LightGBM models with different random seeds and data subsamples; averaging OOF predictions for final ensemble; captures variance across different bootstrap samples of the highly imbalanced dataset.
5. **Open-source collaborative development** — publishing all code publicly during the competition attracted community contributions; external improvements were integrated, accelerating feature discovery.

**How to Reuse:**
- For any multi-table credit/financial dataset, a systematic aggregation of all relational tables at all levels is the most reliable path to high AUC.
- Null importance feature selection is a rigorous alternative to threshold-based importance pruning; works well when you have many correlated features from automated aggregation.
- Publishing intermediate results openly (even during a competition) can attract quality feedback; in non-competitive settings, this is always the right approach.

---

## 14. Feedback Prize — Evaluating Student Writing (2021) — 4th Place

**Task type:** NLP token classification — identify and classify argumentative discourse elements in student essays
**Discussion:** https://www.kaggle.com/c/feedback-prize-2021/discussion/313330

**Approach:** Combination of DeBERTa-v3-large, DeBERTa-v2-xlarge, and DeBERTa-xlarge models with entity-level post-ensembling. Key implementation details: encoding cleanup (handling tokenizer edge cases like `\n` linebreak preservation), offset-based tagging (using character offsets rather than token indices for span alignment), beam search decoding for BIO tagging instead of greedy argmax, and gradient checkpointing to fit large models in memory. Final: 0.735 F1.

**Key Techniques:**
1. **Beam search BIO decoding** — instead of independently predicting the best tag for each token, beam search decodes the full BIO sequence jointly; enforces tag transition validity and finds globally consistent labelings that greedy argmax misses.
2. **Offset-based span alignment** — using character offsets from the tokenizer's offset_mapping to align subword predictions to word boundaries; avoids common boundary misalignment bugs when tokenizers split punctuation or special characters.
3. **Encoding cleanup (linebreak preservation)** — essays contain `\n` characters that tokenizers often collapse; explicitly preserving linebreaks as tokens improved discourse boundary detection since paragraph breaks often align with argument boundaries.
4. **Entity-level post-ensembling** — rather than averaging token-level probabilities across models, the ensemble operates at the entity (span) level; candidate spans from each model are merged with confidence-weighted voting.
5. **DeBERTa-v3-large as backbone** — DeBERTa-v3 with disentangled attention on content and position outperformed v2 variants on long-document NER tasks; the v3 pretraining on a larger corpus gave better long-range discourse understanding.

**How to Reuse:**
- Always use beam search (or Viterbi) decoding for BIO/BIOES tagging tasks; the marginal compute cost is trivial and greedy errors compound across long sequences.
- Offset-based span alignment is the correct approach for any NER task with subword tokenizers; build this properly from the start, not as an afterthought.
- Preserve document structure tokens (linebreaks, section markers) when the task has document-level semantics; tokenizers that collapse these lose structural signal.

---

## 15. Santander Customer Transaction Prediction (2019) — 5th Place

**Task type:** Binary classification — identify which customers will make a specific transaction; highly anonymized features, extreme class imbalance (~10% positive), metric: AUC
**Discussion:** https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/88897

**Approach:** The top solutions (including 5th) leveraged a critical data insight: test set had fake rows (added to prevent feature engineering leaks), and removing those enabled much more powerful count-based features. The 5th place solution used LightGBM with count features — counting how many times each feature value appears in train+test — which dramatically boosted AUC. These value-frequency features encode the "uniqueness" of each customer's feature values and served as a strong implicit regularizer.

**Key Techniques:**
1. **Value frequency (count) features** — for each of the 200 anonymized features, count how often that exact value appears across the entire dataset; rare values signal unique customers, common values indicate "normal" range; this simple transformation was the single biggest performance driver in the competition.
2. **Fake test row detection and removal** — the competition's test set contained artificially duplicated rows added to prevent count-feature leaks; identifying and removing these (via statistical fingerprinting) dramatically improved count feature quality.
3. **LightGBM with frequency augmented features** — combining the original 200 features with their count versions (200 extra features) gave LightGBM the information it needed to model the extremely non-linear decision boundary.
4. **Augmentation via synthetic minority sampling** — SMOTE-like oversampling of the positive class (only ~10% of samples) to balance the dataset; combined with LightGBM's built-in class_weight parameter for stability.
5. **Pseudo-labeling high-confidence test predictions** — adding high-confidence test predictions back into training with the count features updated; iterative refinement of the count statistics.

**How to Reuse:**
- For anonymized tabular datasets, count features (how often does this exact value appear?) are a powerful unsupervised signal that's often overlooked; they encode distributional rarity of feature values.
- Always check if competition test sets contain artificial rows (duplicates, constants); their presence/absence changes optimal feature engineering strategy.
- Value frequency features are particularly effective when the data is tabular, anonymized, and the positive class has distinctive distributional properties.

---

## 16. Predicting Molecular Properties (2019) — #1 Solution (Hybrid)

**Task type:** Graph regression — predict scalar coupling constants between pairs of atoms in organic molecules; 8 coupling types (1JHC, 2JHH, etc.), metric: mean log-MAE across types
**Discussion:** https://www.kaggle.com/c/champs-scalar-coupling/discussion/106575

**Approach:** A hybrid solution combining Graph Neural Networks (message passing) with gradient boosting. GNN operated on molecular graphs with atom and bond features; gradient boosting used hand-crafted quantum chemistry features (Mulliken charges, interatomic distances, dihedral angles, hybridization). The GNN predictions were stacked as features into LightGBM, with separate models per coupling type. Claimed 1st in the discussion title but listed as a top solution rather than the official 1st; a highly influential hybrid approach referenced by many subsequent solutions.

**Key Techniques:**
1. **Graph Neural Network with message passing** — atoms as nodes, bonds as edges; multiple rounds of message passing aggregate neighborhood information; particularly effective for capturing quantum effects that depend on bond topology rather than 3D coordinates alone.
2. **Per-coupling-type models** — separate LightGBM models for each of the 8 scalar coupling types; each type has different physical characteristics (e.g., 1JHC is one-bond, 3JHH is three-bond); specialization dramatically outperformed a single multi-type model.
3. **Quantum chemistry feature engineering** — Mulliken charges, interatomic distances (direct and through-bond), dihedral angles, hybridization state, electronegativity; domain knowledge from computational chemistry encoded as explicit features.
4. **GNN predictions as LightGBM features (model stacking)** — using the GNN's per-atom or per-pair embeddings as input features for gradient boosting; the GNN learns smooth manifold representations while LightGBM captures sharp decision boundaries.
5. **Attention bias from bond probability matrix (BPP)** — adding the bond probability matrix as a bias to the attention function in the GNN, encoding secondary structure constraints directly into the attention mechanism.

**How to Reuse:**
- For molecular/graph problems, always combine GNN predictions with classical chemistry/physics features in a second-level model; GNNs capture topology, classical features capture domain constraints.
- Per-type specialization (separate models per output category) is broadly applicable whenever outputs have structurally different generating processes.
- Using neural network embeddings/predictions as features in gradient boosting is a general stacking approach that combines the representational power of NNs with the robustness of trees.

---

## 17. Feedback Prize — Evaluating Student Writing (2021) — "Placeholder" / Community Insights Post

**Task type:** NLP token classification — argumentative discourse element identification in student essays
**Discussion:** https://www.kaggle.com/c/feedback-prize-2021/discussion/308992

**Approach:** This discussion post served as a community aggregation of insights and techniques shared during the competition, rather than a single team's solution writeup. It documented a range of techniques that were proven effective: masked token augmentation, sliding window inference for long essays, and various post-processing heuristics. Multiple teams referenced this thread as a source of the masked augmentation technique that became widely adopted.

**Key Techniques:**
1. **Masked token augmentation** — randomly masking tokens during training (beyond standard MLM; applied to the full fine-tuning step) acts as word dropout specifically for long text classification; particularly effective for essays which are much longer than typical NLP benchmarks.
2. **Sliding window inference for long texts** — essays exceeded 512 tokens; sliding window with overlap (stride 128, window 512) processed chunks independently, with predictions averaged in overlap zones; enabled use of standard BERT-family models without truncation.
3. **Weighted ensemble from diverse discourse models** — combining models trained with different span selection strategies (BIO, BIOES, span-based) and different transformers (DeBERTa, Longformer, BigBird) at the entity level.
4. **Post-processing rule injection** — domain-specific rules (minimum span length per discourse type, maximum number of claims per essay) injected after model prediction; improved precision significantly without hurting recall.
5. **Longformer for full essay context** — Longformer's sparse attention over 4096 tokens allowed processing full essays without windowing; provided complementary long-range context to windowed DeBERTa models.

**How to Reuse:**
- Masked token augmentation during fine-tuning is a low-cost regularizer for any long-document NLP task; add it before spending time on model architecture changes.
- Sliding window inference is the practical solution for BERT-family models on long documents; use stride ≈ 1/4 of window size for adequate overlap.
- Inject domain rules as post-processing constraints rather than as training signals when you have strong prior knowledge about output structure (min/max counts, span lengths).

---

## 18. Quora Question Pairs (2017) — Interesting Solutions Roundup

**Task type:** NLP binary classification — predict whether two questions are semantically equivalent (duplicate detection); metric: log-loss
**Discussion:** https://www.kaggle.com/c/quora-question-pairs/discussion/30260

**Approach:** This discussion post aggregated interesting techniques from multiple teams. The winning approaches combined: (1) hand-crafted NLP features (TF-IDF cosine similarity, shared word ratios, edit distance, magic features), (2) LSTM/GRU Siamese networks on word embeddings, and (3) "magic features" — graph-based features counting how many questions are connected to each question via the question graph. The graph features were the most impactful individual contribution.

**Key Techniques:**
1. **Graph connectivity "magic features"** — treating questions as nodes and duplicate pairs as edges; counting the degree of each question node (how many duplicate pairs it participates in) revealed that popular questions had systematically different duplicate rates; a simple but enormously powerful meta-feature.
2. **Siamese LSTM networks** — shared-weight LSTM encoders for each question; cosine similarity of final hidden states used for classification; weight sharing enforces symmetric representations without requiring separate encoder training.
3. **TF-IDF interaction features** — element-wise product and difference of TF-IDF vectors; cosine similarity; character n-gram overlap; these classical features captured lexical similarity orthogonally to the neural features.
4. **Feature stacking with XGBoost** — final model was XGBoost (or LightGBM) on top of: neural network outputs, TF-IDF similarities, graph features, and string-matching features; tree models could learn non-linear combinations that a linear blend could not.
5. **Threshold calibration for log-loss** — the positive class rate in train differs from test; calibrating the model's output probabilities to match the expected test base rate via isotonic regression significantly improved log-loss.

**How to Reuse:**
- For any pairwise matching task (duplicate detection, entity resolution), construct graph features from the connectivity structure of matched pairs; degree and community membership are powerful signals.
- Siamese architectures with weight sharing are the right design choice for symmetric tasks (is A the same as B?); they enforce the desired symmetry by construction.
- Always check if train/test class balance differs and recalibrate predicted probabilities accordingly for log-loss or cross-entropy objectives.

---

## 19. Cornell Birdcall Identification (2020) — 1st Place

**Task type:** Audio multi-label classification — identify bird species from 5-second audio clips; severe class imbalance, noisy labels, out-of-domain test audio
**Discussion:** https://www.kaggle.com/c/birdsong-recognition/discussion/183208

**Approach:** Pretrained DenseNet121 as CNN feature extractor on mel spectrograms, replacing a larger SED (Sound Event Detection) baseline. Multiple audio augmentations: pink noise, Gaussian noise, gain adjustment, SpecAugmentation (time/frequency masking). Mixup during training. Custom focal loss variant to handle noisy secondary labels. Ensemble of 13 models with voting. Inference: 30-second clips with TTA. Final: 0.681 (private LB, highly noisy evaluation).

**Key Techniques:**
1. **DenseNet121 on mel spectrograms** — treating audio classification as image classification on spectrograms; DenseNet's dense connectivity provides strong gradient flow for short, noisy audio clips; more robust than end-to-end audio models for small datasets.
2. **Pink noise and gain augmentation** — pink noise (1/f spectrum) matches real-world background audio better than white noise; random gain adjustment simulates distance variation; both are strong regularizers for audio recorded in diverse field conditions.
3. **SpecAugmentation (time + frequency masking)** — randomly masking time steps and frequency bands forces the model to learn from partial spectrograms; standard for speech but highly effective for bird audio where temporal patterns matter.
4. **Custom focal loss for secondary labels** — birds in the background (secondary labels) are noisy; focal loss with high gamma down-weights easy examples and focuses learning on hard, uncertain predictions near the decision boundary.
5. **13-model ensemble with voting** — majority voting across 13 independently trained models; for multi-label audio with noisy labels, hard voting (rather than probability averaging) is more robust to label noise.

**How to Reuse:**
- Pink noise augmentation is the right choice for any audio task recorded in field/environmental conditions; white noise is too uniform and doesn't generalize to real backgrounds.
- Treating audio as spectrogram images and using pretrained ImageNet CNNs is a proven baseline that's hard to beat for small/medium audio datasets.
- For noisy multi-label tasks, focal loss + hard voting ensemble is more robust than soft probability averaging.

---

## 20. RSNA 2022 Cervical Spine Fracture Detection (2022) — 1st Place (listed as 2nd in CSV)

**Task type:** Medical imaging detection — detect and localize cervical vertebra fractures from CT scans; metric: weighted multi-label log-loss at both vertebra and patient levels
**Discussion:** https://www.kaggle.com/c/rsna-2022-cervical-spine-fracture-detection/discussion/362787

**Approach:** Two-stage pipeline. Stage 1: 3D semantic segmentation (3D U-Net with ResNet18d and EfficientNetV2s encoders) to localize and crop individual cervical vertebrae (C1–C7). Stage 2: 2.5D classification using multiple CNNs (EfficientNetV2s, ConvNeXt, NFNet) processing stacks of axial slices per vertebra, followed by LSTM to capture inter-slice context. Two prediction levels: per-vertebra (Type1) and whole-patient (Type2). Final LB: winning score on private.

**Key Techniques:**
1. **3D U-Net segmentation for localization** — using segmentation to extract vertebrae crops rather than relying on bounding box detection; 3D context captures the volumetric structure of vertebrae more completely than 2D detection.
2. **2.5D representation (multi-slice stacks)** — processing N adjacent CT slices as a multi-channel image through a 2D CNN; captures inter-slice context (fractures span multiple slices) while leveraging powerful 2D pretrained backbones.
3. **LSTM over slice features** — after 2D CNN encodes per-slice features, LSTM processes them as a sequence; captures the spatial ordering of slices within a vertebra without full 3D convolutions.
4. **Two-level prediction (vertebra + patient)** — predicting fracture probability at both the vertebra level and the patient level; allows the loss function to capture both fine-grained localization and clinical-level diagnosis.
5. **Timm library model diversity** — using EfficientNetV2, ConvNeXt, and NFNet from Timm in an ensemble; architecturally diverse backbones trained on different data augmentations provide robust generalization across CT scanner manufacturers.

**How to Reuse:**
- The segment-then-classify pipeline (2-stage: segment ROI → classify within ROI) is the gold standard for medical imaging when fine-grained localization matters.
- 2.5D (multi-slice stack as multi-channel 2D image) is the practical compromise between 2D (ignores depth) and 3D (compute-heavy); use N=3–7 slices depending on the target structure's depth extent.
- LSTM over slice-level CNN features is a clean architecture for volumetric classification; more efficient than 3D CNNs and better than simple max pooling.

---

## 21. Riiid Answer Correctness Prediction (2020) — 6th Place

**Task type:** Sequential binary classification — predict whether a student will correctly answer a question from their full interaction history; real-time inference API
**Discussion:** https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/209581

**Approach:** Ensemble of 7 GRU models trained with different seeds on the full dataset. Architecture: U-GRU (bidirectional but not BiGRU — does a reverse pass first, concatenates output, then forward pass) + content cosine similarity attention (16-head, symmetric). Inputs: question ID, part, correctness history, elapsed time, timestamp diff. Engineered features: log-transformed streaks of correct/incorrect answers per question choice (helps identify students who always pick A). Sequence length 256, 15 epochs with learning rate decay. Single model: 0.8136 val AUC; ensemble: 0.815 LB.

**Key Techniques:**
1. **U-GRU (reverse-then-forward)** — performs a reverse pass over the interaction sequence first, concatenates its output to the original sequence, then runs the forward pass; allows the forward GRU to "see" future context in its initial hidden state without leakage at inference.
2. **Content cosine similarity with 16 heads** — for each interaction, computes cosine similarity between the current question content vector and all historical content vectors under 16 different linear transformations; a lightweight attention mechanism that captures question-to-question semantic similarity.
3. **Answer streak features** — for each possible answer choice (A, B, C, D), log-transform of: number of questions since last pick of this choice, length of current streak, length of streak on any other choice; captures response bias (students who habitually pick the same answer).
4. **Seed ensemble of same architecture** — 7 models with identical architecture trained on different random seeds; diversity from stochastic optimization and mini-batch sampling; provides ~0.002 AUC improvement over single model with minimal engineering cost.
5. **Numpy arrays partitioned by user** — avoiding DataFrames in favor of pre-partitioned numpy arrays organized by user_id; critical for training efficiency on a 100M+ row dataset within Kaggle's memory constraints.

**How to Reuse:**
- U-GRU is a clever trick for sequential prediction when bidirectional context would help but strict causal masking is required; the reverse pre-pass gives the forward GRU lookahead without leakage.
- Answer/choice streak features are applicable to any multiple-choice behavioral prediction problem (A/B testing response, survey data, clickstream).
- For large-scale user interaction data, numpy arrays partitioned by user are dramatically more memory-efficient than DataFrames; always profile memory before running on full training data.

---

## 22. APTOS 2019 Blindness Detection (2019) — 4th Place (2nd after LB cleaning)

**Task type:** Medical image ordinal classification — predict diabetic retinopathy severity grade (0–4) from fundus photographs; metric: quadratic weighted kappa
**Discussion:** https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107926

**Approach:** EfficientNet ensemble (B3, B4, B5) with two-round pseudo-labeling. Pretrained on 2015 Diabetic Retinopathy dataset (~35,000 images). Heavy augmentation via albumentations: Blur, Flip, RandomBrightnessContrast, ShiftScaleRotate, ElasticTransform, Transpose, GridDistortion, HueSaturationValue, CLAHE, CoarseDropout. Pseudo-labeled current test data, fine-tuned, then repeated. Used flips as TTA. Final private LB: top-4 after some LB shake-up from label noise.

**Key Techniques:**
1. **Two-round pseudo-labeling** — predict on test set, use high-confidence predictions as additional training data, fine-tune models, repeat; each round iteratively improves the pseudo-label quality and significantly expands effective training set size.
2. **EfficientNet family diversity** — B3 (300px), B4 (460px), B5 (456px) provide resolution diversity; each model captures different spatial detail levels; blended by simple mean for robustness.
3. **External dataset pretraining (2015 DR data)** — pretraining on 35,000 images from the 2015 Kaggle competition before fine-tuning on 2019 data; gave far stronger initialization than ImageNet for this specific retinal image domain.
4. **Heavy albumentations augmentation** — 10 distinct augmentations per image including CLAHE (contrast enhancement specific to fundus images), GridDistortion, and ElasticTransform; mimics real-world camera and patient variability in fundus photography.
5. **Black background cropping** — retinal fundus images have black borders from the camera aperture; cropping before resizing preserves the actual retinal region at higher effective resolution rather than wasting pixels on uninformative borders.

**How to Reuse:**
- Two-round pseudo-labeling is broadly applicable to any medical imaging task with small labeled sets and large unlabeled test sets; the key is only using high-confidence predictions (e.g., top/bottom decile of predicted probability) in each round.
- Always search for external datasets from prior years of the same competition or related tasks; domain-specific pretraining almost always beats ImageNet initialization for specialized medical imaging.
- CLAHE augmentation is specifically valuable for any fundus/retinal imaging task; for other medical imaging modalities, use modality-appropriate augmentations (window/level for CT, bias field for MRI).

---

## 23. Google QUEST Q&A Labeling (2019) — 3rd Place

**Task type:** NLP multi-label regression — predict 30 subjective quality labels for question-answer pairs from StackExchange; metric: mean Spearman correlation
**Discussion:** https://www.kaggle.com/c/google-quest-challenge/discussion/129927

**Approach:** Ensemble of 12 models including BERT (base & large, cased & uncased), ALBERT, RoBERTa, GPT-2, XLNet, and an LSTM model with Universal Sentence Encoder. Text truncation was strategic: allocating more tokens to the answer than the question (answers are longer and more content-rich for quality prediction). Training: Min-Max target scaling, weighted loss for rare labels, gelu_new activation, cosine warmup, EMA. Post-processing: golden section search-based clipping of predictions per label. Final: 3rd place Spearman correlation.

**Key Techniques:**
1. **Strategic head-tail token truncation** — when text exceeds max length, taking both the first and last N/2 tokens (head-tail) rather than just truncating from the end; preserves both the beginning and conclusion of long answers, which are most informative for quality judgments.
2. **Longer token budget for answers** — allocating more of the max_length budget to answers than to questions in Q&A modeling; domain-informed truncation that outperforms symmetric splitting.
3. **Rare label weighting** — several of the 30 quality labels have very low variance (most questions score similarly); upweighting these labels in the loss function prevents the model from ignoring them due to low gradient signal.
4. **Exponential Moving Average (EMA) of weights** — maintaining an EMA of model weights during training (decay≈0.999) and using the EMA weights for evaluation/inference; consistently improves final model quality at no additional training cost.
5. **Golden section search for post-processing clipping** — many labels are bounded (e.g., 0 or 1); golden section search finds optimal clipping thresholds per label to maximize Spearman correlation; more precise than grid search on the same objective.

**How to Reuse:**
- Head-tail truncation is a simple, effective technique for any long-document NLP task; implement it as the default when documents exceed max_length.
- EMA weight averaging is nearly free and consistently helpful; add it to any deep learning training loop as a default practice.
- When optimizing a ranking metric (Spearman, Kendall) with bounded predictions, post-processing with golden section search or Optuna per-label threshold tuning adds meaningful gains.