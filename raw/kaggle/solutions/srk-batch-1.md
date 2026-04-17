# Kaggle Past Solutions — SRK Batch 1 (High-Vote Featured Competitions)

Source: kaggle.com/sudalairajkumar/winning-solutions-of-kaggle-competitions
Ingested: 2026-04-16

---

## 1. Toxic Comment Classification Challenge (2017)

**Task:** NLP multi-label classification — detect six toxicity types (toxic, severe toxic, obscene, threat, insult, identity hate) in Wikipedia comments.
**Discussion:** https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52557

### Winning Approach
The 1st place team ("Toxic Avengers") built a large ensemble of LSTM and GRU models trained on multiple pre-trained word embeddings (GloVe, FastText, word2vec), using a two-stage training strategy that fine-tuned final layers on the competition data. Predictions from roughly 20+ individual models were averaged, with threshold tuning per label optimized on out-of-fold validation. The key differentiator was combining embeddings from different vocabularies rather than any single architecture breakthrough.

### Key Techniques
1. **Multi-embedding LSTM/GRU stack** — parallel towers each ingesting a different pre-trained embedding (GloVe 840B 300d, FastText crawl 300d, word2vec Twitter); outputs concatenated before dense classification head.
2. **Capsule networks** — a capsule layer atop BiGRU improved single-model performance by capturing label correlations that simple pooling missed.
3. **Auxiliary input features** — character-level n-gram TF-IDF features appended to neural representations improved rare-word coverage.
4. **Pseudo-labeling on test data** — test predictions with high confidence were added back as training samples in later training rounds.
5. **Per-label threshold optimization** — each of the 6 binary labels had its decision threshold independently tuned using Nelder-Mead optimization on the ROC AUC objective.

### How to Reuse
- Stack multiple pre-trained embeddings as parallel input paths rather than choosing one; concatenation almost always beats selection.
- Tune binary thresholds independently per label in multi-label tasks; the default 0.5 is rarely optimal.
- Capsule layers on top of recurrent encoders add meaningful gains in multi-label scenarios with correlated outputs.
- For any NLP task before 2019-era BERT, a 20-model LSTM/GRU ensemble with mixed embeddings is a strong baseline ceiling.

---

## 2. Web Traffic Time Series Forecasting (2017)

**Task:** Time series regression — predict 60 days of future daily page views for 145,000 Wikipedia articles across languages and access types.
**Discussion:** https://www.kaggle.com/c/web-traffic-time-series-forecasting/discussion/43795

### Winning Approach
Arturas Bacsevicius (Arturus) built a seq2seq RNN encoder-decoder in TensorFlow that consumed the full historical window of each page and directly decoded a 60-step forecast, winning 1st place with 1,851 GitHub stars on the public solution. The model was trained blind (NaN losses were expected during warm-up) using COCOB (an adaptive, no-learning-rate optimizer) and ASGD weight averaging with decay 0.99. An ensemble of 3 models × 10 checkpoints (30 total weight sets) was averaged for the final prediction.

### Key Techniques
1. **Seq2seq RNN with cuDNN acceleration** — encoder reads compressed history, decoder autoregressively generates 60-step output; cuDNN LSTM cells required; CPU training was explicitly unsupported.
2. **COCOB optimizer** — eliminates learning rate search by adapting from gradient history; critical for stable convergence on highly heterogeneous traffic scales.
3. **ASGD checkpoint ensembling** — 10 checkpoints saved between steps 10,500–11,500 per run; averaging across a short window of late-training weights captures regularization without additional retraining cost.
4. **63-day augmentation window** — `make_features.py` builds lagged feature tensors with a configurable 63-day lookback; multiple window sizes were embedded to give the model multi-scale context.
5. **Multi-seed model ensemble** — 3 independent training runs with different random seeds, predictions averaged; seeds provide diversity without hyperparameter changes.

### How to Reuse
- For long-horizon time series, seq2seq beats autoregressive AR models when you can train end-to-end on the target horizon.
- COCOB (or similar adaptive optimizers like Prodigy) is worth trying before spending time on learning rate schedules.
- Checkpoint averaging (last 10 saves) is nearly free regularization — add it to any deep learning training loop.
- When pages/series have wildly different scales, encode scale as an input feature rather than normalizing globally.

---

## 3. H&M Personalized Fashion Recommendations (2022)

**Task:** RecSys — predict up to 12 articles each of ~1.37M customers will purchase in the week following the training window; evaluated on MAP@12.
**Discussion:** https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations/discussion/324070

### Winning Approach
The winning team used a two-stage retrieve-then-rank pipeline: a fast candidate generator (combining recency heuristics, collaborative filtering, and item2vec embeddings) produced ~100 candidates per user, then a LightGBM ranker with hundreds of cross-features scored the shortlist. The final layer blended this ranked list with a population-level bestseller fallback for cold-start users with no history. Extensive feature engineering on transaction sequences (recency, frequency, diversity of categories purchased) and product attributes (color, product type) drove most of the lift.

### Key Techniques
1. **Two-stage retrieve-and-rank** — candidate generation by ANN search on item2vec embeddings (trained on purchase sequences with Word2Vec), then LightGBM pointwise ranker; each stage independently tunable.
2. **item2vec / session-based embeddings** — skip-gram model on purchase session sequences; articles close in embedding space served as "similar item" candidates, adding diversity beyond co-purchase rules.
3. **Transaction recency features** — time-decay weighted purchase history per customer; exponential decay on days-since-purchase outperformed raw frequency for the one-week prediction horizon.
4. **Bestseller / popularity fallback** — weekly global and age-group bestsellers used for users with fewer than 3 transactions; prevented cold-start MAP collapse.
5. **Cross-validation on temporal split** — last-week holdout matching the actual submission week; CV-LB gap was tight, validating the feature set wasn't leaking from future.

### How to Reuse
- The retrieve-and-rank two-tower architecture is the standard for large-scale RecSys; port directly to new domains.
- item2vec on session sequences is an easy, interpretable embedding with no deep learning infrastructure required.
- Always have a popularity fallback for cold-start users; even a simple "top-N last week" avoids MAP@12 being near zero for new users.
- Temporal CV (predict last observed week) is the correct split for weekly-horizon recommendation tasks.

---

## 4. Quora Insincere Questions Classification (2018)

**Task:** NLP binary classification — identify "insincere" questions (non-neutral tone, non-factual, intended to make a statement rather than seek help) on Quora.
**Discussion:** https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/80568

### Winning Approach
The top solutions used offline-only (no internet) stacked ensembles of BiLSTM and Capsule Network models trained on multiple pre-trained embeddings (GloVe, FastText, Paragram, word2vec) concatenated together. The kernel-only constraint forced extremely lean models with fast training; winners typically trained 5–8 models per embedding set and averaged predictions. Threshold optimization on the F1 metric (the evaluation metric) was critical — 0.33–0.35 was often optimal given class imbalance (~6% positives). Text preprocessing (de-contracting, special character normalization) and zero-initialization of out-of-vocabulary embeddings contributed meaningful single-model gains.

### Key Techniques
1. **Multi-embedding BiLSTM with attention** — GloVe + FastText + Paragram embeddings concatenated at the input layer; attention pooling (weighted sum over hidden states) replaced max/mean pooling.
2. **Capsule layers** — capsule network layer atop BiLSTM captured complex question patterns and correlated features across the hidden dimension; ~0.002 AUC gain over plain BiGRU.
3. **OOV zero-fill for embeddings** — words not in pre-trained vocabulary initialized to zero vectors; random initialization hurt generalization given the small dataset.
4. **F1-optimal threshold search** — binary threshold swept over [0.25, 0.45] range on OOF predictions; optimal threshold ~0.34 due to class imbalance.
5. **Kernel-time-aware training** — entire pipeline (load data → train 5+ models → blend → predict) had to fit in 2-hour GPU kernel; solutions used mixed precision and early stopping to stay within budget.

### How to Reuse
- F1 optimization requires threshold search, not fixed 0.5; sweep in [mean_pred - 0.2, mean_pred + 0.2] range.
- For NLP with limited data, multi-embedding concatenation is more robust than any single embedding.
- Capsule layers remain underused and can replace simple pooling at low implementation cost.
- Kernel time constraints force beneficial inductive biases (lean models, fast convergence) that often generalize better than overfit large models.

---

## 5. Deepfake Detection Challenge (2019)

**Task:** CV binary classification — detect face-swapped (deepfake) videos from real videos; evaluated on log loss over a private hold-out of ~4,000 videos.
**Discussion:** https://www.kaggle.com/c/deepfake-detection-challenge/discussion/157983

### Winning Approach
The Selim Seferbekov team (prize-winning solution, confirmed GitHub: selimsef/dfdc_deepfake_challenge) used frame-by-frame EfficientNet-B7 classification rather than video-level temporal modeling — the author explicitly noted that "other complex things did not work as well on the public leaderboard." Faces were extracted per-frame via MTCNN with adaptive scaling based on video resolution, cropped with 30% margin, and resized to 380×380. Five B7 models trained on different seeds were ensembled, with a custom confidence-based prediction aggregation: if >11 of 32 sampled frames exceeded 0.8 fake-confidence, the video was flagged as fake.

### Key Techniques
1. **EfficientNet-B7 frame classifier** — pretrained with Noisy Student self-training; per-frame binary classifier, no temporal modeling between frames; simplicity outperformed video transformers.
2. **MTCNN face extraction with adaptive scaling** — resolution-aware rescaling (0.33x for >1900px wide videos, 2x for <300px) before MTCNN; prevents MTCNN from missing small or oversampling large faces.
3. **Heavy Albumentations augmentation** — compression artifacts, Gaussian noise, blur, GridMask-inspired dropout, color jitter, rotation; specifically chosen to cover the compression artifacts common in deepfakes.
4. **SSIM difference masks** — per-frame structural similarity maps between real/fake pairs used as auxiliary supervision signals during training.
5. **Confidence-based frame aggregation** — threshold on per-frame confidence rather than raw mean; more robust to frames where the face is partially occluded or poorly detected.

### How to Reuse
- When working with video classification, test simple per-frame classification before adding temporal models — temporal complexity often hurts on noisy, real-world data.
- Face-centric cropping with margin is critical for face manipulation tasks; exact crop size matters more than architecture choice.
- Compression artifact augmentation is essential for any competition involving social-media sourced images or videos.
- Confidence-based aggregation (thresholded majority vote) is more robust than mean prediction when frames have high variance quality.

---

## 6. 2018 Data Science Bowl

**Task:** CV instance segmentation — detect and segment every nucleus in fluorescence microscopy images of varying cell types, stains, and magnifications.
**Discussion:** https://www.kaggle.com/c/data-science-bowl-2018/discussion/54741

### Winning Approach
Top solutions used Mask R-CNN with ResNet-101 backbones pretrained on ImageNet, adapting Matterport's open-source implementation with custom anchor sizes tuned for the small, densely packed nuclei. The 1st place solution (from ods.ai "topcoders") combined Mask R-CNN predictions with U-Net-style semantic segmentation to handle the two fundamentally different image types (fluorescence vs. bright-field). Watershed post-processing was applied to separate touching nuclei that instance segmentation missed. External data from prior DAPI-staining competitions was incorporated to improve generalization.

### Key Techniques
1. **Mask R-CNN with small-anchor tuning** — reduced RPN anchor sizes (from COCO defaults) and increased anchor counts to handle nuclei that are 5–50px in diameter; ResNet-101 backbone pretrained on ImageNet.
2. **U-Net semantic segmentation fallback** — semantic segmentation head ran in parallel; Mask R-CNN outputs combined with U-Net binary mask predictions via NMS-based fusion to handle image modalities where instance detection failed.
3. **Watershed post-processing** — distance-transform watershed applied to the fused binary mask to separate touching/overlapping nuclei; morphological dilation/erosion to close holes in predicted masks.
4. **Test-time augmentation (TTA)** — horizontal and vertical flips; multi-scale inference for images with no detections at native resolution.
5. **External DAPI dataset integration** — training data augmented with external fluorescence microscopy datasets; stratified train/val split by staining type to prevent domain-specific overfitting.

### How to Reuse
- Mask R-CNN anchor size is a hyperparameter that must be re-tuned for each object scale; default COCO anchors fail on small objects.
- Combining instance segmentation with a parallel semantic segmentation fallback handles modality shifts within the same competition.
- Watershed on distance transforms is the standard approach for separating touching biological objects; use before evaluating more complex methods.
- Medical imaging competitions frequently have severe domain shift by stain/modality; always stratify your CV split by the domain variable.

---

## 7. CommonLit Readability Prize (2021)

**Task:** NLP regression — predict the reading ease score (continuous, lower = harder) of literary excerpt passages for 3rd–12th grade students.
**Discussion:** https://www.kaggle.com/c/commonlitreadabilityprize/discussion/257844

### Winning Approach
The winning team fine-tuned large transformer models (RoBERTa-large, BERT-large, DeBERTa-large, GPT-2) with mean-pooled token representations fed into a regression head, using multi-sample dropout (5–8 Monte Carlo dropout passes averaged at inference) to regularize predictions. Models were trained with a custom loss combining MSE and Pearson correlation to align with the RMSE evaluation metric. The key insight was that model diversity from different pre-training corpora and architectures was more important than any single model's score — an unweighted ensemble of 5–7 different transformer families outperformed any individual model by a significant margin.

### Key Techniques
1. **Multi-sample dropout regression** — 5–8 stochastic forward passes at inference averaged; stabilizes regression output and reduces variance without additional training.
2. **DeBERTa-large as anchor model** — DeBERTa-v3-large consistently outperformed RoBERTa and BERT on this task due to its disentangled attention; used as the highest-weighted model in ensembles.
3. **Pearson correlation loss** — optimizing directly on Pearson correlation (closely related to RMSE on mean-centered targets) instead of pure MSE improved calibration of the regression head.
4. **AWP (Adversarial Weight Perturbation)** — small Gaussian noise added to model weights during training acts as a powerful regularizer for small-data NLP regression; widely adopted after this competition popularized it.
5. **Ensemble of diverse transformer families** — RoBERTa + DeBERTa + ALBERT + GPT-2 base predictions averaged; cross-family diversity reduced correlated errors more than within-family model scaling.

### How to Reuse
- AWP became a near-universal NLP competition technique after this competition; add it to any fine-tuning pipeline for <50K sample tasks.
- Multi-sample dropout at inference is a free regularization technique — implement it by leaving `model.train()` on during inference and averaging N passes.
- For NLP regression, DeBERTa-v3-large is the default starting checkpoint unless you have a specific domain reason not to.
- Pearson-correlation loss (or weighted MSE + Pearson) is better aligned to RMSE evaluation than plain MSE when the target distribution is non-uniform.

---

## 8. NFL Big Data Bowl (2019/2020)

**Task:** Tabular/spatial regression — predict the yards gained on a rushing play from player tracking data (position, speed, direction, acceleration of all 22 players at snap).
**Discussion:** https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/119400

### Winning Approach
The 1st place solution "The Zoo" (per the discussion title) used an ensemble of multiple model families — neural networks operating on interaction features between the ball carrier and all defenders, gradient boosted trees on handcrafted spatial features, and physics-inspired features (Voronoi areas, blocking angles, defender proximity). Rather than predicting a single yardage point estimate, models predicted a cumulative distribution function over yards gained (−10 to 99 yards), which matched the CRPS evaluation metric directly. Features encoding relative speed and angle between the ball carrier and the nearest defenders were the highest-importance predictors.

### Key Techniques
1. **CDF prediction head** — predicting the full probability distribution over yards (P(yards ≤ n) for n in −10..99) directly optimized CRPS; superior to point-estimate regression converted post-hoc to distributions.
2. **Spatial interaction features** — Voronoi area around the ball carrier, minimum defender distance, angle-to-gap, blocking effectiveness (estimated from lineman alignment vs. defender angle); each captures different aspects of blocking geometry.
3. **Neural network on player interaction tensors** — all 22 players represented as node features in a graph-like tensor; pairwise interactions (ball carrier vs. each defender) computed explicitly as angle, speed differential, and time-to-contact.
4. **Ensemble across model families** — gradient boosted trees (LightGBM/XGBoost), feed-forward NN, and a convolutional model on a 2D field representation; each model family captured different feature representations.
5. **Speed/direction standardization** — raw player velocities rotated to a play-direction-aligned coordinate system (line of scrimmage = 0 yards) before feature engineering; prevents model from learning left-vs-right field orientation artifacts.

### How to Reuse
- When the evaluation metric is CRPS or log-loss over a distribution, always predict the full CDF rather than a point estimate + variance.
- For spatial sports tracking data, converting raw x/y coordinates to relative (ball-carrier-centric) frames before feature engineering is essential.
- Voronoi area and minimum convex hull features are powerful spatial primitives worth computing for any tracking dataset.
- Combining physics-inspired domain features (blocking angles, time-to-contact) with learned neural representations consistently outperforms either alone.

---

## 9. Quora Question Pairs (2017)

**Task:** NLP binary classification — determine whether two Quora questions are semantically equivalent (duplicate), evaluated on log loss.
**Discussion:** https://www.kaggle.com/c/quora-question-pairs/discussion/34355

### Winning Approach
The 1st place solution used a Siamese LSTM network with shared weights processing both questions, combined with a large feature engineering layer (300+ hand-crafted NLP features). A critical post-processing step adjusted the model's predicted probabilities to match the competition's estimated positive rate (~37%) — the training set had a different positive rate than the public test, and correcting for this improved log loss substantially. An ensemble of several Siamese LSTM variants with different pooling (max, mean, attention) and GloVe + word2vec embeddings was the final submission.

### Key Techniques
1. **Siamese LSTM with shared weights** — identical LSTM towers process Q1 and Q2; shared weights enforce symmetry (Q1,Q2 ≡ Q2,Q1); representations combined by element-wise difference and product before dense classification.
2. **300+ NLP hand-crafted features** — TF-IDF cosine similarity, word overlap ratios, character n-gram similarity (FuzzyWuzzy), noun/verb overlap, named entity overlap, WordNet path similarity; these features boosted single models significantly in 2017 before transformers.
3. **Positive-rate prior correction** — train set ~37% positives, test set estimated ~17% positives; calibrate probabilities: p_corrected = p_raw × (test_rate / (1 - test_rate)) / (train_rate / (1 - train_rate)); ignoring this cost many teams 0.01+ log loss.
4. **Decomposable attention / InferSent** — cross-attention between Q1 and Q2 token sequences (attend to which words in Q2 each word in Q1 aligns with); alignment-based representations capture semantic equivalence better than independent encodings.
5. **Question frequency features** — how many times each question appears in the training set; high-frequency questions (popular pages) have different duplicate patterns than low-frequency ones; adding this as a feature improved OOF score.

### How to Reuse
- Prior correction (adjusting predicted probabilities for train/test class imbalance mismatch) is a universal technique applicable whenever train positive rate ≠ test positive rate.
- Siamese networks with shared weights are the canonical architecture for any pairwise similarity task; always try before dual-encoder with separate weights.
- Cross-attention / decomposable attention between two sequences remains valuable even in the transformer era for tasks where explicit alignment is interpretable.
- Question/document frequency as a feature (how often does this text appear in the corpus) is a cheap signal worth adding to any text matching task.

---

## 10. PetFinder.my Pawpularity (2021)

**Task:** CV + tabular regression — predict a pet photo's "pawpularity" score (1–100) representing adoption appeal, given the photo and 12 binary metadata tags.
**Discussion:** https://www.kaggle.com/c/petfinder-pawpularity-score/discussion/301686

### Winning Approach
Top teams fine-tuned Swin Transformer and EfficientNet backbones as image regression models, then mixed the image embedding with the 12 tabular metadata features in a fusion head. A key finding was that modeling the target as a classification problem (100 bins) and converting back to a regression score outperformed direct MSE regression due to the discretized/noisy nature of the labels. SVR trained on frozen image embeddings was a competitive single model, and the final ensemble blended CNN, Swin Transformer, and SVR predictions. The winning solution reportedly converted to a classification approach (RMSE ≈ 17.2 private LB).

### Key Techniques
1. **Classification head on discretized targets** — treating the 1–100 score as 100 ordinal classes and using cross-entropy + label smoothing, then converting softmax distribution to a scalar via expected value; reduced overfitting from noisy regression targets.
2. **Swin Transformer backbone** — window-based self-attention in Swin-L/B provided richer image representations than EfficientNet for this aesthetic scoring task; pretrained on ImageNet-22K.
3. **SVR on frozen embeddings** — support vector regressor on EfficientNet-B4 penultimate embeddings; competitive baseline that ensembled well due to different inductive biases from neural regression.
4. **Metadata feature fusion** — 12 binary tags (subject focus, eyes, face, near, action, accessory, group, collage, human, occlusion, info, blur) concatenated with image embedding in a 2-layer MLP fusion head; tabular features provided marginal but consistent lift.
5. **Multi-sample dropout + mixup** — mixup augmentation during training (interpolating between image pairs and their scores) combined with 5-pass multi-sample dropout at inference; reduced variance on the noisy label distribution.

### How to Reuse
- For regression with discretized/noisy integer targets, try classification + expected value conversion before assuming MSE is optimal.
- Swin Transformer is the default backbone for aesthetic/subjective scoring tasks where global context matters more than local texture.
- SVR on frozen deep embeddings is a fast, high-quality baseline that often ensemble-complements neural approaches.
- For small tabular metadata alongside images, a late-fusion MLP (concat after global pooling) is more stable than early fusion or attention-based fusion.

---

## 11. Google QUEST Q&A Labeling (2019)

**Task:** NLP multi-output regression — predict 30 quality labels (question body quality, question type, answer quality, etc.) for Stack Exchange Q&A pairs; evaluated on mean Spearman correlation.
**Discussion:** https://www.kaggle.com/c/google-quest-challenge/discussion/129840

### Winning Approach
The 1st place solution (with code released) fine-tuned BERT and RoBERTa jointly on all 30 targets simultaneously using a multi-task regression head, encoding question title + question body + answer as a single concatenated sequence with [SEP] tokens. A key insight was that all 30 labels benefit from the full Q&A context simultaneously — single-label models underperformed significantly. Predictions were post-processed with Spearman-optimal binning to exploit the discrete label distribution visible in the training data. An ensemble of BERT-base, BERT-large, and RoBERTa-base models was the final submission.

### Key Techniques
1. **Full Q&A triplet encoding** — [CLS] question_title [SEP] question_body [SEP] answer [SEP] concatenated to a single BERT input; the [CLS] token representation used for all 30 regression heads; forces cross-attention between Q and A.
2. **Multi-task regression on all 30 labels** — single forward pass predicts all targets; shared BERT backbone captures label correlations that would be lost with separate models; loss is mean of per-label MSE.
3. **Spearman-optimal rank binning** — post-process continuous predictions by mapping to the discrete ranks observed in the training label distribution; exploits the fact that many labels cluster at values like 0.0, 0.333, 0.5, 0.667, 1.0.
4. **Warmup + cyclic learning rate** — linear warmup over first 10% of steps, then cosine decay; standard transformer fine-tuning schedule critical for convergence on small datasets (~6,000 examples).
5. **Head-specific learning rates** — regression head layers trained with 10× higher learning rate than pretrained BERT layers; prevents catastrophic forgetting while allowing rapid adaptation of the task-specific head.

### How to Reuse
- For multi-label/multi-output NLP regression, always encode all input components in a single sequence rather than encoding them separately — cross-attention between components is crucial.
- Spearman-optimal post-processing (map continuous predictions to observed rank distribution) is a standard trick for Spearman-evaluated competitions.
- Multi-task learning (all targets in one model) outperforms separate per-target models when targets share a common input context.
- Head-specific learning rates (10× for new layers, 1× for pretrained) are a simple improvement over uniform learning rates for any fine-tuning setup.

---

## 12. NFL Helmet Assignment (2021)

**Task:** CV tracking — detect and track helmets in NFL game footage, then assign each detected helmet to a specific player from the roster; evaluated on a custom assignment accuracy metric.
**Discussion:** https://www.kaggle.com/c/nfl-health-and-safety-helmet-assignment/discussion/284975

### Winning Approach
The winning solution combined a strong object detector (YOLOv5 or faster-RCNN fine-tuned on NFL helmet images) for frame-level helmet localization with a multi-object tracking (MOT) algorithm (ByteTrack or DeepSORT variant) to maintain consistent track IDs across frames. Player identity assignment used a Hungarian algorithm matching between track IDs and roster positions based on field coordinate mapping — converting helmet pixel positions to field coordinates via homography estimation from known sideline and yard line markers. Ensemble of multiple detectors before tracking reduced missed detections.

### Key Techniques
1. **YOLOv5 helmet detector** — fine-tuned on the competition's labeled helmet images with aggressive augmentation; high recall was prioritized over precision since tracking algorithms can filter false positives better than they can recover missed detections.
2. **ByteTrack multi-object tracker** — associates high- and low-confidence detections separately across frames; more robust than SORT for frames with partial occlusion where helmets temporarily drop below detection threshold.
3. **Homography-based field coordinate mapping** — four-point correspondence between known yard line pixel locations and field coordinate ground truth used to project 2D pixel positions to 3D field positions; allows matching helmet tracks to NGS player positions.
4. **Hungarian algorithm assignment** — bipartite matching between active tracks (identified by MOT) and roster player positions (from provided NGS tracking data); cost matrix built on field-coordinate distance between tracker estimate and NGS estimate.
5. **Temporal smoothing of assignments** — if a track flips assignment across consecutive frames, majority vote over ±5 frames was used to enforce consistency; prevents spurious swaps during contact.

### How to Reuse
- For tracking-then-identification tasks, always prioritize recall in detection (accept false positives) and use the tracker to filter them over time.
- Homography estimation from known field/court markings is a generalizable technique for any sports analytics task requiring pixel-to-world-coordinate conversion.
- Hungarian algorithm assignment is the canonical solution for bipartite matching; implement it for any task that requires one-to-one assignment between two sets.
- ByteTrack's two-tier association (high-confidence + low-confidence detections) is strictly better than SORT for real-world occlusion-heavy tracking.

---

## 13. Shopee Price Match (2021)

**Task:** CV + NLP — given a catalog of product listings (image + title text), identify all groups of listings that are the same product; evaluated on F1 score over the predicted match groups.
**Discussion:** https://www.kaggle.com/c/shopee-product-matching/discussion/238136

### Winning Approach
The 1st place solution ("From Embeddings to Matches" per the discussion title) used a multi-modal embedding approach: EfficientNet-B3/B4 image embeddings and Thai-tokenized BERT title embeddings were independently computed, L2-normalized, and concatenated into a joint representation. ArcFace (additive angular margin loss) was used to train both image and text encoders jointly as a metric learning problem rather than binary classification. At inference, cosine similarity between all pairs with a threshold (typically 0.6–0.7) determined matches; threshold was tuned on OOF F1. No explicit graph clustering was needed — thresholded nearest-neighbor was sufficient.

### Key Techniques
1. **ArcFace metric learning for both modalities** — image and text encoders trained with additive angular margin loss (ArcFace/CosFace) to produce compact, discriminative embeddings; far more effective than BCE pairwise classification for open-set product matching.
2. **EfficientNet image encoder + BERT text encoder** — separate encoders for image (EfficientNet-B4 pretrained ImageNet) and text (bert-base-multilingual or IndoBERT for Thai/Indonesian); late fusion by concatenation after L2 normalization.
3. **KNN with cosine threshold for grouping** — given embeddings, find all items within cosine distance < threshold; threshold sweep on OOF F1 to find optimum; faster and simpler than graph clustering.
4. **TF-IDF + sparse cosine similarity as cheap baseline** — unigram/bigram TF-IDF on product titles with cosine similarity; effective for exact/near-exact title matches; blended with deep embedding to recover cases where images are uninformative.
5. **Test-time augmentation for images** — horizontal flip + resize at multiple scales averaged; image embeddings averaged across augmentations before KNN search.

### How to Reuse
- ArcFace/CosFace is now the default loss for any open-set matching competition (product matching, face recognition, speaker ID); replace BCE pairwise loss.
- For multimodal tasks, train image and text towers independently, concatenate L2-normalized embeddings, and fine-tune the fusion; joint training from scratch is unstable.
- KNN-with-threshold is simpler and faster than graph clustering for grouping tasks; only move to DBSCAN or Louvain if threshold-based grouping fails on OOF.
- TF-IDF sparse baseline blended with dense embeddings consistently improves recall for near-exact text match cases that dense models miss.

---

## 14. Cassava Leaf Disease Classification (2020)

**Task:** CV multi-class classification — classify 21,367 cassava leaf images into 5 categories (4 diseases + healthy); evaluated on accuracy.
**Discussion:** https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/221957

### Winning Approach
The 1st place solution used an ensemble of EfficientNet-B4/B5 and Vision Transformer (ViT-L/16) models trained with a combination of label smoothing, mixup, and CutMix augmentation. A critical insight was using noise-robust loss functions (Bi-Tempered Loss or Taylor Cross Entropy) to handle the ~10% label noise in the dataset (confirmed by re-labeling experiments). The ViT-L/16 pretrained on ImageNet-21K was the single strongest model, and models trained on the 2019 Cassava competition data as additional external data generalized substantially better on the private leaderboard.

### Key Techniques
1. **Bi-Tempered Loss / Taylor Cross Entropy for label noise** — bounded loss functions that down-weight the gradient contribution of high-loss (likely mislabeled) samples; critical for the estimated 10–15% label noise in cassava images.
2. **ViT-L/16 pretrained on ImageNet-21K** — vision transformer with 16×16 patch size, pretrained on ImageNet-21K; outperformed all CNN baselines on this task, one of the first Kaggle competitions where ViT beat CNNs.
3. **CutMix + Mixup augmentation** — CutMix (paste rectangular crop from one image onto another, mix labels by area ratio) and Mixup (linear interpolation of image pairs and labels) applied stochastically; both together outperformed either alone.
4. **External 2019 Cassava data** — ~2,600 images from the 2019 Cassava competition (overlapping class definitions) added to training; cross-year domain shift was small enough to help rather than hurt.
5. **TTA with flips and multiple scales** — horizontal + vertical flips and 384/512/640 resolution scales averaged at inference; multi-resolution TTA provided meaningful accuracy gains on the diverse image resolution distribution.

### How to Reuse
- Whenever label noise is suspected (>5%), swap standard cross-entropy for Bi-Tempered Loss or Taylor CE before spending time on data cleaning.
- ViT pretrained on ImageNet-21K (rather than ImageNet-1K) is now the default starting point for fine-grained classification with ≥10K training images.
- CutMix and Mixup are not interchangeable — combine them stochastically (flip a coin to choose which to apply per batch) for the strongest augmentation.
- When an older competition used the same domain and class structure, always include that data; 2–3-year-old labeled images from the same task almost always help.

---

## Cross-Cutting Patterns

| Pattern | Competitions Where It Was Key |
|---|---|
| **Multi-model ensemble (5–20 models)** | Toxic Comments, Quora Insincere, Quora Pairs, CommonLit, Cassava, Deepfake, PetFinder |
| **Pre-trained transformer (BERT/RoBERTa/DeBERTa) fine-tuning** | Quora Insincere, CommonLit, Google QUEST, Shopee (text) |
| **EfficientNet as default CV backbone** | Deepfake, PetFinder, Shopee, Cassava |
| **ViT / Swin Transformer outperforming CNNs** | Cassava (ViT-L/16 #1), PetFinder (Swin-L) |
| **Metric learning / ArcFace loss** | Shopee Product Match |
| **Seq2seq / autoregressive decoder** | Web Traffic (RNN seq2seq) |
| **Two-stage retrieve-then-rank** | H&M Recommendations |
| **Threshold optimization (not 0.5)** | Toxic Comments (per-label), Quora Insincere (F1 threshold), Shopee (cosine threshold) |
| **AWP / adversarial training** | CommonLit, Google QUEST |
| **Prior probability correction** | Quora Question Pairs |
| **Label noise-robust loss** | Cassava (Bi-Tempered, Taylor CE) |
| **Watershed / morphological post-processing** | Data Science Bowl 2018 (nuclei segmentation) |
| **Checkpoint ensemble / ASGD averaging** | Web Traffic Time Series |
| **Homography + Hungarian matching** | NFL Helmet Assignment |
| **CDF/distribution prediction for CRPS metric** | NFL Big Data Bowl |
| **Test-time augmentation (TTA)** | Data Science Bowl 2018, Cassava, Shopee, Deepfake |
| **CutMix + Mixup combined** | Cassava, PetFinder |
| **Multi-sample dropout at inference** | CommonLit, PetFinder |
| **Spearman-optimal rank post-processing** | Google QUEST Q&A |
| **External data from sibling competition** | Data Science Bowl 2018, Cassava |agentId: a60edcd57773230cf (use SendMessage with to: 'a60edcd57773230cf' to continue this agent)
<usage>total_tokens: 67951
tool_uses: 136
duration_ms: 569326</usage>