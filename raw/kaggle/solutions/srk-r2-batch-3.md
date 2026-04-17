# Kaggle Past Solutions — SRK Round 2, Batch 3

Source: kaggle.com/sudalairajkumar/winning-solutions-of-kaggle-competitions
Ingested: 2026-04-17
Note: These are non-1st-place solutions (2nd, 3rd, top-10) with 100+ upvotes. Writeup content sourced from the thedrcat/kaggle-winning-solutions-methods dataset (licensed CC-BY-NC-SA-4.0); for 3 entries not covered by that dataset, details were synthesized from web research and GitHub repositories.

---

## 1. TensorFlow - Help Protect the Great Barrier Reef (2022) — 5th Place

**Task type:** Underwater object detection (crown-of-thorns starfish in video frames)
**Discussion:** [https://www.kaggle.com/c/tensorflow-great-barrier-reef/discussion/308007](https://www.kaggle.com/c/tensorflow-great-barrier-reef/discussion/308007)

**Approach:** CenterNet detector built on a DeepLabV3+ architecture with EfficientNetV2 backbone, trained at 1280×720 and inferred at 1792×1008 (1.4× upscale) with two-resolution blending.

**Key Techniques:**
1. **CenterNet on DeepLabV3+** — Repurposed a semantic segmentation architecture (DeepLabV3+) as a detection backbone for CenterNet by adding a regression head, giving a stronger feature pyramid than vanilla detection backbones.
2. **EfficientNetV2 backbone scaling** — Systematically swapped backbone from B0 through XL; larger backbones consistently improved score.
3. **Heatmap resolution trick** — Output heatmap set to 1/8 of input resolution (not the standard 1/4 or 1/16), improving localization precision.
4. **Multi-resolution inference blending** — Training at 1280×720, then inferring at both 1.4× and 1.6× scale and averaging predictions gave a meaningful boost (+0.01 LB).
5. **Avoiding overfitting strategy** — Author noted that an "overfitting strategy" (pseudo-labeling on test video sequences) failed; clean single-model training was more reliable on private LB.

**How to Reuse:**
- When adapting segmentation architectures (U-Net, DeepLabV3+) for detection, adding a lightweight CenterNet head can be competitive with purpose-built detectors.
- Test at multiple inference scales and blend outputs — often adds 0.005–0.02 on detection benchmarks with no retraining cost.
- Be cautious with overfitting strategies on video competitions where train/test sequences differ; CV on held-out video clips is more reliable than pseudo-label loops.

---

## 2. DFL - Bundesliga Data Shootout (2022) — 1st Place

**Task type:** Action spotting in soccer video — detect and timestamp events (challenges, play, throwin) from multi-camera broadcast footage
**Discussion:** [https://www.kaggle.com/c/dfl-bundesliga-data-shootout/discussion/359932](https://www.kaggle.com/c/dfl-bundesliga-data-shootout/discussion/359932)

**Approach:** End-to-end 2.5D model — EfficientNet backbone with Temporal Shift Module (TSM), feeding into a 1D UNet head that produces per-frame event probability time-series. Pretrained on SoccerNet ball detection before fine-tuning on the event targets.

**Key Techniques:**
1. **Temporal Shift Module (TSM) on EfficientNet** — Inserts temporal shifts into 2D CNN feature maps to model motion across frames without the cost of full 3D convolutions; critical for stabilizing training when TSM changes model structure vs. plain ImageNet init.
2. **Ball-detection pretraining** — Pretrained the full backbone+TSM on SoccerNet ball tracking (20,000+ manually annotated frames) before event spotting fine-tuning; gave better feature initialization than ImageNet alone for sports video.
3. **1D UNet head on temporal features** — Instead of a simple classification head, a 1D encoder-decoder (UNet) operates on the time axis of extracted frame features; allows the model to leverage context from surrounding frames when scoring each timestamp.
4. **Manifold Mixup** — Applied mixup at the image-feature (manifold) level rather than pixel level; worked well for this time-series-of-frames setup.
5. **Label design as 1D Gaussian + binary eval mask** — Event labels were ±5-frame Gaussian heatmaps on a binary mask indicating the evaluation window; separated foreground/background cleanly and improved convergence.

**How to Reuse:**
- TSM is a drop-in temporal module for any 2D CNN in video tasks — adds minimal parameters while giving multi-frame context; good first experiment before committing to 3D CNNs.
- For action detection/spotting, a 1D UNet on temporal feature sequences outperforms single-frame classifiers and is much cheaper than full video transformers.
- Pretraining on a related auxiliary task (e.g., ball detection before event detection) is highly effective when target labels are sparse.

---

## 3. Shopee - Price Match Guarantee (2021) — 6th Place

**Task type:** Multimodal product matching — given product images and text titles, find all products in the catalog that are the same item
**Discussion:** [https://www.kaggle.com/c/shopee-product-matching/discussion/238010](https://www.kaggle.com/c/shopee-product-matching/discussion/238010)

**Approach:** Separate image and text embedding models (ArcFace for images, DistilBERT for text) whose embeddings are concatenated, then matched via cosine nearest-neighbors. A second-stage XGBoost model is trained on additional graph/meta features derived from matches.

**Key Techniques:**
1. **ArcFace (CurricularFace) for image embeddings** — Metric learning with ArcFace loss trained on product images; produces tightly clustered per-product embeddings far better than softmax classification for retrieval tasks.
2. **Multilingual DistilBERT + ViT/Swin for separate modalities** — English and Indonesian DistilBERT for title embeddings, ViT-384 and Swin Transformer for image embeddings (size 384), then EfficientNet-B4 at 512; separate models for each modality outperformed cross-modal fusion.
3. **Weighted Database Augmentation** — After initial matching, each item's embedding is updated as a weighted mean of itself and its nearest match (weight proportional to cosine distance); acts as regularization by propagating matched-pair information without explicit labels.
4. **Two-stage matching: embedding + XGBoost** — The first stage retrieves top-K candidates via cuML NearestNeighbors; the second stage XGBoost model uses features like cosine distance, text length, Levenshtein distance, image file size, and graph connectivity to re-rank and threshold matches.
5. **GroupKFold for second-stage training** — The second-stage model is validated with GroupKFold on product groups, preventing leakage since items from the same group appear together in train/val.

**How to Reuse:**
- For any entity-matching or deduplication task, combine metric-learning embeddings (ArcFace/TripletLoss) for each modality separately, then fuse via concatenation rather than cross-modal attention — simpler and often just as effective.
- Two-stage retrieval (approximate NN → feature-based re-ranker) is a universal pattern for large-scale matching; cuML's GPU NearestNeighbors is a practical near-drop-in for sklearn at scale.
- Weighted mean embedding updates (using your own model's predictions as soft labels for database augmentation) is a self-supervised regularization trick applicable to any retrieval competition.

---

## 4. RSNA STR Pulmonary Embolism Detection (2020) — 2nd Place

**Task type:** Multi-label classification of pulmonary embolism types from CT pulmonary angiography (CTPA) DICOM sequences
**Discussion:** [https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection/discussion/193401](https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection/discussion/193401)

**Approach:** Five-stage pipeline: (1) 2D CNN slice-level feature extraction, (2) Transformer sequence model for exam-level PE labels, (3) Time-Distributed CNN trained end-to-end, (4) CNN classifier to identify heart slices, (5) 3D CNN for RV/LV ratio prediction from heart slices only, with a linear stacking model combining all outputs.

**Key Techniques:**
1. **2D CNN → Transformer sequence modeling** — Extracted 512-D slice features with ResNeSt50 using PE-specific DICOM windowing (WL=100, WW=700), then fed sequences into a 4-layer Transformer (DistilBERT architecture); enabled learning temporal dependencies across hundreds of CT slices without processing the full 3D volume.
2. **Generalized mean pooling for feature aggregation** — Used generalized mean pooling rather than global average pooling to aggregate spatial features into the 512-D vector; gave better discriminative representations for slice-level PE detection.
3. **RandAugment for medical imaging** — Applied RandAugment data augmentation during 2D CNN pretraining; medical imaging competitions rarely use aggressive augmentation but it helped generalization here.
4. **Hand-labeled heart-slice classifier** — Manually labeled ~1,000 CT scans to train a heart-slice detector (EfficientNet-B1, AUC 0.998), then restricted RV/LV ratio prediction to those slices only; domain-specific preprocessing stages that isolate the relevant anatomy dramatically improve downstream model quality.
5. **Pretrained video model (ip-CSN-101) for 3D RV/LV** — Used a 101-layer channel-separated network pretrained on 65M Instagram videos (VMZ) as a 3D CNN for cardiac structure classification; video pretraining transfers better than random init or ImageNet for 3D medical imaging.

**How to Reuse:**
- The 2D CNN → sequence model pipeline is a standard pattern for any 3D medical imaging task where you cannot fit full volumes; it generalizes to brain MRI, colonoscopy, and pathology slides.
- For multi-label CT competition tasks, build specialized sub-models for each label group (e.g., PE presence vs. cardiac ratio) rather than one model for all labels — dramatically easier to optimize.
- When domain annotation is feasible (a day's work for 1,000 scans), hand-labeled anatomical localization models pay off in competitions with complex multi-target metrics.

---

## 5. HuBMAP - Hacking the Kidney (2021) — 1st Place

**Task type:** Instance segmentation of glomeruli in high-resolution kidney histology whole-slide images
**Discussion:** [https://www.kaggle.com/c/hubmap-kidney-segmentation/discussion/238198](https://www.kaggle.com/c/hubmap-kidney-segmentation/discussion/238198)

**Approach:** Two-class segmentation (healthy vs. non-functional glomeruli separately), using external data carefully re-annotated by hand with both classes. EfficientNet-based UNet ensemble with a Swin Transformer backbone variant, plus careful exclusion of a pathological slide (d488c759a) that destabilized CV-LB correlation.

**Key Techniques:**
1. **Two-class annotation strategy** — The training set only labeled healthy glomeruli; the team identified that the competition actually required distinguishing healthy from non-functional (FC) glomeruli, and re-annotated external data with both classes, making their model fundamentally different from single-class baselines.
2. **Exclusion of pathological slide from LB evaluation** — Slide d488c759a had mixed healthy/non-functional annotations that corrupted LB scores; the team correctly identified it and excluded it when interpreting LB, enabling trust in CV and accurate model selection.
3. **BoT middle layer + Swin Transformer backbone** — Combined Bag-of-Tricks attention modules in the UNet middle with a Swin Transformer backbone and FPN skip connections; gave strong multi-scale feature representation for variable-size glomeruli.
4. **Human+AI guided re-annotation of external data** — Used AI model predictions to guide human annotators in labeling missing glomeruli in external HPA datasets; a hybrid labeling workflow that scales better than pure manual annotation for large WSI data.
5. **Ensemble of two architecturally diverse models** — Final ensemble: EfficientNet-UNet and Swin-FPN; diversity from architecture (CNN vs. Transformer backbone) was more valuable than ensembling multiple same-architecture checkpoints.

**How to Reuse:**
- Always audit whether the training labels match what the test metric actually measures — label schema mismatches (like healthy-only vs. all glomeruli) are a major source of CV-LB gap.
- Identify and handle "poison" slides or outlier samples that corrupt CV-LB correlation before tuning; one bad sample can invalidate hundreds of experiments.
- Hybrid human+AI annotation loops are practical for WSI competitions: train a baseline, use it to find annotation gaps, correct them, retrain — 2–3 loops often outperform simply buying more labels.

---

## 6. RSNA Pneumonia Detection Challenge (2018) — 2nd Place

**Task type:** Bounding-box detection of pneumonia opacities in chest X-rays (3-class: normal, not pneumonia lung opacity, pneumonia)
**Discussion:** [https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/discussion/70427](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/discussion/70427)

**Approach:** Single RetinaNet detector with an SE-ResNeXt101 backbone at 512×512 resolution, trained with multi-task 3-class classification simultaneously with bounding-box regression. 4-fold cross-validation with ImageNet pretraining.

**Key Techniques:**
1. **SE-ResNeXt101 as RetinaNet backbone** — Squeeze-and-excitation networks with ResNeXt grouping outperformed standard ResNet backbones for chest X-ray detection; SE channel attention recalibrates feature maps toward diagnostically relevant regions.
2. **Multi-task 3-class classification + localization** — Added an auxiliary 3-class (normal / non-pneumonia opacity / pneumonia) classification head alongside the RetinaNet detection head; joint training improved both classification and localization.
3. **Extra output head for small bounding boxes** — Added a dedicated head predicting smaller anchor scales; small pneumonia patches were systematically missed by standard RetinaNet anchors.
4. **ReduceLROnPlateau scheduling** — Used patience=4, factor=0.2 LR reduction; more robust than fixed schedules on relatively small medical datasets where plateau timing is unpredictable.
5. **4-fold CV with ImageNet init** — Standard 4-fold cross-validation; ImageNet pretrained weights transferred well to chest X-ray detection despite domain gap.

**How to Reuse:**
- For any single-class medical detection task, add an auxiliary multi-class classification head — it provides the detector with global context (is there pathology at all?) that improves box precision.
- SE-type backbones (SE-ResNeXt, SENet154) consistently outperform plain ResNets on medical imaging detection tasks; use them as default.
- Always include dedicated small-anchor detection heads when the target pathology can appear at multiple scales in medical images.

---

## 7. Cornell Birdcall Identification (2020) — 3rd Place

**Task type:** Multi-label audio classification of bird species from 5-second soundscape recordings (265 species)
**Discussion:** [https://www.kaggle.com/c/birdsong-recognition/discussion/183199](https://www.kaggle.com/c/birdsong-recognition/discussion/183199)

**Approach:** CNN ensemble trained on Mel spectrograms pre-saved to disk for speed, with extensive manual data cleaning and domain-specific audio augmentations. Ensemble of 6 models including a 2.5-second model and a 150-class restricted model.

**Key Techniques:**
1. **Pre-saved Mel spectrograms** — Converted all 20,000 audio files to Mel spectrograms saved as images before training; eliminated real-time audio decoding bottleneck on weak hardware and enabled standard image augmentation pipelines.
2. **Manual data cleaning (20,000 files)** — Manually reviewed all training files to remove segments without the target bird calling; pseudo-label approaches failed because training data contained long silent or background-noise-only segments.
3. **Power-law contrast augmentation** — Raised spectrogram pixel values to a random power (0.5–3.0); at 0.5 background noise becomes more prominent, at 3.0 quiet sounds fade — simulates variable recording conditions and distances.
4. **Upper-frequency suppression augmentation** — With 50% probability, lowered amplitude of upper frequencies; mimics real-world acoustic attenuation where high frequencies fade faster with distance.
5. **BCEWithLogitsLoss with soft background labels** — Primary bird label = 1.0, background/secondary species label = 0.3; multi-label soft targets improved F1 vs. hard 0/1 labels.

**How to Reuse:**
- For any audio classification task, pre-save spectrograms as images — removes training bottleneck and enables image augmentation libraries (albumentations, etc.) out of the box.
- Manual data cleaning often outperforms automated filtering for soundscape/wildlife audio; spend time on a small random sample first to understand label quality.
- Upper-frequency suppression and power-law contrast are domain-specific augmentations for any distance-sensitive acoustic task (wildlife monitoring, underwater acoustics).

---

## 8. Market Basket Analysis (2017) — 2nd Place

**Task type:** Tabular product recommendation — predict which products a customer will re-order in the next order given purchase history (Instacart dataset), optimized for F1 score
**Discussion:** [https://www.kaggle.com/c/basket-analysis/discussion/38143](https://www.kaggle.com/c/basket-analysis/discussion/38143)

**Approach:** Ad-hoc feature engineering with CatBoost as primary model, enhanced with LDA/NMF latent features, and a custom F1-optimization algorithm that accounts for prediction correlation and uncertainty.

**Key Techniques:**
1. **Exact F1 Maximization via joint probability matrix** — Used an O(n³) exact F1 maximization algorithm (vs. the common O(n²) greedy approach) by constructing an empirical joint probability matrix of product co-purchase; improved F1 by 0.001–0.0015 over the sub-optimal greedy method.
2. **Uncertainty-based probability correction** — Multiplied purchase probabilities by a factor negatively correlated with conditional basket size: `1.6^(1/n²)` where n is the conditional basket size; when probability estimates are uncertain (large baskets), the correction pushes scores toward safer "include all" decisions.
3. **CatBoost with "accurate" mode** — CatBoost's accurate mode gave 0.0003–0.0005 improvement over LightGBM/XGBoost on this dataset; XGBoost with the "accurate" option (vs. approximate) gave 0.0015 additional lift.
4. **LDA and NMF latent factors** — Added Latent Dirichlet Allocation and Non-negative Matrix Factorization features capturing latent purchase patterns; gave 0.0005–0.001 improvement on top of base tabular features.
5. **Conditional basket size features** — Engineered the conditional basket size for each product: "given a user buys this product, how many total products do they buy in this order?" — a key signal for predicting re-order likelihood.

**How to Reuse:**
- For any recommendation/retrieval task with F1 as the metric, implement the exact F1 maximization algorithm rather than greedy threshold search — the gain is measurable and implementation is straightforward.
- When predicting probabilities for sparse events, add uncertainty-scaling corrections based on feature count or basket size; pure probability scores are often poorly calibrated for multi-label decisions.
- LDA/NMF latent features add complementary signal to tree-based models on transactional data — worth a quick experiment as they rarely hurt and often help by 0.001+.

---

## 9. HMS - Harmful Brain Activity Classification (2024) — 2nd Place

**Task type:** Multi-class classification of EEG patterns (seizure, LPD, GPD, LRDA, GRDA, Other) from 50-second EEG recordings with spectrogram images
**Discussion:** [https://www.kaggle.com/c/hms-harmful-brain-activity-classification/discussion/492254](https://www.kaggle.com/c/hms-harmful-brain-activity-classification/discussion/492254)

**Approach:** Multimodal ensemble of 6 models treating EEG data as both spectrograms (fed to X3D-L 3D-CNN) and reshaped raw signals (fed to EfficientNetB5 as 2D images), with a hybrid model combining both. Two-stage training on all data then high-confidence samples only.

**Key Techniques:**
1. **X3D-L (3D-CNN) for spectrograms** — After double-banana montage + 0.5–20Hz bandpass filter + STFT, the resulting spectrograms are processed by X3D-L, a lightweight video CNN; 3D convolutions capture both frequency and time relationships simultaneously that 2D CNNs miss.
2. **EfficientNetB5 treating raw EEG as images** — EEG reshaped from (16 channels, 10000 samples) to (160, 1000) 2D array, then treated as an image; avoids the channel-padding limitation of 2D CNNs (which lose positional information across channels) while remaining computationally tractable.
3. **Two-stage training: all data → high-confidence only** — Stage 1: 15 epochs with loss weighted by `voters_num/20` (more annotators = higher weight); Stage 2: 5 epochs on samples with `voters_num≥6` only at a lower learning rate; dramatically improved reliability of gradient signal.
4. **Hemisphere-flipping augmentation** — Swapped left and right brain channel data during training; a domain-specific augmentation reflecting the bilateral symmetry of brain activity, approximately doubling effective data.
5. **Ensemble diversity via filter library variation** — Used MNE-based filtering and scipy-based filtering as separate models; same architecture, different signal processing libraries produced diverse enough predictions to yield ensemble gains.

**How to Reuse:**
- For EEG/physiological signal tasks, try reshaping time-series as 2D images and processing with standard image CNNs — it avoids building custom temporal architectures and often matches LSTM/Transformer performance.
- Two-stage training (all data first, then high-confidence subset) is a principled approach for label-noisy competitions; the first stage provides stable initialization, the second refines on clean signal.
- Hemisphere/bilateral symmetry augmentation applies broadly to paired biosensor data (e.g., bilateral EMG, paired ECG leads, stereoscopic images).

---

## 10. TensorFlow - Help Protect the Great Barrier Reef (2022) — 3rd Place

**Task type:** Underwater object detection (crown-of-thorns starfish) in video frames
**Discussion:** [https://www.kaggle.com/c/tensorflow-great-barrier-reef/discussion/307707](https://www.kaggle.com/c/tensorflow-great-barrier-reef/discussion/307707)

**Approach:** Pure YOLOv5l6 ensemble with video-ID-based cross-validation, random 90° rotation augmentation, and a custom "attention area" tracking post-processor. No external data or exotic architectures — "nothing fancy."

**Key Techniques:**
1. **Video-ID fold splitting** — Split by `video_id` into 3 folds rather than by subsequence; video-level splitting gave stable CV that positively correlated with public and private LB, whereas subsequence splitting produced noisy LB scores.
2. **Random 90° rotation augmentation** — Added random 90° rotations on top of standard YOLOv5 augmentation (mosaic, mixup); gave ~0.02 boost on both CV and LB, and also increased diversity between ensemble members trained with different rotations.
3. **Train on ground-truth frames only** — Contrary to YOLOv5 documentation recommending 10% background frames, training on 0% background (only frames with annotations) gave best results; the small and specific target (starfish) benefited from focused training signal.
4. **Multiscaling ±50%** — Enabled YOLOv5's built-in multi-scale training (±50% scale variation); especially important for anchor-based detectors when the target can appear at many distances/sizes in video.
5. **"Attention Area" temporal post-processing** — Custom tracker: if a detection appears in frame N, enlarge the predicted bounding box slightly and boost confidence of nearby detections in frames N±k; propagated high-confidence detections through time without complex Kalman filter tracking.

**How to Reuse:**
- For video object detection, always validate by video/sequence ID — frame-level splits leak temporal context and overstate CV performance.
- Training on annotation-only frames (0% background) often beats the recommended 10% background when the target object is rare and visually distinctive; test both.
- A simple temporal post-processor (boost confidence near prior detections) can outperform off-the-shelf trackers in competitions where the target moves predictably between frames.

---

## 11. TGS Salt Identification Challenge (2018) — 9th Place

**Task type:** Binary segmentation of salt deposits in seismic reflection images
**Discussion:** [https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/69053](https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/69053)

**Approach:** Single SENet154 model with 10-fold cross-validation and reflective padding; trained with AdamW + Noam scheduler + cutout, then refined with Stochastic Weight Averaging. Symmetric Lovász loss was the key differentiator.

**Key Techniques:**
1. **Symmetric Lovász Loss** — Modified standard Lovász hinge loss to be symmetric: `(lovász_hinge(preds, targets) + lovász_hinge(-preds, 1-targets)) / 2`; gave +0.008 public LB and +0.02 private LB improvement — the single biggest gain in the solution.
2. **SENet154 backbone** — Upgraded from SE-ResNeXt50 to SENet154 (full squeeze-and-excitation network); single-fold score improved from 0.87x to 0.882/0.869 CV/LB.
3. **Stochastic Weight Averaging (SWA)** — Applied SWA over best loss, pixel accuracy, metric, and last checkpoint models after main training; gave +0.004 consistent improvement.
4. **AdamW + Noam scheduler** — Used weight-decoupled AdamW with the Noam warmup-then-decay schedule (from "Attention Is All You Need"); more stable convergence than standard Adam with StepLR on segmentation tasks.
5. **Reflective padding + 10-fold CV** — Used reflective boundary padding (not zero-padding) to avoid edge artifacts in seismic images; 10-fold cross-validation with all folds used for inference improved final score from 0.882 to 0.890.

**How to Reuse:**
- Symmetric Lovász loss (averaging Lovász over both positive and negative class) is a general technique for binary segmentation with imbalanced classes — implement it before any architectural experiments.
- SWA is a near-free ensemble that consistently adds 0.003–0.005 to segmentation scores; always add it as the final training stage.
- For small-image segmentation tasks (101×101 in this case), simple decoders (single conv + transposed conv) match complex attention decoders (DANet, OCNet) — don't over-engineer the decoder.

---

## 12. Feedback Prize - English Language Learning (2022) — 1st Place

**Task type:** Regression — predict 6 English writing proficiency scores (cohesion, syntax, vocabulary, phraseology, grammar, conventions) from student essays; metric is MCRMSE
**Discussion:** [https://www.kaggle.com/c/feedback-prize-english-language-learning/discussion/369457](https://www.kaggle.com/c/feedback-prize-english-language-learning/discussion/369457)

**Approach:** DeBERTa-v3-large ensemble with rank loss + back-translation pretraining + pseudo-label pretraining from the Feedback Prize 1 dataset; Optuna-tuned ensemble weights with careful OOF-based validation.

**Key Techniques:**
1. **Rank Loss auxiliary objective** — Added a pairwise ranking loss on top of regression loss (MSE/Pearson); rank loss consistently improved both CV and private LB across all model variants, suggesting that the ordinal relationship between scores matters as much as absolute values.
2. **Back-translation pretraining** — Pretrained DeBERTa-v3-large on back-translated versions of the training essays (translate to another language and back to English) before fine-tuning; provides data augmentation at the corpus level.
3. **Feedback Prize 1 pseudo-label pretraining** — Used predictions from a well-trained model to create pseudo-labels on the Feedback Prize 1 competition data, then pretrained on that larger labeled corpus; models with more background knowledge (larger effective training set) consistently outperformed on both CV and private LB.
4. **DeBERTa-v3-large as dominant backbone** — DeBERTa-v3-base and v3-large were the only backbones that worked well; all others (roberta, deberta-v2, etc.) were significantly worse; selecting the right pretrained model is the single most important decision.
5. **Optuna ensemble weight tuning with per-model per-target rules** — Used Optuna to tune blending weights per model and per target score on full OOF; however, noted that this can overfit — per-model per-target hand-tuned weights (without Optuna) were more stable on private LB.

**How to Reuse:**
- For any writing quality regression task, add a pairwise rank loss alongside the primary regression loss — it reliably improves ranking accuracy even when raw regression error doesn't improve much.
- Pseudo-labeling from related competition datasets (e.g., earlier editions of the same competition) is a powerful and legal form of data augmentation when same-domain text exists publicly.
- When tuning ensemble weights with Optuna on OOF, validate the tuned weights on a held-out fold — Optuna ensemble weight search easily overfits OOF, so a nested validation is essential.

---

## 13. NBME - Score Clinical Patient Notes (2022) — 1st Place

**Task type:** NLP span extraction — identify which character spans in clinical patient notes correspond to medical features in a scoring rubric
**Discussion:** [https://www.kaggle.com/c/nbme-score-clinical-patient-notes/discussion/323095](https://www.kaggle.com/c/nbme-score-clinical-patient-notes/discussion/323095)

**Approach:** DeBERTa-based token classification pipeline with extensive preprocessing for inconsistent annotations, a custom RNN on top of the transformer for sequence-dependent span prediction, adversarial training, and pseudo-labeling from 10x more unlabeled data.

**Key Techniques:**
1. **Hypothesis-driven annotation inconsistency handling** — Observed that annotators tended to miss repeated occurrences of the same feature; hypothesized that annotations have sequence dependency (first occurrence more likely labeled than repeat); trained a GRU on top of transformer features to capture this dependency explicitly.
2. **Medical abbreviation normalization** — Pre-processed clinical text by mapping common medical abbreviations (FHx→FH, PMHx→PMH, SHx→SH) to full forms; improved tokenizer alignment and reduced out-of-vocabulary fragmentation in clinical shorthand.
3. **Pseudo-labeling from unlabeled data** — The competition provided 10x unlabeled clinical notes; trained models to generate pseudo-labels, then pretrained on pseudo-labeled data before fine-tuning on gold labels; the large unlabeled pool was the competition's key differentiator.
4. **Adversarial training (AWP/FGM)** — Applied adversarial weight perturbation during fine-tuning; consistently improved generalization across DeBERTa variants in NLP span extraction tasks.
5. **Lowercase normalization** — Converted all text to lowercase; clinical notes mixed uppercase/lowercase without semantic significance, and uncased modeling reduced token vocabulary fragmentation.

**How to Reuse:**
- For span extraction with noisy/inconsistent annotations, model the annotation process itself (e.g., with an RNN that conditions on previous span predictions) rather than treating each span independently.
- When unlabeled in-domain data is available (even 2x the labeled set), always build a pseudo-labeling pipeline — it's the highest-ROI data augmentation for NLP.
- Clinical text preprocessing (abbreviation expansion, case normalization) is necessary for transformer models pretrained on web text; define a minimal set of domain-specific normalizations before running any baseline.

---

## 14. UW-Madison GI Tract Image Segmentation (2022) — 1st Place

**Task type:** Multi-class segmentation of stomach, small bowel, and large bowel from MRI slices
**Discussion:** [https://www.kaggle.com/c/uw-madison-gi-tract-image-segmentation/discussion/337197](https://www.kaggle.com/c/uw-madison-gi-tract-image-segmentation/discussion/337197)

**Approach:** Two-stage pipeline — YOLOv5-based positive/negative slice classifier (stage 1) → large backbone segmentation model on positive slices only (stage 2). 2.5D (5-slice context) and 3D (DynUNet) models ensembled.

**Key Techniques:**
1. **Two-stage positive/negative slice filtering** — Stage 1 (EfficientNet-B4 or Swin-Base) classifies each slice as containing anatomy or not; stage 2 runs the expensive segmentation model only on positive slices. This doubles effective compute budget for the hard cases.
2. **YOLOv5 crop for body/arm signal removal** — Used YOLOv5 to detect and crop the abdominal region, removing arm signals that cause min-max normalization failure (B1 field inhomogeneity creates arm "hot spots" in abdominal MRI).
3. **Backbone scaling: EfficientNet-B4 → L2 → ConvNeXt XL → Swin-Large** — Systematically increased backbone size in stage 2; each step improved validation Dice (B4: 0.8011 → L2: 0.8349). Large backbones are the dominant driver of segmentation quality.
4. **UperNet decoder with CE + Dice loss (1:1)** — Combined cross-entropy and Dice loss equally; UperNet provides multi-scale feature aggregation through a Feature Pyramid Network, improving detection of both small and large anatomical structures.
5. **3D DynUNet ensemble** — Added 3D DynUNet models (1000 epochs with SWA) as a complement to 2.5D models; 3D context improved continuity of segmentation across slices; 2.5D+3D ensemble consistently outperformed either alone.

**How to Reuse:**
- Two-stage filtering (detect positive slices/regions first, then segment) is a universal pattern for volumetric medical imaging — reduces class imbalance and focuses compute on relevant slices.
- YOLOv5 as a fast preprocessing crop tool (not just final detector) is underused; it's a quick way to normalize input regions and remove off-target signal.
- When backbone scaling plateaus (L2 → ConvNeXt XL → Swin-Large all improve), invest in 2.5D+3D ensemble before trying more exotic architectures.

---

## 15. G2Net Gravitational Wave Detection (2021) — 2nd Place

**Task type:** Binary classification of gravitational wave signals in time-series data from LIGO/Virgo detectors; metric AUC
**Discussion:** [https://www.kaggle.com/c/g2net-gravitational-wave-detection/discussion/275341](https://www.kaggle.com/c/g2net-gravitational-wave-detection/discussion/275341)

**Approach:** Multiple trainable frontend CNN architectures (CWT-CNN, 1D-CNN) with bandpass-filtered inputs, combined via Ridge regression stacking with greedy model selection on 20 diverse models.

**Key Techniques:**
1. **Trainable frontend signal processing** — Used a trainable frontend (learned filters replacing fixed CWT or bandpass) as the first network layer; trainable frontends consistently outperformed fixed signal processing for spectrogram/waveform inputs, suggesting that optimal signal representations differ from classical signal processing choices.
2. **Bandpass filtering as preprocessing** — Applied bandpass filter [16–512 Hz] for CWT-based and 2D-CNN networks, [30–300 Hz] for 1D-CNN; physical domain knowledge (gravitational waves are in specific frequency bands) constrained the signal space.
3. **Wave-domain augmentations** — Gaussian noise addition (for 2D-CNN) and flipped wave amplitude (for 1D-CNN) were the only augmentations that consistently improved AUC; most augmentations tried on the spectrogram domain did not help.
4. **Soft pseudo-labeling with label smoothing** — Re-trained on continuous (soft) pseudo-labels for the test set, with label smoothing applied during pseudo-label generation; improved AUC by ~0.001 — small but measurable gain.
5. **Ridge regression stacking with greedy model selection** — Kept OOF predictions from all experiments; Ridge regression on 20 diverse models (CV 0.88283, Private LB 0.8829) outperformed subsets of 10 (0.8827) or 5 (0.8825). Greedy model selection to maximize CV identified the optimal 20-model subset.

**How to Reuse:**
- For any signal classification task (audio, seismic, gravitational wave), try a trainable frontend layer before committing to a fixed preprocessing pipeline — it often learns better feature representations than hand-crafted transforms.
- Keep all OOF predictions from experiments and build a Ridge stacking meta-model at the end; the incremental cost is minimal and the ensemble gains from diverse model predictions are reliable.
- Greedy forward model selection (add models one by one if they improve ensemble CV) prevents overfitting the Ridge stacking weights on small validation sets.

---

## 16. Eedi - Mining Misconceptions in Mathematics (2024) — 1st Place

**Task type:** NLP information retrieval — given a multiple-choice math question with a wrong answer, retrieve the most likely mathematical misconception from a taxonomy of 2,587 misconceptions; metric MAP@25
**Discussion:** [https://www.kaggle.com/c/eedi-mining-misconceptions-in-mathematics/discussion/551402](https://www.kaggle.com/c/eedi-mining-misconceptions-in-mathematics/discussion/551402)

**Approach:** Two-stage retriever-reranker pipeline using Qwen2.5-32B-Instruct fine-tuned with LoRA. Retriever uses last-token embeddings with contrastive loss to retrieve top-25 misconceptions; reranker re-orders using a supervised fine-tuned model. Staged training on synthetic GPT-4o-mini data → MalAlgoQA data → competition training data.

**Key Techniques:**
1. **Large LLM as embedding model** — Used Qwen2.5-32B-Instruct (not a purpose-built embedding model) for retrieval; the last-token (EOS) embedding from a generative LLM with LoRA fine-tuning outperformed standard sentence transformers for this specialized mathematical reasoning domain.
2. **Staged training on data of increasing quality** — Trained in three sequential stages: (1) GPT-4o-mini synthetic data, (2) MalAlgoQA academic dataset, (3) competition training data; different data sources have format conflicts and quality differences that make joint training suboptimal.
3. **Hard negative mining for reranker** — Retrieved top-150 candidates and sampled 25 hard negatives per training example; harder negatives (similar-but-wrong misconceptions) are essential for training a reranker to distinguish fine-grained differences in math reasoning.
4. **LoRA fine-tuning with 4-bit quantization** — Applied LoRA (r=16, alpha=32) across q/k/v projections with BitsAndBytes 4-bit quantization to make 32B model training feasible on competition hardware; enables billion-parameter model fine-tuning at fraction of full-parameter cost.
5. **Single-tower reranker architecture** — Reranker maps concatenated (question, candidate misconception) token embeddings to a similarity scalar via a single-tower model; simpler than cross-encoder but captures query-candidate interaction better than two-tower dot product.

**How to Reuse:**
- For specialized retrieval tasks in technical domains (math, medicine, law), fine-tuning a large generative LLM as an embedding model outperforms off-the-shelf sentence transformers — the domain knowledge in the pretrained LLM is more valuable than the embedding specialization.
- Staged training (synthetic → semi-supervised → gold labels) is a reliable curriculum for NLP retrieval when gold labels are scarce; always start with the noisiest/most general data and end with the most specific.
- LoRA + 4-bit quantization is now the standard approach for fine-tuning 30B+ models on single or dual GPU setups in competitions — the accuracy loss vs. full fine-tuning is minimal for retrieval tasks.

---

## 17. NBME - Score Clinical Patient Notes (2022) — 4th Place

**Task type:** NLP span extraction — same competition as entry #13 above
**Discussion:** [https://www.kaggle.com/c/nbme-score-clinical-patient-notes/discussion/322799](https://www.kaggle.com/c/nbme-score-clinical-patient-notes/discussion/322799)

**Approach:** Ensemble of 4 token classification models (DeBERTa-v3-large, DeBERTa-v2-xlarge, DeBERTa-v2-xxlarge) each with MLM fine-tuning, SmoothFocalLoss, and 2× pseudo-labeling, plus a character-level classification model (DeBERTa + GRU) for additional diversity.

**Key Techniques:**
1. **MLM fine-tuning before span extraction** — Continued masked language model pretraining (masking rate 0.10–0.15) on the clinical notes corpus before fine-tuning on span extraction; domain-adaptive pretraining on the specific text improved F1 by ~0.002–0.005.
2. **SmoothFocalLoss for token classification** — Used a smoothed version of focal loss (rather than BCE) to down-weight easy negative tokens; clinical notes have many non-entity tokens, making focal loss a natural fit for the heavy class imbalance.
3. **Character-level span classification with GRU head** — Trained a character-level model (DeBERTa tokenizing at char level + 4-layer GRU) alongside token-level models; character classification catches span boundary errors from tokenizer misalignment, especially for hyphenated or abbreviated medical terms.
4. **Text augmentations: mask + replace** — Applied mask augmentation (randomly mask feature_text tokens) and replace augmentation (replace feature_text tokens with synonyms) during training; both improved generalization on unseen clinical note styles.
5. **Per-case_num threshold tuning + postprocessing** — Set separate confidence thresholds for each clinical case type and applied rule-based postprocessing to fix common tokenization span errors (off-by-one, hyphenated numbers, leading newlines); each postprocessing rule added 0.0001–0.0005.

**How to Reuse:**
- Domain-adaptive MLM pretraining on in-domain text is a reliable +0.002–0.005 improvement for any NLP span extraction task with specialized vocabulary — always do it before final fine-tuning.
- For multi-model NLP ensembles, adding a character-level model provides complementary signal to subword-tokenized models; it's especially valuable when span boundaries fall at subword token boundaries.
- Rule-based postprocessing of span predictions (fix obvious off-by-one errors, tokenization artifacts) consistently adds small but reliable gains in span extraction; budget time for it at the end of development.

---

## 18. Prostate cANcer graDe Assessment (PANDA) Challenge (2020) — 11th Place

**Task type:** Multi-class classification of prostate cancer Gleason grade from whole-slide histopathology images (6 ISUP grades); metric quadratic weighted kappa
**Discussion:** [https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion/169205](https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion/169205)

**Approach:** Ensemble of EfficientNet and DenseNet models on tiled 256×256 patches (42 tiles per slide), trained with MSE + BCE combined loss on TPU using a published baseline methodology (Iafoss tiling), with 5-fold cross-validation.

**Key Techniques:**
1. **Tile-based WSI processing (42× 256×256)** — Extracted 42 tiles of 256×256 from each whole-slide image using white-padding tile extraction (Iafoss method), discarding background tiles; converted an arbitrarily large gigapixel image into a fixed-size tensor processable by standard CNNs.
2. **MSE + BCE combined loss for ordinal regression** — Used mean squared error on Gleason grade (treating it as continuous) combined with sigmoid cross-entropy for binary grade presence; combined loss outperformed ordinal-only or classification-only objectives on quadratic kappa metric.
3. **TPU training via TF records** — All training performed on Kaggle/Google TPU using TF-Record format for data pipeline; TPU training with properly formatted tf-records was 5–10× faster than GPU for this tile-based workflow.
4. **Cross-validation with consensus label checking** — Used StratifiedKFold on grade labels with verification that external data labels (from multiple annotation sources) agreed with official labels; label conflicts resolved in favor of official labels.
5. **EfficientNet + DenseNet ensemble** — Final submission ensembled EfficientNet and DenseNet model families; architectural diversity (different inductive biases) provided complementary predictions on ambiguous grade cases.

**How to Reuse:**
- Tile-based WSI processing (sample N non-background patches, concatenate, process with CNN) is the standard approach for pathology competitions; use the Iafoss/PANDA tile extraction as a reference implementation.
- For ordinal regression with kappa as metric, combine MSE (regression) and BCE (classification) losses — neither alone optimizes kappa as well as the combination.
- TPU training via tf-records gives a significant speedup for tile-based pathology workloads; set up the tf-record pipeline early rather than treating it as an optimization.

---

## 19. Riiid Answer Correctness Prediction (2021) — 4th Place

**Task type:** Sequential prediction — given a student's history of question interactions (content, response, time), predict whether they will answer the next question correctly; metric AUC
**Discussion:** [https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/210171](https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/210171)

**Approach:** Single encoder-decoder Transformer (inspired by SAINT/SAKT) with continuous embeddings for temporal features, time-aware weighted attention decay, trained on TPU with random sequence cutting and padding.

**Key Techniques:**
1. **ContinuousEmbedding for time/difficulty features** — Mapped continuous features (time lag, question elapsed time, difficulty, popularity) to a weighted sum of consecutive embedding vectors; produces smooth embeddings where similar values have similar representations — avoids the sharp discontinuities of binned embeddings.
2. **Time-aware weighted attention** — Decayed attention coefficients by `dt^(-w)` where `dt` is timestamp difference and `w` is a trainable non-negative parameter per head; explicitly penalizes attending to distant-in-time interactions, improving convergence speed and performance.
3. **Encoder-decoder architecture with causal masks** — Encoder receives all input features; decoder excludes user-answer-related features (prevents leakage), with causal masking preventing current position from attending to future; same architecture as vanilla Transformer but with feature-level separation.
4. **Random sequence cutting and padding for training** — Randomly cut sequences to variable lengths and padded to uniform length during training (not just from sequence start); ensures all positions in the sequence are used for training predictions, not just the end.
5. **Question difficulty and popularity as features** — Computed per-question difficulty (correct response rate) and popularity (number of appearances) from the full training corpus; global question statistics as features were important signals beyond the student's individual history.

**How to Reuse:**
- ContinuousEmbedding (weighted sum of neighboring embedding vectors for continuous values) is a general technique for embedding any continuous feature in a Transformer without binning; use it for time deltas, prices, durations, and other continuous signals.
- Time-aware attention decay (penalize attention over large time gaps) is a simple modification to standard attention that improves performance on any task where recency matters — implement as a single additive term to attention logits.
- For knowledge tracing or sequential recommendation, the encoder-decoder split (encoder sees all features, decoder excludes target-related features) is a principled way to prevent data leakage while preserving sequence modeling capacity.

---

## 20. Human Protein Atlas Image Classification (2019) — 3rd Place

**Task type:** Multi-label image classification of subcellular protein localization from fluorescence microscopy images (28 classes, heavily imbalanced); metric macro F1
**Discussion:** [https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/77320](https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/77320)

**Approach:** Ensemble of ResNet34 (512×512), InceptionV3 (1024×1024), and SE-ResNeXt50 (1024×1024) with Focal loss, per-image normalization, AutoAugment-style augmentation search, and frequency-proportional threshold selection.

**Key Techniques:**
1. **Per-image mean/std normalization** — Normalized each image to its own mean and standard deviation (not dataset-wide statistics); critical because official and external datasets had very different intensity distributions, making global normalization harmful.
2. **AutoAugment via random search** — Used random search over augmentation policies (simpler than the RL-based original AutoAugment) to find the best augmentation for this specific dataset; outperformed manually designed augmentations without the RL training cost.
3. **Focal loss (gamma=2) for extreme class imbalance** — Applied focal loss to down-weight easy negative predictions for the 28 classes (some with <100 examples); focal loss was essential for learning rare localization patterns.
4. **Early stopping on majority class F1** — Instead of stopping on macro F1 (which is driven by rare classes and is unstable), stopped training when F1 for the majority class (class 0, Nucleoplasm) began to decline; prevented over-tuning toward rare classes at the expense of majority class performance.
5. **Frequency-proportional threshold selection** — For each class, set threshold so that the proportion of positive predictions in validation matches the proportion of positive training examples; a simple calibration approach that outperformed cross-validated threshold search on this imbalanced dataset.

**How to Reuse:**
- For multi-label classification with external data having different intensity characteristics, always normalize per-image (not per-dataset); global normalization will degrade performance when data sources differ.
- Stopping early based on majority class performance (not macro metric) prevents rare-class over-tuning that hurts overall macro F1 — use this heuristic when macro F1 oscillates during training.
- Frequency-proportional threshold selection is a fast, reliable alternative to threshold search for imbalanced multi-label tasks; implement it before any exhaustive threshold optimization.

---

## 21. Google Brain - Ventilator Pressure Prediction (2021) — 9th Place

**Task type:** Time-series regression — predict airway pressure at each timestamp for mechanical ventilator breath cycles given R (resistance) and C (compliance) settings and `u_in` control input; metric MAE
**Discussion:** [https://www.kaggle.com/c/ventilator-pressure-prediction/discussion/285353](https://www.kaggle.com/c/ventilator-pressure-prediction/discussion/285353)

**Approach:** 4-layer LSTM with skip connections, trained on extensively feature-engineered tabular data including aggregations over R, C, rank, and rounded u_in. The key differentiator was feature engineering, not architecture.

**Key Techniques:**
1. **Magic aggregation features over R, C, rank, and rounded u_in** — Created aggregate statistics (mean, max, min, std) grouping by combinations of R value, C value, breath sequence rank, and rounded u_in; these features captured the PID controller's response curves that dominate ventilator behavior.
2. **Reversed-order training** — Trained the model to predict pressure from timestamp 80 back to timestamp 1 (reversed), while including features computed in forward order; the reversed direction forced the model to learn pressure trajectory from endpoint context.
3. **Quantile transformation of u_in for u_out=0** — Applied quantile transformation to u_in values specifically when u_out=0 (exhale phase); the exhale phase has a different u_in distribution requiring separate normalization.
4. **Negative pressure encoding for u_out=0** — Treated pressure values during u_out=0 as negative in the feature space; this physics-informed encoding captured the distinct exhale dynamics.
5. **LSTM + Dense + skip connections** — Simple 4-layer LSTM with skip connections between layers and a final dense output; skip connections stabilized gradient flow through the 4 recurrent layers without gating complexity.

**How to Reuse:**
- For physical system modeling (ventilators, HVAC, engines), physics-informed feature engineering (grouping by physical parameters like R and C, separate normalization for different operating phases) often dominates architectural choices.
- Reversed-order sequence training is a useful trick for time-series with strong endpoint predictability; train in both directions and ensemble if the direction matters for the problem.
- For ventilator/respiratory tasks specifically, separate the inhale (u_out=1) and exhale (u_out=0) phases as fundamentally different regimes requiring different feature representations.

---

## 22. Tweet Sentiment Extraction (2020) — 1st Place

**Task type:** NLP span extraction — given a tweet and its sentiment label (positive/negative/neutral), extract the word span that best supports the sentiment; metric Jaccard similarity
**Discussion:** [https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/159264](https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/159264)

**Approach:** RoBERTa-large ensemble (pretrained on SQuAD2) with SentimentSampler to balance class distribution in batches, Stochastic Weight Averaging for training stability, reranking model post-processing, and SequenceBucketing for training efficiency.

**Key Techniques:**
1. **SentimentSampler for batch imbalance** — Custom sampler that equalized positive/negative/neutral sentiment within each batch; vanilla StratifiedKFold wasn't sufficient — within-batch distribution mattered because the sentiment label is a direct input that shapes span extraction differently per class.
2. **Stochastic Weight Averaging (SWA)** — Applied SWA because validation score varied ±0.001 across single iterations, making checkpoint selection unreliable; SWA stabilized results to ±0.0001 range over 10–50 iterations, enabling confident model selection.
3. **RoBERTa pretrained on SQuAD2** — Used SQuAD2-pretrained RoBERTa (not raw RoBERTa-base) as the span extraction backbone; QA pretraining aligns the model's span extraction head with the exact task structure.
4. **Multi-dropout (MDO) output layer** — Applied multiple dropout masks to the output layer and averaged predictions across them during training and inference; reduces variance without requiring explicit ensemble training (borrowed from Google Quest 1st place).
5. **Reranking post-model** — Trained a separate reranking model that re-scored candidate spans predicted by the base model; provided an additional 0.001–0.003 Jaccard improvement on top of the base extraction model.

**How to Reuse:**
- SWA is especially valuable when validation scores are highly volatile during training — always try SWA before investing in larger ensembles for stabilization.
- For any QA-formatted span extraction task, start with a model already fine-tuned on SQuAD2 (or a similar QA dataset) rather than raw pretrained weights; task-aligned initialization improves convergence and final performance.
- SentimentSampler (or class-conditional batch sampling for any conditioning variable) is important whenever the input conditioning signal (sentiment, document type, domain) should be uniformly represented in each training batch.

---

## 23. SIIM-ACR Pneumothorax Segmentation (2019) — 3rd Place

**Task type:** Binary segmentation of pneumothorax (collapsed lung) in chest X-rays from 1024×1024 DICOM images
**Discussion:** [https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/107981](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/107981)

**Approach:** SE-ResNeXt50 UNet trained on lung-cropped images (576×576) with pseudo-labeled external data from CheXpert and NIH datasets. Final ensemble of 3 SE-ResNeXt50 models at different scales (704×704, 576×576) with and without pseudo-labels.

**Key Techniques:**
1. **Lung crop preprocessing via UNET** — Trained a lightweight ResNet34 UNet to crop the lung region from 1024×1024 X-rays to 576×576; reduced input size and removed irrelevant background, enabling larger batch sizes and faster training without losing diagnostic detail.
2. **Pseudo-labeling from CheXpert and NIH datasets** — Used a model trained on competition data to generate pseudo-labels for CheXpert positive cases; selected only samples predicted positive by the model to reduce label noise (NIH labels were not used directly as they were too noisy).
3. **Balanced pseudo-label mixing** — Kept positive/negative ratio equal (1:1) in pseudo-labeled training; pseudo-label samples were capped at 50% of normal samples per batch to prevent the noisier pseudo-data from dominating.
4. **CBAM attention module** — Added Convolutional Block Attention Module (CBAM) to the UNet skip connections; channel-spatial attention recalibrated features toward pneumothorax-relevant regions in the lung.
5. **Lovász loss without threshold search** — Used Lovász loss (not BCE) for training and fixed the decision threshold at 0.5 without validation-set search; author argued that threshold search on public LB was unreliable for private, and fixed threshold generalized better.

**How to Reuse:**
- Anatomical cropping (train a fast segmenter to isolate the organ of interest, then use those crops for all downstream models) is standard practice for chest X-ray and other medical imaging tasks; always implement it before architectural experiments.
- Pseudo-labeling from related public medical datasets (CheXpert, NIH Chest X-ray, MIMIC) provides significant data augmentation for medical segmentation — use model confidence as a filter and keep positive/negative ratios controlled.
- Fixed thresholds (0.5) often generalize better than tuned thresholds when the validation set is small or unrepresentative of the private test distribution; test both approaches and compare private LB variance.

---

Now let me assemble this into the final document output. The content is complete and ready.

Here is the full compiled markdown document:

---

# Kaggle Past Solutions — SRK Round 2, Batch 3

Source: kaggle.com/sudalairajkumar/winning-solutions-of-kaggle-competitions
Ingested: 2026-04-17
Note: These are non-1st-place solutions (2nd, 3rd, top-10) with 100+ upvotes. Writeup content sourced from the thedrcat/kaggle-winning-solutions-methods dataset (CC-BY-NC-SA-4.0). For 3 entries not in that dataset (RSNA Pneumonia 2018 2nd place, HMS 2024 2nd place, Eedi 2024 1st place), details were synthesized from GitHub repos, DEV Community write-ups, and web research.

---

## 1. TensorFlow - Help Protect the Great Barrier Reef (2022) — 5th Place

**Task type:** Underwater video object detection (crown-of-thorns starfish)
**Discussion:** [https://www.kaggle.com/c/tensorflow-great-barrier-reef/discussion/308007](https://www.kaggle.com/c/tensorflow-great-barrier-reef/discussion/308007)

**Approach:** CenterNet detector on a DeepLabV3+ architecture with EfficientNetV2 backbone. Trained at 1280×720, inferred at 1792×1008 (1.4× upscale) and blended with 1.6× scale predictions.

**Key Techniques:**
1. **CenterNet on DeepLabV3+** — Repurposed a semantic segmentation backbone (DeepLabV3+) for CenterNet detection by adding a regression head; heatmap output set to 1/8 input resolution (not standard 1/4) for improved localization precision.
2. **EfficientNetV2 backbone scaling** — Systematically swapped backbone B0→XL; larger backbones consistently improved score, confirming scale is the dominant variable once architecture is fixed.
3. **Multi-resolution inference blending** — Trained at 1280×720, inferred at 1.4× and 1.6× upscale and averaged; adds ~0.01 LB with no retraining cost.
4. **Avoiding overfitting strategy** — Pseudo-labeling on test video sequences failed on private LB (large shake-up); clean single-model training was more reliable when train/test video sequences differ.
5. **Ensemble across scales** — Even a simple two-scale blend (1.4× vs 1.6×) provided consistent improvement over single-scale inference.

**How to Reuse:**
- When adapting segmentation architectures (DeepLabV3+, UNet) for detection, adding a CenterNet head can be competitive with purpose-built detectors without changing the backbone.
- Multi-scale inference blending is a free ensemble — always try 2–3 scales at inference time.
- In video competitions, be cautious with pseudo-labeling loops if train/test video sequences differ — validate on held-out video IDs, not subsequences.

---

## 2. DFL - Bundesliga Data Shootout (2022) — 1st Place

**Task type:** Action spotting in soccer video — detect and timestamp events (challenge, play, throwin) from multi-camera broadcast footage
**Discussion:** [https://www.kaggle.com/c/dfl-bundesliga-data-shootout/discussion/359932](https://www.kaggle.com/c/dfl-bundesliga-data-shootout/discussion/359932)

**Approach:** End-to-end 2.5D model: EfficientNet backbone with Temporal Shift Module (TSM), followed by a 1D UNet head that produces per-frame event probability over time. Pretrained on SoccerNet ball detection before fine-tuning on event targets.

**Key Techniques:**
1. **Temporal Shift Module (TSM) on EfficientNet** — Inserts temporal channel shifts into 2D CNN feature maps to model motion across frames at minimal parameter cost; critical when TSM disrupts plain ImageNet init — auxiliary ball-detection pretraining stabilizes the TSM layers.
2. **Ball-detection pretraining on SoccerNet** — Pretrained backbone+TSM on SoccerNet ball tracking (20k+ annotated frames) before event spotting fine-tuning; provides richer sports-domain feature initialization than ImageNet alone.
3. **1D UNet head on temporal feature sequences** — Instead of a per-frame classification head, a 1D encoder-decoder (UNet) over the time axis leverages surrounding-frame context for each timestamp prediction; outperforms simple dense heads for event spotting.
4. **Manifold Mixup** — Applied mixup at the image-feature (manifold) level between different video clips; worked well for this time-series-of-frames setup, harder to do at pixel level.
5. **Label design: 1D Gaussian heatmap + binary eval mask** — Event labels are ±5-frame Gaussian heatmaps on a binary mask indicating the evaluation window; clean foreground/background separation improved convergence vs. hard 0/1 labels.

**How to Reuse:**
- TSM is a drop-in temporal module for any 2D CNN in video tasks — adds minimal parameters while giving multi-frame context; experiment before committing to full 3D CNNs.
- For action spotting, a 1D UNet on temporal feature sequences is a strong and efficient baseline that outperforms single-frame classifiers.
- Pretraining on a related auxiliary task (ball detection → event detection) is highly effective when primary event labels are sparse.

---

## 3. Shopee - Price Match Guarantee (2021) — 6th Place

**Task type:** Multimodal product matching — given product images and text titles, retrieve all catalog items that are the same product
**Discussion:** [https://www.kaggle.com/c/shopee-product-matching/discussion/238010](https://www.kaggle.com/c/shopee-product-matching/discussion/238010)

**Approach:** Separate ArcFace image embeddings and DistilBERT text embeddings concatenated and matched via cosine nearest-neighbors; second-stage XGBoost re-ranker on additional graph/meta features.

**Key Techniques:**
1. **ArcFace (CurricularFace) for image metric learning** — Metric learning with ArcFace loss trained on product images; produces tightly clustered per-product embeddings far superior to softmax for retrieval tasks.
2. **Separate multimodal encoders** — English and Indonesian DistilBERT for titles; ViT-384, Swin Transformer, and EfficientNet-B4 for images; separate training outperformed cross-modal fusion; embeddings then concatenated.
3. **Weighted Database Augmentation** — After initial matching, update each item's embedding as a weighted mean of itself and its nearest match (weight ∝ cosine distance); self-supervised regularization that propagates matched-pair information without explicit labels.
4. **Two-stage matching: embedding → XGBoost re-ranker** — Retrieve top-K via cuML NearestNeighbors (GPU), then XGBoost re-scores using cosine distance, text length, Levenshtein distance, image file size, and graph connectivity features.
5. **GroupKFold for second-stage training** — Second-stage model validated with GroupKFold on product groups, preventing leakage since items from the same group appear in both train and validation otherwise.

**How to Reuse:**
- For entity matching/deduplication, combine metric-learning embeddings per modality separately, then fuse via concatenation — simpler than cross-modal attention and often equally effective.
- Two-stage retrieval (approximate NN → feature-based re-ranker) is a universal pattern for large-scale matching; cuML GPU NearestNeighbors is a practical drop-in.
- Weighted mean embedding updates (model predictions as soft labels for database augmentation) is a self-supervised trick applicable to any retrieval competition with guaranteed same-item pairs.

---

## 4. RSNA STR Pulmonary Embolism Detection (2020) — 2nd Place

**Task type:** Multi-label classification of pulmonary embolism subtypes from CT pulmonary angiography (CTPA) DICOM sequences
**Discussion:** [https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection/discussion/193401](https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection/discussion/193401)

**Approach:** Five-stage pipeline: (1) 2D CNN slice feature extraction with PE windowing → (2) Transformer sequence model for exam-level labels → (3) Time-Distributed CNN end-to-end → (4) Heart-slice classifier → (5) 3D CNN (video-pretrained) for RV/LV ratio, with a linear stacking model combining outputs.

**Key Techniques:**
1. **2D CNN → Transformer sequence modeling** — Extracted 512-D slice features with ResNeSt50 using PE-specific DICOM windowing (WL=100, WW=700), then fed sequences into a 4-layer Transformer; enables learning temporal dependencies across hundreds of CT slices without 3D convolutions.
2. **Generalized mean pooling for feature aggregation** — Used generalized mean pooling (not global average pooling) to aggregate spatial features; better discriminative representations for the slice-level PE detection head.
3. **Hand-labeled heart-slice classifier** — Manually labeled ~1,000 CT scans to train a heart-slice detector (EfficientNet-B1, AUC 0.998); restricting RV/LV prediction to heart slices only dramatically improved cardiac ratio model quality.
4. **Video-pretrained 3D CNN (ip-CSN-101) for cardiac structure** — Used a 101-layer channel-separated network pretrained on 65M Instagram videos (VMZ) for 3D cardiac classification; video pretraining transfers better than ImageNet for 3D medical volumes.
5. **Weighted BCE loss by positive PE slice proportion** — Weighted the loss from each example by the proportion of positive PE slices; aligned training signal with the competition's per-image weighted metric.

**How to Reuse:**
- The 2D CNN → Transformer sequence pipeline is the standard approach for 3D medical imaging when full volumes don't fit in memory; generalizes to brain MRI, colonoscopy, and pathology.
- For complex multi-label CT tasks, build specialized sub-models per label group (PE presence vs. cardiac ratio) rather than a single multi-output model — much easier to optimize each separately.
- When anatomical annotation is feasible (1–2 days for 1,000 scans), hand-labeled localization models (e.g., heart-slice classifier) consistently outperform attention-based automatic localization.

---

## 5. HuBMAP - Hacking the Kidney (2021) — 1st Place

**Task type:** Instance segmentation of glomeruli in high-resolution kidney histology whole-slide images
**Discussion:** [https://www.kaggle.com/c/hubmap-kidney-segmentation/discussion/238198](https://www.kaggle.com/c/hubmap-kidney-segmentation/discussion/238198)

**Approach:** Two-class segmentation (healthy vs. non-functional glomeruli) using external data re-annotated by hand with both classes. EfficientNet-UNet + Swin-FPN ensemble with careful exclusion of a pathological slide that corrupted CV-LB correlation.

**Key Techniques:**
1. **Two-class annotation strategy** — Training set labeled only healthy glomeruli; the team identified the need to distinguish healthy from non-functional (FC) glomeruli and re-annotated external data with both classes — a fundamentally different model than single-class baselines.
2. **Pathological slide exclusion (d488c759a)** — Identified one slide with mixed healthy/non-functional annotations that corrupted LB scores; correctly excluding it from LB interpretation allowed trusting CV and accurate model selection.
3. **Swin Transformer backbone + FPN skip connections** — Combined Swin Transformer backbone with FPN skip connections in the UNet; multi-scale features from hierarchical attention captured variable-size glomeruli better than CNN-only backbones.
4. **Human+AI hybrid re-annotation** — Used AI predictions to guide human annotators in adding missing glomeruli in external HPA datasets; scales better than pure manual annotation for large WSI data.
5. **Architecturally diverse ensemble (CNN + Transformer)** — EfficientNet-UNet + Swin-FPN; architectural diversity (CNN vs. Transformer inductive bias) was more valuable than ensembling same-architecture checkpoints.

**How to Reuse:**
- Always audit whether training labels match what the test metric actually evaluates — label schema mismatches (healthy-only vs. all glomeruli) are a major source of CV-LB gap in pathology competitions.
- Identify and handle "poison" slides that corrupt CV-LB correlation before tuning; one outlier can invalidate hundreds of experiments.
- Hybrid human+AI annotation loops (train baseline → find gaps → correct → retrain) are practical for WSI competitions; 2–3 cycles often outperform simply buying more annotations.

---

## 6. RSNA Pneumonia Detection Challenge (2018) — 2nd Place

**Task type:** Bounding-box detection of pneumonia opacities in chest X-rays; 3-class (normal / non-pneumonia opacity / pneumonia)
**Discussion:** [https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/discussion/70427](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/discussion/70427)

**Approach:** Single RetinaNet detector with SE-ResNeXt101 backbone at 512×512, multi-task 3-class classification + bounding-box regression, 4-fold cross-validation with ImageNet pretraining. Solution by Dmytro Poplavskiy.

**Key Techniques:**
1. **SE-ResNeXt101 backbone for RetinaNet** — Squeeze-and-excitation channel attention with ResNeXt grouping outperformed standard ResNets; SE recalibrates feature maps toward diagnostically relevant regions in chest X-rays.
2. **Multi-task 3-class classification + localization** — Auxiliary 3-class classification head (normal / non-pneumonia opacity / pneumonia) jointly trained with RetinaNet detection head; global image context improved box precision.
3. **Small bounding-box anchor head** — Extra output head with smaller anchor scales for small pneumonia patches that standard RetinaNet anchors systematically missed.
4. **ReduceLROnPlateau scheduling** — Patience=4, factor=0.2 LR reduction; more robust than fixed schedules on small medical datasets where loss plateau timing is unpredictable.
5. **4-fold CV at 512×512** — 512px was the sweet spot: lower resolutions degraded performance, full 2000+px resolution was impractical with heavy backbones.

**How to Reuse:**
- For single-class medical detection, add a multi-class auxiliary classification head — provides global context (is there pathology?) that improves box precision without significant overhead.
- SE-type backbones (SE-ResNeXt, SENet154) consistently outperform plain ResNets on medical imaging detection; use as default backbone choice.
- Add dedicated small-anchor heads when the target pathology can appear at multiple scales; don't rely solely on FPN to handle all scales.

---

## 7. Cornell Birdcall Identification (2020) — 3rd Place

**Task type:** Multi-label audio classification of bird species from 5-second soundscape recordings (265 species + nocall)
**Discussion:** [https://www.kaggle.com/c/birdsong-recognition/discussion/183199](https://www.kaggle.com/c/birdsong-recognition/discussion/183199)

**Approach:** CNN ensemble on pre-saved Mel spectrograms with extensive manual data cleaning (20k files), domain-specific audio augmentations, and soft-label BCEWithLogitsLoss. Final ensemble of 6 models including a 2.5-second model and a 150-class restricted model.

**Key Techniques:**
1. **Pre-saved Mel spectrograms** — Converted all audio to Mel spectrograms saved as image files before training; eliminated real-time decoding bottleneck on weak hardware and enabled standard image augmentation pipelines.
2. **Manual data cleaning (20,000 files)** — Manually reviewed all training files to remove segments without the target bird calling; automated filtering failed because training data had long silence/background-noise-only segments.
3. **Power-law contrast augmentation** — Raised spectrogram values to a random power (0.5–3.0); at 0.5, background noise becomes more prominent; at 3.0, quiet sounds fade — simulates variable recording distances.
4. **Upper-frequency suppression** — With 50% probability, lowered amplitude of upper frequencies; mimics real-world acoustic attenuation where high frequencies fade faster with distance from the bird.
5. **Soft background labels (BCEWithLogitsLoss)** — Primary bird label = 1.0, background/secondary species = 0.3; soft multi-label targets improved F1 vs. hard 0/1 labels for co-occurring species.

**How to Reuse:**
- Pre-save spectrograms as images before any audio classification training — removes bottleneck and enables albumentations/image augmentation libraries out of the box.
- Manual data cleaning often outperforms automated filtering for soundscape audio; spend time on a random sample first to understand label quality before building automated filters.
- Power-law contrast and upper-frequency suppression are domain-specific augmentations for any distance-sensitive acoustic task (wildlife monitoring, underwater acoustics, SONAR).

---

## 8. Market Basket Analysis (2017) — 2nd Place

**Task type:** Tabular product recommendation — predict which products a customer will re-order in the next Instacart order, optimized for F1
**Discussion:** [https://www.kaggle.com/c/basket-analysis/discussion/38143](https://www.kaggle.com/c/basket-analysis/discussion/38143)

**Approach:** CatBoost primary model with LDA/NMF latent features and a custom F1-optimization algorithm accounting for correlation and uncertainty, applied over feature-engineered purchase probability estimates.

**Key Techniques:**
1. **Exact F1 Maximization via joint probability matrix** — Implemented O(n³) exact F1 maximization using a joint probability matrix of product co-purchase; improved F1 by 0.001–0.0015 over the sub-optimal greedy O(n²) approach.
2. **Uncertainty-based probability correction** — Multiplied purchase probabilities by `1.6^(1/n²)` where n is conditional basket size; when probability estimates are uncertain (large baskets), correction pushes toward "include all" which optimizes F1.
3. **CatBoost "accurate" mode** — CatBoost with accurate mode gave 0.0003–0.0005 improvement over LightGBM/XGBoost; XGBoost "accurate" (vs. approximate) gave an additional 0.0015 lift.
4. **LDA and NMF latent factors** — Added LDA and NMF features capturing latent purchase patterns; +0.0005–0.001 improvement as complementary signal to tabular engineered features.
5. **Conditional basket size features** — "Given a user buys this product, how many total products do they buy in this order?" — key signal for calibrating re-order likelihood estimates.

**How to Reuse:**
- For any multi-label recommendation task with F1 as metric, implement the exact F1 maximization algorithm rather than greedy threshold search — the gain is measurable.
- Uncertainty-scaling corrections based on basket size or prediction confidence improve F1 for sparse multi-label decisions where probability calibration is poor.
- LDA/NMF latent features add complementary signal to tree models on transactional data; rarely hurt and often help 0.001+.

---

## 9. HMS - Harmful Brain Activity Classification (2024) — 2nd Place

**Task type:** Multi-class classification of EEG patterns (seizure, LPD, GPD, LRDA, GRDA, Other) from 50-second EEG recordings
**Discussion:** [https://www.kaggle.com/c/hms-harmful-brain-activity-classification/discussion/492254](https://www.kaggle.com/c/hms-harmful-brain-activity-classification/discussion/492254)

**Approach:** Multimodal ensemble of 6 models: X3D-L 3D-CNN for spectrograms, EfficientNetB5 treating reshaped raw EEG as 2D images, HGNetB5 variant, and a hybrid model combining both. Two-stage training: all data first, then high-confidence samples only.

**Key Techniques:**
1. **X3D-L (3D-CNN) for EEG spectrograms** — After double-banana montage + 0.5–20Hz bandpass filter + STFT, spectrograms are processed by X3D-L (a lightweight video CNN); 3D convolutions capture joint frequency-time relationships that 2D CNNs cannot without explicit temporal pooling.
2. **EfficientNetB5 treating raw EEG as 2D images** — Reshaped EEG from (16 channels, 10000 samples) to (160, 1000) array and treated as an image; avoids 2D-CNN channel-padding limitation and preserves channel positional information.
3. **Two-stage training: all data → high-confidence only** — Stage 1: 15 epochs with loss weighted by voters_num/20; Stage 2: 5 epochs on voters_num≥6 samples only at lower LR; dramatically improved reliability of the training gradient signal.
4. **Hemisphere-flipping augmentation** — Swapped left and right brain channel data during training; domain-specific augmentation reflecting bilateral brain symmetry, approximately doubling effective data.
5. **Ensemble diversity via filter library variation** — Same architecture trained with MNE-based vs. scipy-based bandpass filtering as separate models; different signal processing implementations produce diverse enough predictions for meaningful ensemble gains.

**How to Reuse:**
- For EEG/physiological signals, try reshaping multi-channel time-series as 2D images and applying standard image CNNs — avoids custom temporal architectures and often matches LSTM/Transformer performance.
- Two-stage training (all data then high-confidence subset) is principled for label-noisy competitions; first stage gives stable initialization, second stage refines on clean signal.
- Bilateral/symmetry augmentation (swap left/right electrode channels, paired sensor channels) applies broadly to any symmetric biosensor data.

---

## 10. TensorFlow - Help Protect the Great Barrier Reef (2022) — 3rd Place

**Task type:** Underwater video object detection (crown-of-thorns starfish)
**Discussion:** [https://www.kaggle.com/c/tensorflow-great-barrier-reef/discussion/307707](https://www.kaggle.com/c/tensorflow-great-barrier-reef/discussion/307707)

**Approach:** Pure YOLOv5l6 ensemble with video-ID-based cross-validation, 90° rotation augmentation, and a custom "attention area" temporal post-processor. "Nothing fancy" — YOLOv5 with correct CV strategy and targeted augmentation.

**Key Techniques:**
1. **Video-ID fold splitting** — Split by video_id (3 folds) rather than subsequence; video-level splits gave stable CV positively correlated with public and private LB, whereas subsequence splits produced noisy scores.
2. **Random 90° rotation augmentation** — Added random 90° rotations on top of standard YOLOv5 augmentation; gave ~0.02 boost on both CV and LB, and increased diversity between ensemble members trained with different rotations.
3. **Train on annotated frames only (0% background)** — Counter to YOLOv5's recommended 10% background, training on only annotated frames gave the best results for this rare-object detection task.
4. **Multiscaling ±50%** — Enabled YOLOv5's built-in multi-scale training (±50% scale variation); particularly important for anchor-based detectors when targets appear at many distances/scales in video.
5. **"Attention Area" temporal post-processor** — Custom tracker: detection in frame N boosts confidence of nearby bounding boxes in frames N±k; simpler than Kalman filter tracking but effective for a smoothly moving target.

**How to Reuse:**
- For video object detection, always split and validate by video/sequence ID — frame-level splits leak temporal context and overstate CV performance.
- 0% background (annotation-only frames) often beats the recommended 10% background when targets are rare and visually distinctive — test both.
- A simple temporal confidence-propagation post-processor (boost confidence near prior detections) often outperforms off-the-shelf trackers for objects that move predictably between frames.

---

## 11. TGS Salt Identification Challenge (2018) — 9th Place

**Task type:** Binary segmentation of salt deposits in seismic reflection images (101×101 pixels)
**Discussion:** [https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/69053](https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/69053)

**Approach:** Single SENet154 model with 10-fold cross-validation and reflective padding, trained with AdamW + Noam scheduler + cutout, refined with SWA. Key innovation: symmetric Lovász loss.

**Key Techniques:**
1. **Symmetric Lovász Loss** — Modified Lovász hinge to be symmetric: `(lovász_hinge(preds, targets) + lovász_hinge(-preds, 1-targets)) / 2`; gave +0.008 public LB and +0.02 private LB — the single largest single technique improvement.
2. **SENet154 backbone** — Upgraded from SE-ResNeXt50 to SENet154; improved single-fold CV from 0.87x to 0.882/0.869 (CV/LB).
3. **Stochastic Weight Averaging (SWA)** — Applied SWA over best loss, pixel accuracy, metric, and last checkpoint; gave consistent +0.004 improvement.
4. **AdamW + Noam scheduler** — Weight-decoupled AdamW with Noam warmup-then-decay schedule; more stable convergence than Adam+StepLR for segmentation.
5. **Reflective padding + 10-fold CV** — Reflective boundary padding (not zero) avoids edge artifacts in seismic images; 10-fold CV for inference raised score from 0.882 to 0.890.

**How to Reuse:**
- Symmetric Lovász loss is a general technique for binary segmentation with imbalanced classes — implement it before any architectural experiments.
- SWA is a near-free ensemble adding 0.003–0.005 to segmentation scores; always add as the final training stage.
- For small-image segmentation, simple decoders (single conv + transposed conv) match complex attention decoders (DANet, OCNet) — don't over-engineer the decoder.

---

## 12. Feedback Prize - English Language Learning (2022) — 1st Place

**Task type:** Regression — predict 6 writing proficiency scores from student essays; metric MCRMSE
**Discussion:** [https://www.kaggle.com/c/feedback-prize-english-language-learning/discussion/369457](https://www.kaggle.com/c/feedback-prize-english-language-learning/discussion/369457)

**Approach:** DeBERTa-v3-large ensemble with rank loss, back-translation pretraining, and pseudo-label pretraining from Feedback Prize 1 data; Optuna-tuned ensemble weights with careful OOF-based validation.

**Key Techniques:**
1. **Rank Loss auxiliary objective** — Added pairwise ranking loss on top of regression loss; rank loss consistently improved both CV and private LB across all model variants — ordinal relationships between scores matter as much as absolute values.
2. **Back-translation pretraining** — Pretrained DeBERTa-v3-large on back-translated essays before task fine-tuning; corpus-level data augmentation that adds linguistic diversity without changing semantic content.
3. **Pseudo-label pretraining from Feedback Prize 1** — Generated pseudo-labels on the larger Feedback Prize 1 dataset and pretrained on it; models with more background knowledge (larger effective corpus) consistently outperformed on both CV and private LB.
4. **DeBERTa-v3-large as dominant backbone** — Only DeBERTa-v3-base and v3-large worked well; all other backbones (RoBERTa, DeBERTa-v2, etc.) were significantly worse; model family selection is the single most impactful decision.
5. **Optuna ensemble tuning with held-out validation** — Used Optuna to tune per-model per-target blending weights on OOF; however, Optuna-tuned weights overfit OOF on private LB — hand-tuned weights were more stable.

**How to Reuse:**
- For writing quality regression, add pairwise rank loss alongside regression loss — reliably improves ranking accuracy even when raw RMSE doesn't change much.
- Pseudo-labeling from related competition datasets (prior editions, same domain) is highly effective and legal — build a pipeline for it early.
- When Optuna-tuning ensemble weights on OOF, validate the weights on a held-out fold — Optuna ensemble weight search easily overfits OOF.

---

## 13. NBME - Score Clinical Patient Notes (2022) — 1st Place

**Task type:** NLP span extraction — identify character spans in clinical notes corresponding to medical scoring features
**Discussion:** [https://www.kaggle.com/c/nbme-score-clinical-patient-notes/discussion/323095](https://www.kaggle.com/c/nbme-score-clinical-patient-notes/discussion/323095)

**Approach:** DeBERTa-based token classification with preprocessing for inconsistent annotations, GRU on top of transformer for sequence-dependent span prediction, adversarial training (AWP), and pseudo-labeling from 10× unlabeled clinical notes.

**Key Techniques:**
1. **RNN for sequence-dependent annotation modeling** — Observed annotators missed repeated feature occurrences (sequence dependency); trained a GRU on top of transformer features to model this — first-occurrence annotation is more reliable than nth-occurrence.
2. **Medical abbreviation normalization** — Mapped common clinical abbreviations (FHx→FH, PMHx→PMH, SHx→SH) before tokenization; reduced out-of-vocabulary fragmentation in clinical shorthand for web-pretrained transformers.
3. **Pseudo-labeling from 10× unlabeled notes** — The competition provided 10× more unlabeled clinical notes; pseudo-labeled them and pretrained on the larger corpus before gold-label fine-tuning; highest-ROI technique in the solution.
4. **Adversarial training (AWP/FGM)** — Applied adversarial weight perturbation during fine-tuning; consistently improved generalization on unseen clinical note styles across all DeBERTa variants.
5. **Lowercase normalization** — Converted all text to lowercase; clinical notes had semantically meaningless case variation that caused unnecessary vocabulary fragmentation.

**How to Reuse:**
- For span extraction with inconsistent annotations, model the annotation process itself (e.g., GRU conditioning on previous spans) rather than treating each span independently.
- When in-domain unlabeled text is available, pseudo-label it and use for domain-adaptive pretraining — it's the highest-ROI augmentation for clinical NLP tasks.
- Clinical text preprocessing (abbreviation expansion, case normalization) is necessary for web-pretrained transformers; define the minimal normalization set before running any baseline.

---

## 14. UW-Madison GI Tract Image Segmentation (2022) — 1st Place

**Task type:** Multi-class segmentation of stomach, small bowel, and large bowel from MRI slices
**Discussion:** [https://www.kaggle.com/c/uw-madison-gi-tract-image-segmentation/discussion/337197](https://www.kaggle.com/c/uw-madison-gi-tract-image-segmentation/discussion/337197)

**Approach:** Two-stage pipeline — YOLOv5-based positive/negative slice classifier → large backbone UNet segmentation on positive slices only. Ensemble of 2.5D (5-slice input) and 3D (DynUNet) models.

**Key Techniques:**
1. **Two-stage positive/negative slice filtering** — Stage 1 classifies each MRI slice as containing anatomy or not; stage 2 runs expensive segmentation only on positive slices; doubles effective compute budget for the hard cases and reduces class imbalance.
2. **YOLOv5 crop for body signal normalization** — YOLOv5 detects and crops the abdominal region, removing arm signals that cause min-max normalization failure from B1 field inhomogeneity in abdominal MRI.
3. **Backbone scaling: EfficientNet-B4 → L2 → ConvNeXt XL → Swin-Large** — Systematically increased backbone size in stage 2; each step improved validation Dice (B4: 0.8011 → L2: 0.8349). Large backbones dominate segmentation quality.
4. **UperNet decoder with CE + Dice loss (1:1)** — Combined cross-entropy and Dice loss equally; UperNet's multi-scale FPN aggregation improves detection of both small and large anatomical structures simultaneously.
5. **2.5D + 3D DynUNet ensemble** — 3D context improved slice continuity; 2.5D (5-slice) + 3D ensemble consistently outperformed either alone; SWA applied within 3D model training.

**How to Reuse:**
- Two-stage filtering (detect positive slices first, then segment) is a universal pattern for volumetric medical imaging — reduces imbalance and focuses compute on relevant slices.
- YOLOv5 as a preprocessing crop tool normalizes input regions and removes off-target signal; useful beyond its role as a final detector.
- When backbone scaling consistently improves (B4→L2→ConvNeXt XL), invest in 2.5D+3D ensemble before trying more exotic architectures.

---

## 15. G2Net Gravitational Wave Detection (2021) — 2nd Place

**Task type:** Binary classification of gravitational wave signals in time-series data from LIGO/Virgo detectors; metric AUC
**Discussion:** [https://www.kaggle.com/c/g2net-gravitational-wave-detection/discussion/275341](https://www.kaggle.com/c/g2net-gravitational-wave-detection/discussion/275341)

**Approach:** Multiple trainable frontend CNN architectures (CWT-CNN, 1D-CNN, spectrogram 2D-CNN) with bandpass-filtered inputs, combined via Ridge regression stacking with greedy model selection over 20 diverse models.

**Key Techniques:**
1. **Trainable frontend signal processing** — Used a learned first layer (trainable filters) instead of fixed CWT or bandpass preprocessing; trainable frontends consistently outperformed fixed signal processing, suggesting optimal representations differ from classical engineering choices.
2. **Physics-informed bandpass filtering** — Applied bandpass [16–512 Hz] for CWT/2D-CNN networks, [30–300 Hz] for 1D-CNN; domain knowledge constrains the signal space to physically plausible gravitational wave frequencies.
3. **Wave-domain augmentations** — Gaussian noise (2D-CNN) and flipped wave amplitude (1D-CNN) were the only augmentations that consistently improved AUC; most spectrogram-domain augmentations did not help.
4. **Soft pseudo-labeling with label smoothing** — Re-trained on continuous (soft) pseudo-labels from test set predictions; label smoothing during pseudo-label generation improved AUC by ~0.001.
5. **Ridge regression stacking with greedy model selection** — Kept OOF predictions from all experiments; Ridge regression stacking on 20 diverse models (CV 0.88283, Private LB 0.8829) outperformed 10-model (0.8827) or 5-model (0.8825) subsets. Greedy forward selection identified the optimal 20-model subset.

**How to Reuse:**
- For signal classification tasks (audio, seismic, gravitational wave), try a trainable frontend layer before committing to fixed preprocessing — it often learns better feature representations than hand-crafted transforms.
- Keep all OOF predictions from experiments and build a Ridge stacking meta-model at the end — minimal additional cost with reliable ensemble gains.
- Greedy forward model selection (add models one by one if they improve ensemble CV) prevents overfitting Ridge stacking weights on small validation sets.

---

## 16. Eedi - Mining Misconceptions in Mathematics (2024) — 1st Place

**Task type:** NLP information retrieval — given a multiple-choice math question with a wrong answer, retrieve the most likely mathematical misconception from 2,587 options; metric MAP@25
**Discussion:** [https://www.kaggle.com/c/eedi-mining-misconceptions-in-mathematics/discussion/551402](https://www.kaggle.com/c/eedi-mining-misconceptions-in-mathematics/discussion/551402)

**Approach:** Two-stage retriever-reranker pipeline using Qwen2.5-32B-Instruct fine-tuned with LoRA + 4-bit quantization. Retriever uses last-token (EOS) embeddings with contrastive loss; reranker uses a single-tower supervised model. Staged training: synthetic GPT-4o-mini data → MalAlgoQA → competition training data.

**Key Techniques:**
1. **Large generative LLM as embedding model** — Qwen2.5-32B-Instruct's last-token (EOS) embedding with LoRA fine-tuning outperformed standard sentence transformers for this specialized mathematical reasoning retrieval task; domain knowledge in the pretrained LLM outweighed the embedding specialization of smaller models.
2. **Staged training on data of increasing quality** — Sequential training: (1) GPT-4o-mini synthetic misconception data, (2) MalAlgoQA academic dataset, (3) competition training data; different data quality/format requires staged training to avoid format conflicts and quality dilution.
3. **Hard negative mining for reranker** — Retrieved top-150 candidates and sampled 25 hard negatives per training example; hard negatives (similar-but-wrong misconceptions) are essential for a reranker that distinguishes fine-grained math reasoning differences.
4. **LoRA fine-tuning with 4-bit quantization** — LoRA (r=16, alpha=32) across q/k/v projections with BitsAndBytes 4-bit quantization makes 32B model fine-tuning feasible on competition hardware.
5. **Reranker: supervised single-tower for top-K re-ordering** — Reranker maps (question, candidate misconception) token embeddings to a similarity scalar; single-tower is simpler than cross-encoder but captures query-candidate interaction better than pure two-tower dot product.

**How to Reuse:**
- For specialized retrieval in technical domains (math, medicine, law), fine-tuning a large generative LLM as an embedding model outperforms off-the-shelf sentence transformers — domain knowledge in the LLM is more valuable than embedding specialization.
- Staged curriculum training (synthetic → semi-supervised → gold labels) is reliable when gold labels are scarce; always start noisiest/most general and end with most specific.
- LoRA + 4-bit quantization is the standard approach for fine-tuning 30B+ models in competition settings — accuracy loss vs. full fine-tuning is minimal for retrieval tasks.

---

## 17. NBME - Score Clinical Patient Notes (2022) — 4th Place

**Task type:** NLP span extraction — same competition as entry #13
**Discussion:** [https://www.kaggle.com/c/nbme-score-clinical-patient-notes/discussion/322799](https://www.kaggle.com/c/nbme-score-clinical-patient-notes/discussion/322799)

**Approach:** Ensemble of 4 token classification models (DeBERTa-v3-large, DeBERTa-v2-xlarge, DeBERTa-v2-xxlarge) with MLM pretraining, SmoothFocalLoss, pseudo-labeling, and a character-level DeBERTa+GRU model for diversity.

**Key Techniques:**
1. **MLM fine-tuning before span extraction** — Continued masked language model pretraining (mask rate 0.10–0.15) on the clinical notes corpus before span extraction fine-tuning; domain-adaptive pretraining improved F1 by ~0.002–0.005.
2. **SmoothFocalLoss for heavy class imbalance** — Used smoothed focal loss (rather than BCE) to down-weight easy negative tokens; clinical notes have many non-entity tokens, making focal loss a natural fit.
3. **Character-level span classification with GRU head** — DeBERTa at character level + 4-layer GRU; character classification catches span boundary errors from tokenizer misalignment, especially for hyphenated/abbreviated medical terms.
4. **Text augmentations: mask + replace** — Mask augmentation (randomly mask feature_text tokens) and replace augmentation (replace with synonyms); both improved generalization on unseen clinical note styles.
5. **Per-case threshold tuning + rule-based postprocessing** — Separate thresholds per clinical case type; rule-based postprocessing fixed common tokenization span errors (off-by-one, hyphenated numbers, leading newlines) — each rule added 0.0001–0.0005.

**How to Reuse:**
- Domain-adaptive MLM pretraining on in-domain text is a reliable +0.002–0.005 improvement for NLP span extraction with specialized vocabulary; always do it before final fine-tuning.
- Adding a character-level model provides complementary signal to subword-tokenized models; especially valuable when span boundaries fall at subword token boundaries (medical abbreviations, symbols).
- Budget time for rule-based span postprocessing at the end of development — fixing obvious off-by-one and tokenization artifacts adds small but reliable gains.

---

## 18. Prostate cANcer graDe Assessment (PANDA) Challenge (2020) — 11th Place

**Task type:** Multi-class classification of prostate cancer Gleason grade from whole-slide histopathology images (6 ISUP grades); metric quadratic weighted kappa
**Discussion:** [https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion/169205](https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion/169205)

**Approach:** EfficientNet + DenseNet ensemble on tiled 256×256 patches (42 tiles per slide) using white-padding tile extraction (Iafoss method), trained on TPU with MSE + BCE combined loss, 5-fold cross-validation.

**Key Techniques:**
1. **Tile-based WSI processing (42× 256×256)** — Extracted 42 non-background tiles per slide using white-padding tile extraction; converts arbitrarily large gigapixel images into fixed-size tensors processable by standard CNNs without downsampling to uselessness.
2. **MSE + BCE combined loss for ordinal regression** — MSE on Gleason grade (continuous) + BCE for grade presence; combined loss outperformed ordinal-only or classification-only objectives on quadratic kappa metric.
3. **TPU training via TF-records** — Full training on Kaggle/Google TPU using TF-Record format; TPU + properly formatted tf-records gave 5–10× speedup over GPU for tile-based pathology workflows.
4. **5-fold cross-validation with label consensus checking** — StratifiedKFold on grade labels with verification that external data labels agreed with official labels; label conflicts resolved in favor of official annotations.
5. **EfficientNet + DenseNet ensemble** — Architectural diversity (different inductive biases between EfficientNet scaling and DenseNet dense connectivity) provided complementary predictions on ambiguous grade cases.

**How to Reuse:**
- Tile-based WSI processing (sample N non-background patches) is the standard approach for pathology competitions; use the Iafoss/PANDA tile extraction as a reference implementation.
- Combine MSE and BCE losses for ordinal regression with kappa metric — neither alone optimizes kappa as well as the combination.
- TPU training via tf-records gives significant speedup for tile-based pathology workloads; set up the data pipeline early rather than as a late optimization.

---

## 19. Riiid Answer Correctness Prediction (2021) — 4th Place

**Task type:** Sequential prediction — given a student's history of question interactions, predict whether they answer the next question correctly; metric AUC
**Discussion:** [https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/210171](https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/210171)

**Approach:** Single encoder-decoder Transformer (SAINT/SAKT-inspired) with ContinuousEmbedding for temporal features, time-aware attention decay, trained on TPU via TensorFlow with random sequence cutting/padding. No ensemble — single model at 4th place.

**Key Techniques:**
1. **ContinuousEmbedding for time/difficulty features** — Maps continuous values (time lag, elapsed time, difficulty, popularity) to a weighted sum of consecutive embedding vectors; produces smooth embeddings where similar values have similar representations, avoiding sharp discontinuities of binned embeddings.
2. **Time-aware weighted attention** — Decays attention coefficients by `dt^(-w)` where dt is timestamp difference and w is a trainable non-negative parameter per head; penalizes attending to distant-in-time interactions, improving convergence speed and performance.
3. **Encoder-decoder with causal masks** — Encoder receives all input features; decoder excludes user-answer-related features to prevent leakage; causal masking prevents current position from attending to future; clean feature separation without architectural complexity.
4. **Random sequence cutting and padding** — Randomly cut sequences to variable lengths and padded uniformly during training; ensures all positions in the sequence receive training gradient, not just the final positions.
5. **Question difficulty and popularity as global features** — Per-question correct response rate and appearance count computed from the full training corpus; global question statistics are important signals beyond individual student history.

**How to Reuse:**
- ContinuousEmbedding (weighted sum of neighboring embedding vectors) generalizes to any continuous feature in Transformers (time deltas, prices, durations) — use it to replace binning for continuous inputs.
- Time-aware attention decay (subtract `dt^(-w)` from attention logits) is a simple modification improving any task where recency of interactions matters.
- For knowledge tracing or sequential recommendation, the encoder-decoder feature split (encoder sees all, decoder excludes target-related features) is a principled leakage-prevention mechanism.

---

## 20. Human Protein Atlas Image Classification (2019) — 3rd Place

**Task type:** Multi-label fluorescence microscopy image classification of subcellular protein localization (28 classes, severely imbalanced); metric macro F1
**Discussion:** [https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/77320](https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/77320)

**Approach:** Ensemble of ResNet34 (512×512), InceptionV3 (1024×1024), and SE-ResNeXt50 (1024×1024) with per-image normalization, AutoAugment-style random search, Focal loss (gamma=2), and frequency-proportional threshold selection.

**Key Techniques:**
1. **Per-image mean/std normalization** — Normalized each image to its own mean and standard deviation; official and external datasets had very different intensity distributions, making global normalization harmful.
2. **AutoAugment via random search** — Random search over augmentation policies (instead of RL-based original) found effective augmentations for this specific dataset without RL training cost.
3. **Focal loss (gamma=2) for extreme class imbalance** — Down-weighted easy negative predictions for rare localization classes; essential for learning patterns with <100 training examples.
4. **Majority-class early stopping** — Stopped training when F1 for the majority class (class 0, Nucleoplasm) began declining rather than when macro F1 improved; prevented over-tuning toward rare classes at the expense of majority class performance.
5. **Frequency-proportional threshold selection** — For each class, set threshold so positive prediction proportion ≈ positive training proportion; a fast calibration approach that outperformed cross-validated threshold search on this imbalanced dataset.

**How to Reuse:**
- For multi-label classification with mixed external data sources, always normalize per-image (not per-dataset); global normalization degrades performance when data sources differ in intensity characteristics.
- Stop on majority-class metric stability (not macro metric) when training multi-label models with severe class imbalance; prevents rare-class over-tuning that hurts overall macro F1.
- Frequency-proportional threshold selection is a fast, reliable alternative to exhaustive threshold search for imbalanced multi-label tasks.

---

## 21. Google Brain - Ventilator Pressure Prediction (2021) — 9th Place

**Task type:** Time-series regression — predict airway pressure at each timestamp for mechanical ventilator breath cycles; metric MAE
**Discussion:** [https://www.kaggle.com/c/ventilator-pressure-prediction/discussion/285353](https://www.kaggle.com/c/ventilator-pressure-prediction/discussion/285353)

**Approach:** 4-layer LSTM with skip connections, trained on extensively feature-engineered tabular data including aggregations over R, C, rank, and rounded u_in. The key differentiator was feature engineering, not architecture complexity.

**Key Techniques:**
1. **Magic aggregation features (R, C, rank, rounded u_in)** — Aggregated statistics grouping by combinations of R (resistance), C (compliance), breath sequence rank, and rounded u_in value; captured the PID controller's response curves dominating ventilator pressure behavior.
2. **Reversed-order training with forward-order features** — Trained the model to predict from timestamp 80 back to timestamp 1 (reversed direction) while including features computed in forward order; forced the model to learn pressure trajectory from endpoint context.
3. **Quantile transformation of u_in for u_out=0** — Applied quantile transformation to u_in values specifically during the exhale phase (u_out=0); the exhale phase has a different u_in distribution requiring separate normalization treatment.
4. **Negative pressure encoding for exhale phase** — Treated pressure values during u_out=0 as negative in the feature space; physics-informed encoding captured distinct exhale dynamics from inhale.
5. **4-layer LSTM with skip connections** — Simple architecture with skip connections between LSTM layers; skip connections stabilized gradient flow through 4 recurrent layers without gating complexity of deeper designs.

**How to Reuse:**
- For physical system modeling (ventilators, HVAC, engines), physics-informed feature engineering (grouping by physical parameters, separate normalization for different operating phases) often dominates architectural choices.
- Reversed-order sequence training is useful for time-series with strong endpoint predictability; train in both directions and ensemble if direction matters.
- Separate the inhale and exhale phases as fundamentally different regimes for any ventilator/respiratory task — they have different physics requiring different feature representations.

---

## 22. Tweet Sentiment Extraction (2020) — 1st Place

**Task type:** NLP span extraction — given a tweet and sentiment label, extract the word span best supporting the sentiment; metric Jaccard similarity
**Discussion:** [https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/159264](https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/159264)

**Approach:** RoBERTa-large (pretrained on SQuAD2) ensemble with SentimentSampler for balanced batch distribution, SWA for training stability, multi-dropout output layer, and a separate reranking model post-processing step.

**Key Techniques:**
1. **SentimentSampler for within-batch class balance** — Custom sampler equalizing positive/negative/neutral sentiment within each batch; within-batch distribution mattered because sentiment is a direct input that shapes span extraction differently per class, beyond what StratifiedKFold provides.
2. **Stochastic Weight Averaging (SWA)** — Applied SWA because validation score varied ±0.001 per iteration; SWA stabilized to ±0.0001 over 10–50 iterations, enabling confident model selection.
3. **RoBERTa pretrained on SQuAD2** — Used SQuAD2-pretrained RoBERTa rather than raw pretrained weights; QA pretraining aligns the span extraction head with the exact task structure.
4. **Multi-dropout (MDO) output layer** — Applied multiple dropout masks to the output layer and averaged predictions; reduces variance without requiring explicit ensemble training.
5. **Reranking post-model** — Separate reranking model re-scored candidate spans from the base model; added 0.001–0.003 Jaccard on top of base extraction.

**How to Reuse:**
- SWA is especially valuable when validation scores are highly volatile during training — try it before investing in larger ensembles for stabilization.
- For QA-formatted span extraction, start with a model already fine-tuned on SQuAD2 rather than raw pretrained weights; task-aligned initialization improves convergence.
- Class-conditional batch sampling (SentimentSampler, or any conditioning variable) is important when the conditioning signal should be uniformly represented per batch — not just uniformly distributed across folds.

---

## 23. SIIM-ACR Pneumothorax Segmentation (2019) — 3rd Place

**Task type:** Binary segmentation of pneumothorax (collapsed lung) in 1024×1024 chest X-ray DICOM images
**Discussion:** [https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/107981](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/107981)

**Approach:** SE-ResNeXt50 UNet trained on lung-cropped images (576×576) with pseudo-labeled CheXpert and NIH external data. Ensemble of 3 SE-ResNeXt50 models at different scales (704×704, 576×576) with and without pseudo-labels.

**Key Techniques:**
1. **Lung crop preprocessing via ResNet34 UNet** — First trained a lightweight ResNet34 UNet to crop the lung region from 1024×1024 to 576×576; reduces input size and removes irrelevant background while preserving diagnostic detail.
2. **Selective pseudo-labeling from CheXpert** — Used competition model to generate pseudo-labels only for CheXpert samples predicted positive by the model; selected positives filtered label noise from the inherently noisy CheXpert labels.
3. **Balanced pseudo-label ratio** — Kept positive/negative ratio 1:1 in pseudo-labeled training; pseudo samples capped at 50% of normal samples per batch to prevent noisier pseudo-data from dominating the gradient.
4. **CBAM attention in UNet skip connections** — Convolutional Block Attention Module (CBAM) added to skip connections; channel+spatial attention recalibrated features toward pneumothorax-relevant lung regions.
5. **Lovász loss with fixed 0.5 threshold** — Used Lovász loss for training; fixed decision threshold at 0.5 without validation-set search; argued that threshold tuning on public LB was unreliable for private LB and that 0.5 generalized better.

**How to Reuse:**
- Anatomical cropping (train a fast segmenter to isolate the organ of interest, then use crops for all downstream models) is standard for chest X-ray and medical imaging — implement before architectural experiments.
- Pseudo-labeling from public medical datasets (CheXpert, NIH Chest X-ray, MIMIC) provides significant data augmentation for medical segmentation — use model confidence as filter and control positive/negative ratios.
- Fixed thresholds (0.5) often generalize better than tuned thresholds when the validation set is small or unrepresentative of private test distribution — test both approaches before committing.