# Kaggle Past Solutions — SRK Batch 6

**Source:** kaggle.com/sudalairajkumar/winning-solutions-of-kaggle-competitions
**Ingested:** 2026-04-16

---

## 1. TensorFlow 2.0 Question Answering (2019)

**Task type:** NLP — open-domain extractive QA from Wikipedia articles (long-form and short-span answers)
**Discussion:** https://www.kaggle.com/c/tensorflow2-question-answering/discussion/127551

**Approach:** The winning solution fine-tuned large pretrained transformer models (BERT-large and ALBERT) using the "BERT-Joint" framework, which jointly predicts long-answer candidates and short-answer spans in a single forward pass over a 512-token sliding window over Wikipedia pages. Multiple models were trained with different tokenizers and hyperparameters, then ensembled to handle the dual-output prediction task. The key challenge was handling documents far longer than the 512-token context limit while keeping both long-answer selection and short-answer span extraction jointly calibrated.

**Key techniques:**
1. BERT-joint architecture — single model predicting long-answer logits, short-answer start/end spans, and answer type (yes/no/no-answer) simultaneously
2. Sliding window with stride over full Wikipedia articles to handle documents exceeding 512 tokens, retaining context overlap
3. ALBERT-xxlarge fine-tuning as a complementary model with higher parameter efficiency for ensemble diversity
4. Multi-task loss combining cross-entropy over long-answer candidates and span extraction
5. Ensemble of BERT-large and ALBERT-xxlarge models with tuned thresholds for null-answer vs. extractive answer decision

**How to Reuse:**
- Use BERT-joint or DeBERTa-v3 with a dual-head (span + candidate selector) for any document QA task where answers are spans within long passages
- Implement sliding window tokenization with `stride` parameter — this is now standard in HuggingFace `AutoModelForQuestionAnswering` pipelines
- Add a null-answer threshold calibrated on dev set; training without it will over-predict answers on unanswerable questions
- Ensemble 2–3 models of different sizes (base, large, xxlarge) — diversity from parameter count adds more than diversity from random seeds alone

---

## 2. RSNA STR Pulmonary Embolism Detection (2020)

**Task type:** Medical imaging — multi-label classification of pulmonary embolism from 3D CT scan volumes at both image and exam level
**Discussion:** https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection/discussion/194145

**Approach:** Winner Guanshuo Xu built a two-stage pipeline: first, a 2D CNN (SE-ResNeXt family) classified individual CT slices at high resolution using a 3-channel windowed input (current slice + two neighbors), then a bidirectional GRU aggregated slice-level embeddings to produce exam-level multi-label predictions (PE presence, laterality, chronicity, RV/LV ratio). Images were preprocessed to 576×576 by expanding to 640×640 and applying lung localization bounding boxes. The solution required no cross-validation due to dataset size, using a simple train-validation split.

**Key techniques:**
1. 3-channel "neighbor slice" input encoding — current image plus direct axial neighbors in PE Hounsfield window, giving the 2D CNN implicit volumetric context
2. SE-ResNeXt-50 and SE-ResNeXt-101 image encoders trained at high resolution (576×576)
3. Bidirectional GRU sequence model over per-slice embeddings for exam-level label prediction
4. Lung localization with bounding boxes to crop irrelevant anatomy before classification
5. Augmentations: random contrast, shift-scale-rotate, cutout; zero-padding for variable-length scan series

**How to Reuse:**
- The 3-channel neighbor-slice trick is directly portable to any 3D medical volume task where full 3D CNNs are cost-prohibitive — treat adjacent slices as channels
- Stack a bidirectional LSTM or GRU on top of per-frame CNN embeddings for any sequence-of-images classification (video, multi-slice CT, echocardiograms)
- Use Hounsfield window presets (e.g., PE window: C=100, W=700) as fixed preprocessing rather than learning intensity normalization from scratch
- Lung/organ localization with a lightweight detector before the main classifier consistently improves performance in medical imaging

---

## 3. Indoor Location & Navigation (2021)

**Task type:** Signal regression — predicting smartphone (x, y, floor) position in shopping malls from WiFi BSSID signal fingerprints and IMU sensor data
**Discussion:** https://www.kaggle.com/c/indoor-location-navigation/discussion/240176

**Approach:** The winning team "Track me if you can" (Are Haartveit, Dmitry Gordeev, Tom Van de Wiele) built a multi-stage pipeline that first trained sensor models on known waypoints, then applied signal-graph matching across floor plans to propagate predictions. A key element was exploiting the sequential structure of phone trajectories: waypoints within a path are physically constrained, so position predictions were smoothed and corrected using the trajectory graph. The pipeline required 32 GB RAM and over 150 GB disk, with the sensor training step being the slowest component. Expected private leaderboard score was around 1.3–1.5 mean position error.

**Key techniques:**
1. WiFi BSSID fingerprint matching — extracting features from BSSIDs present in both train and test as a locality-sensitive fingerprint representation
2. Trajectory-aware post-processing — using known physical constraints (walking speed, floor transitions) to smooth and correct sequential position predictions
3. Sensor fusion of WiFi RSSI signal strengths with IMU (accelerometer/gyroscope) pedestrian dead reckoning
4. Linear interpolation and Kalman filter for hidden waypoints between labeled observations
5. Floor prediction as a separate classification sub-problem before (x, y) regression

**How to Reuse:**
- WiFi BSSID presence/strength as features for indoor localization is a robust baseline; always filter to BSSIDs seen in both train and test to avoid sparse-feature noise
- Kalman filtering over sequential GPS or sensor predictions is a universally applicable smoothing step for any position regression problem
- Model waypoints on a graph with physical constraints; pure ML predictions that violate physics can be corrected cheaply in post-processing
- Pedestrian dead reckoning from IMU can serve as a strong prior for short-horizon position prediction when signal quality is low

---

## 4. ISIC 2024 Skin Cancer Detection (2024)

**Task type:** Medical imaging + tabular — binary classification of malignant skin lesions from 3D total body photograph crops plus patient metadata
**Discussion:** https://www.kaggle.com/c/isic-2024-challenge/discussion/533196

**Approach:** Winner Ilya Novoselskiy built a hybrid image + tabular pipeline. Multiple image models (EVA02, EfficientNet variants, ResNeXt, ConvNeXt) were trained on the cropped lesion images with heavy augmentation, generating out-of-fold (OOF) predictions that were then stacked as additional features alongside engineered metadata features into a GBDT second stage (LightGBM/CatBoost/XGBoost ensemble). The extreme class imbalance (malignant cases were rare) was addressed via careful threshold calibration and partial AUC optimization.

**Key techniques:**
1. EVA02 and EfficientNet image backbones with GeM (Generalized Mean) pooling for robust feature extraction from small lesion crops
2. OOF stacking: image model predictions used as meta-features for a second-stage GBDT ensemble, combining image and tabular signal
3. Engineered metadata features from patient demographics and lesion site, integrated alongside image OOF predictions in the GBDT
4. Partial AUC optimization (pAUC above 80% sensitivity threshold) as the competition metric — threshold tuning and calibration critical
5. Heavy image augmentation including color jitter, random erasing, and test-time augmentation to handle the small and diverse lesion appearances

**How to Reuse:**
- The image-OOF → GBDT stacking pattern is a powerful template for any medical competition with tabular metadata: train image models, extract embeddings or OOF preds, feed into XGBoost/LightGBM alongside structured features
- GeM pooling (replacing global average pooling with a learnable power mean) reliably helps for fine-grained retrieval and classification tasks
- For severe class imbalance, optimize directly on the competition metric (partial AUC, F1) rather than cross-entropy; use stratified sampling
- Always include patient-level features (age, sex, lesion site, lesion count) as tabular features even when the task appears purely visual

---

## 5. Stanford Ribonanza RNA Folding (2023)

**Task type:** Biology — sequence-to-sequence regression predicting RNA chemical reactivity (DMS and 2A3 probing) at each nucleotide position
**Discussion:** https://www.kaggle.com/c/stanford-ribonanza-rna-folding/discussion/460121

**Approach:** The winning team "vigg" (autosome-ru) developed a single model called ArmNet (Artificial Reactivity Mapping using Neural Networks), a transformer encoder augmented with convolutional operations and biased self-attention matrices derived from pairwise representations. ArmNet used EternaFold Base Pair Probability (BPP) matrices as auxiliary structural inputs to initialize pairwise representations, giving the model access to predicted secondary structure alongside the raw sequence. The model was trained for 270 epochs with cyclic learning rate scheduling and SGD fine-tuning, on dual-EPYC / 6×V100 hardware.

**Key techniques:**
1. BPP (Base Pair Probability) matrix features from EternaFold as pairwise auxiliary inputs, encoding predicted RNA secondary structure into the attention bias
2. Triangular self-attention operations (adapted from AlphaFold2's evoformer) applied to the pairwise representation to model non-local nucleotide interactions
3. Squeeze-and-Excitation (SE) modules on convolutional pathways for channel-wise recalibration alongside transformer layers
4. Dynamic positional encodings (not fixed sinusoidal) and adjacency kernels (kernel size 3) for local sequence context
5. Cyclic learning rate with 5% warm-up for 270 epochs, followed by SGD fine-tuning for 25 epochs; weight decay 0.05

**How to Reuse:**
- For any RNA/protein sequence task: integrate secondary structure predictions (from tools like EternaFold, RNAfold, or ESMFold) as auxiliary pairwise features — they carry strong prior structural information
- Triangular attention from AlphaFold2 / OpenFold is publicly available and worth porting to other sequence tasks where pairwise relationships matter (protein-protein binding, genomics)
- BPP matrices as attention bias (rather than hard constraints) let the model learn to use structural priors without being locked into potentially incorrect predictions
- Cyclic LR scheduling + SGD fine-tuning after Adam warmup is a known stabilization trick for transformer convergence on small scientific datasets

---

## 6. Cornell Birdcall Identification (2020)

**Task type:** Audio — multi-label classification of bird species from noisy soundscape recordings with weak labels
**Discussion:** https://www.kaggle.com/c/birdsong-recognition/discussion/183208

**Approach:** Winner Ryan Wong evolved from simple mel-spectrogram image classification to a Sound Event Detection (SED) framework based on PANNs (Pretrained Audio Neural Networks), replacing the standard CNN backbone with a pretrained DenseNet121. The model used an attention mechanism (AttBlock) to produce both clip-level and frame-level predictions from the same forward pass, enabling both global and local bird-presence decisions. The decisive breakthrough was voting-based ensembling of 13 models with dual thresholds on both clipwise and framewise outputs, jumping from 7th on the public leaderboard to 1st on the private leaderboard.

**Key techniques:**
1. PANNs SED framework — weakly-supervised training with attention (AttBlock) producing both clipwise_output and framewise_output simultaneously
2. DenseNet121 as the feature extractor backbone (replacing default CNN in PANNs), outperforming ResNet-style architectures for audio classification per prior literature
3. Attention modification: `torch.tanh` instead of `torch.clamp` in AttBlock to reduce overfitting on the small per-class dataset (< 100 samples per species)
4. Ensemble of 13 models (4-fold, 5-fold, with/without mixup) using voting: prediction accepted only if ≥ 4 of 13 models exceed 0.3 threshold on both clip-level and frame-level outputs
5. Augmentations: mixup, pink noise background via Audiomentation library (AddGaussianNoise, AddBackgroundNoise, AddShortNoises), SpecAugment

**How to Reuse:**
- PANNs + SED approach (weakly-supervised with clip-level labels producing frame-level predictions) is the go-to starting point for any audio event detection task
- Dual thresholding on both clip-level and frame-level outputs to reduce false positives is broadly applicable to soundscape and surveillance audio tasks
- Mixup augmentation on spectrograms consistently helps for audio with limited per-class data
- For multi-label audio with imbalanced classes, ensemble voting (rather than averaging probabilities) is more robust at threshold selection time

---

## 7. Plant Pathology 2020 — FGVC7 (2020)

**Task type:** Computer vision — multi-class classification of apple leaf diseases (healthy, scab, rust, multiple diseases) from high-resolution photos
**Discussion:** https://www.kaggle.com/c/plant-pathology-2020-fgvc7/discussion/154056

**Approach:** The winning team from Alipay Tian Suan Security Lab achieved private LB AUC of 0.98445 using SE-ResNeXt50 as the core backbone, trained with knowledge distillation and soft labels. Rather than using hard one-hot labels, they computed a weighted average between the model's own predictions and the ground truth, which provided implicit label smoothing that helped handle the small number of mislabeled examples in the dataset. Stratified k-fold cross-validation was used to build an ensemble of models with diverse decision boundaries.

**Key techniques:**
1. SE-ResNeXt50 backbone (Squeeze-and-Excitation ResNeXt) for strong visual feature extraction from high-resolution leaf images
2. Knowledge distillation via soft labels — blending ground-truth hard labels with the model's own predictions as a training target, providing label smoothing proportional to model uncertainty
3. Stratified k-fold cross-validation ensemble, with fold-averaged probabilities as final predictions
4. High-resolution training (full image, not cropped) to preserve disease texture detail at the leaf margins
5. Strong augmentations: flips, rotations, color jitter, grid distortion appropriate for plant disease textures

**How to Reuse:**
- Soft label / self-distillation is a cheap regularizer applicable to any image classification task with potential label noise; it is especially valuable for small datasets
- SE-ResNeXt50 and its variants remain competitive backbones for fine-grained image classification, often outperforming larger EfficientNets when training data is limited
- Stratified k-fold (stratifying by class) is essential for multi-class tasks with class imbalance; it both enables OOF ensembling and gives reliable CV scores
- For FGVC tasks, avoid aggressive cropping during training — lesion and disease markers often appear at image boundaries

---

## 8. AI Mathematical Olympiad — Progress Prize 2 (2024)

**Task type:** Math / LLM reasoning — solving National Olympiad-level competition math problems (algebra, combinatorics, geometry, number theory) using LLMs within a 5-hour / 4×L4 GPU constraint
**Discussion:** https://www.kaggle.com/c/ai-mathematical-olympiad-progress-prize-2/discussion/574765

**Approach:** NVIDIA's NemoSkills team won using OpenMath-Nemotron-14B (fine-tuned from Qwen2.5-14B-Base on the OpenMathReasoning dataset) with Tool-Integrated Reasoning (TIR) — training the model to dynamically invoke a Python sandbox for computation rather than solving everything in natural language. Inference used TensorRT-LLM with FP8 quantization for speed, and employed early-stopping majority voting: if 4 of the first 5 generated solutions agreed on an answer, remaining generations were cancelled. The training data (OpenMathReasoning) was synthesized using Qwen2.5-32B-Instruct, DeepSeek-R1, and QwQ-32B to generate diverse solution paths.

**Key techniques:**
1. Tool-Integrated Reasoning (TIR) — the model is trained to write Python code snippets mid-solution, execute them in a sandbox, and continue reasoning from the output, enabling exact arithmetic and algorithmic sub-problems
2. Qwen2.5-14B-Base fine-tuned on millions of synthetically generated math reasoning traces (OpenMathReasoning dataset) covering all Olympiad topic areas
3. TensorRT-LLM inference with FP8 quantization to maximize throughput within the 4×L4 GPU, 5-hour compute budget
4. Early-stopping majority voting — consensus from first 5 samples short-circuits remaining generation, reducing wasted compute on problems where the model is confident
5. Multi-model distillation for training data: Qwen2.5-32B-Instruct, DeepSeek-R1, and QwQ-32B used as solution generators, giving diverse chain-of-thought styles for the fine-tuning corpus

**How to Reuse:**
- TIR (code execution in the reasoning loop) is the key differentiator over pure CoT for math and quantitative tasks — implement via tool-calling APIs or a code sandbox wrapper around any capable LLM
- For math fine-tuning, synthetic data from multiple strong teacher models (diverse CoT styles) consistently outperforms single-teacher distillation
- Majority voting / self-consistency is a reliable accuracy booster for any deterministic-answer LLM task; combine with early stopping to manage compute budgets
- FP8 quantization via TensorRT-LLM or bitsandbytes introduces minimal accuracy loss on math tasks while roughly doubling throughput on Hopper/Ada GPUs

---

## 9. Google Landmark Retrieval 2020 (2020)

**Task type:** Computer vision — image retrieval of the same landmark across a gallery of 5M images; evaluated at mAP@100
**Discussion:** https://www.kaggle.com/c/landmark-retrieval-2020/discussion/176037

**Approach:** The 1st place team (arXiv: 2009.05132) based their solution on metric learning with ArcFace loss on the Google Landmarks Dataset v2, using EfficientNet-B7 backbones trained at progressively larger resolutions (up to 736×736). Training was staged: first on the clean subset, then fine-tuned on the full noisy dataset with reduced weight for unclean samples. Re-ranking was performed by adjusting retrieval scores with classification logits and filtering non-landmark distractors. Ensemble of multiple EfficientNet models trained at different scales produced the final mAP@100 of 0.38677.

**Key techniques:**
1. ArcFace (Additive Angular Margin) loss for metric learning, training image embeddings to be well-separated in angular space across 200K+ landmark classes
2. Multi-stage training: clean data first → full dataset with `NOT_CLEAN_WEIGHT=0.5` for noisy samples → progressive image size scaling to 512→640→736
3. EfficientNet-B7 backbone trained on TPUs (Google Colab), with TFRecords for high-throughput data loading
4. Re-ranking: retrieval scores adjusted by combining embedding similarity with classification logits from the same backbone, boosting precision on the hardest cases
5. Ensemble of models trained at different resolution stages and clean/noisy weight ratios, combined by score averaging

**How to Reuse:**
- ArcFace (or its variants SubCenterArcFace, CurricularFace) is the standard loss for image retrieval and face recognition — use it whenever you need discriminative embedding spaces at scale
- The clean → noisy fine-tuning strategy is applicable to any dataset with known label noise: fit a strong prior on clean data, then adapt with down-weighted noisy samples
- Progressive image resolution training (start small, fine-tune at full resolution) significantly improves detail capture without the memory cost of training large images from scratch
- Distractor filtering (predicting "non-landmark" probability and suppressing those results) is a critical retrieval post-processing step for real-world gallery datasets

---

## 10. Halite by Two Sigma (2020)

**Task type:** Reinforcement learning / game AI — build an agent that collects halite resources and deposits them while competing against three other agents in a grid environment
**Discussion:** https://www.kaggle.com/c/halite/discussion/183543

**Approach:** The winning solution by Tom Van de Wiele (ttvand) combined rule-based decision logic with deep learning components, using a hybrid architecture rather than pure end-to-end reinforcement learning (which was computationally prohibitive for the competition's self-play constraints). The codebase contained separate folders for rule-based and deep learning agents, with the final winning submission primarily leveraging hand-crafted strategic rules refined through thousands of simulation games. The rule-based core handled ship routing, collision avoidance, and shipyard placement heuristics, while neural components assisted with value estimation and opponent modeling.

**Key techniques:**
1. Hybrid rule-based + neural architecture — rule-based core for deterministic tactical decisions (routing, collision), neural module for strategic evaluation
2. Hand-crafted game heuristics: ship routing algorithms, dynamic shipyard placement based on halite density maps, early-game vs. late-game strategy switching
3. Simulation-based evaluation: agent policy refined through repeated self-play simulations to identify dominant strategies against diverse opponent types
4. Opponent modeling: maintaining beliefs about opponent strategy class (aggressive, defensive, mining-focused) and adapting behavior accordingly
5. State representation: halite density maps, ship positions, cargo levels encoded as spatial grids for neural value function input

**How to Reuse:**
- Pure RL with self-play is often impractical within Kaggle compute limits; hybrid rule-based + learned value function is a reliable competitive pattern for game AI competitions
- For multi-agent resource collection games, hard-coding collision avoidance and routing as inviolable constraints (not softly learned) prevents catastrophic failures
- Simulate thousands of games offline and analyze win/loss patterns before committing to a strategy — empirical game analysis is often more efficient than theoretical derivation
- Halite-like resource collection problems map directly to supply chain optimization and vehicle routing; the spatial grid representations transfer to logistics domains

---

## 11. NFL Big Data Bowl 2026 — Prediction (2025)

**Task type:** Tabular + spatial — predicting player movement (position) after the ball is thrown, using pre-snap and pre-pass player tracking (Next Gen Stats) data
**Discussion:** https://www.kaggle.com/c/nfl-big-data-bowl-2026-prediction/discussion/651604

**Approach:** The analytics winner Lucca Ferraz (Rice University) introduced "ghost defenders" — a counterfactual modeling framework that generates hypothetical distributions of where defenders would optimally position themselves while the ball is in the air, given pre-pass tracking data. Rather than a single position regression, the model produces a distributional prediction over player movement trajectories that captures uncertainty about route running and coverage schemes. Training data from 2023–24 seasons was evaluated on live 2025 Week 14–18 games, making this a prospective forecasting challenge.

**Key techniques:**
1. Ghost defender framework — simulating optimal counterfactual defender positions as a probabilistic distribution over the field, capturing coverage uncertainty
2. Pre-snap feature engineering from Next Gen Stats: player speeds, alignments, formations, snap motion, separation distances at ball release
3. Trajectory prediction as a distribution over future positions rather than a point estimate, enabling evaluation against actual player locations at ball arrival
4. Spatial feature encoding: relative player positions, field geometry (hash marks, sidelines, end zones) encoded as spatial offsets from ball location
5. Evaluation against live 2025 season data (Weeks 14–18) as prospective out-of-sample test — resistant to post-hoc overfitting

**How to Reuse:**
- Counterfactual "ghost" modeling (what would an optimal agent do?) is broadly applicable to sports analytics, autonomous driving, and robotics for generating training signal and evaluation baselines
- Pre-snap / pre-event feature engineering for trajectory prediction should always include velocity, heading, and relative spacing — these are more predictive than absolute positions alone
- For player tracking competitions, distributional predictions (Gaussian mixtures, normalizing flows) consistently outperform point regressions on evaluation metrics that penalize overconfidence
- NGS-style tracking data maps directly to logistics fleet tracking and robotics path planning — the same spatial feature engineering patterns apply

---

## 12. Feedback Prize — Predicting Effective Arguments (2022)

**Task type:** NLP — multi-class classification of argumentative discourse elements (lead, position, claim, evidence, concluding statement, counterclaim, rebuttal) as Adequate, Effective, or Ineffective
**Discussion:** https://www.kaggle.com/c/feedback-prize-effectiveness/discussion/347536

**Approach:** Team Hydrogen (Yauhen Babakhin et al.) won with a two-stage stacking ensemble of DeBERTa-v3 models. In stage 1, multiple DeBERTa-v3-large models were fine-tuned with different seeds, fold splits, and additional training data (pulled from a Kaggle dataset supplement). In stage 2, a meta-learner combined stage-1 OOF predictions into a final ensemble. The solution used special token markup to identify discourse element boundaries within the full student essay context, giving the model positional awareness of which argument span was being evaluated.

**Key techniques:**
1. DeBERTa-v3-large as the primary backbone — disentangled attention with absolute and relative position encodings makes it particularly strong for span-level classification within long documents
2. Two-stage stacking: stage-1 DeBERTa models produce OOF predictions → stage-2 meta-learner (lightweight MLP or logistic regression) combines them
3. Special token boundary markers injected around the target discourse span within the full essay context, giving the model explicit span position signal
4. Additional training data sourced from a supplementary Kaggle dataset, significantly expanding the small competition dataset
5. Multi-seed, multi-fold ensemble diversity — training each fold with 3+ seeds and averaging predictions before stacking

**How to Reuse:**
- DeBERTa-v3-large is the default starting point for any NLP span classification or text effectiveness task on Kaggle as of 2022–2024
- Marking target spans with special tokens (e.g., `[TGT_START]...[TGT_END]`) within full document context is more effective than extracting spans in isolation
- Two-stage stacking with OOF predictions is the canonical NLP ensemble pattern — always generate OOF preds from stage 1 before building a meta-learner
- Supplementary external data (even weakly labeled) from related Kaggle datasets is almost always worth incorporating; check Kaggle datasets for related competition data

---

## 13. Peking University / Baidu Autonomous Driving (2019)

**Task type:** Computer vision 3D — estimating full 6-DoF pose (x, y, z, pitch, yaw, roll) of cars from single monocular RGB images; evaluated via mean Average Precision over 3D IoU
**Discussion:** https://www.kaggle.com/c/pku-autonomous-driving/discussion/127037

**Approach:** The top solutions across the board adopted CenterNet (Objects as Points) as the detection backbone. The 1st place approach extended CenterNet with HRNet (High-Resolution Network) as a stronger backbone for maintaining spatial resolution across feature pyramid levels, combined with a camera-intrinsic-aware decoder that regressed all 6 pose parameters. A key trick shared from the winner was camera rotation augmentation to artificially expand training variance in pitch and roll. Multiple CenterNet models with different backbones were blended (blend of blends) in the final submission.

**Key techniques:**
1. CenterNet (Objects as Points) detection framework extended to 3D — predicting heatmaps for car centers with regression heads for depth, dimensions, and all 6 rotation angles
2. HRNet backbone maintaining high spatial resolution throughout the feature hierarchy, critical for precise 3D localization of distant/small vehicles
3. Camera-intrinsic-aware pose decoding — using the known camera matrix to convert 2D detection + depth predictions into 3D world coordinates
4. Camera rotation augmentation — synthetically varying pitch and roll camera angles at training time to improve pose prediction generalization
5. Blend-of-blends ensemble: multiple CenterNet variants (different backbones: DLA34, HRNet, ResNet-based) blended via score averaging

**How to Reuse:**
- CenterNet is still one of the fastest anchor-free detectors for monocular 3D object detection — use it as a baseline for any single-camera 3D pose task (autonomous driving, robotics)
- HRNet or HRFormer backbones outperform standard FPN architectures for tasks requiring precise spatial localization; worth the compute cost for 3D regression heads
- For monocular depth/3D tasks, always incorporate camera intrinsics explicitly in the decoding step rather than learning a free-form coordinate transformation
- Augmenting camera angle and perspective transforms is underused in autonomous driving datasets where viewpoint variation is limited by the fixed dashboard camera position

---

## 14. DFL Bundesliga Data Shootout (2022)

**Task type:** Computer vision — temporal event detection in long soccer match videos; identifying and timestamping throw-ins, passes, and tackles to within 1-second accuracy
**Discussion:** https://www.kaggle.com/c/dfl-bundesliga-data-shootout/discussion/359932

**Approach:** Winners Dr. Philipp Singer, Pascal Pfeiffer, and Yauhen Babakhin (a recurring gold-medal team on Kaggle) applied a "avoid reinventing the wheel" philosophy: they framed the temporal detection task as a sliding-window frame classification problem, training efficient CNN classifiers on frame crops to produce per-frame event probability scores, then using peak detection on the smoothed probability time series to find event timestamps. The solution ran comfortably within the 9-hour execution limit (~5 hours actual), using efficient architectures and avoiding complex video transformers. Their team's strength came from clean experimental design, strong cross-validation, and ensemble averaging.

**Key techniques:**
1. Sliding-window frame extraction — sampling frames at fixed stride around candidate event windows, treating each frame as an independent classification input
2. Frame-level binary classifiers (one per event class: throw-in, pass, tackle) trained on CNN backbones (EfficientNet family), producing dense per-frame probability scores
3. Temporal smoothing + peak detection on the probability time series to localize event timestamps, tuned by optimizing the competition's mAP metric
4. Efficient inference design — prioritizing fast CNN architectures over video transformers to meet strict 9-hour execution wall clock
5. Ensemble of multiple CNN models with averaged probabilities before peak-finding, reducing false positive rate on rare events

**How to Reuse:**
- For temporal action detection in long video, the frame-classifier + temporal smoothing + peak detection pipeline is simpler and often competitive with full video transformer approaches
- Training separate binary classifiers per event class (rather than multi-class) gives cleaner decision boundaries when event classes have very different visual signatures
- Peak detection hyperparameters (minimum peak distance, prominence threshold) should be tuned directly on the competition metric, not on a proxy loss
- Sliding window feature extraction from video is directly applicable to any temporal event detection domain (medical procedure videos, sports analytics, security footage)

---

## 15. NFL Player Contact Detection (2022)

**Task type:** Computer vision + tabular — binary detection of player-player and player-ground contact events from broadcast video frames combined with player tracking data; evaluated on Matthews Correlation Coefficient
**Discussion:** https://www.kaggle.com/c/nfl-player-contact-detection/discussion/391635

**Approach:** 1st place winner nvnnghia built a 3D CNN ensemble (CSN — Channel-Separated Networks with ResNet50 backbone) pretrained on Kinetics-400 via mmaction2, processing multi-frame video clips cropped around player helmet bounding boxes. A separate XGBoost post-processing model refined the raw 3D CNN predictions using tabular tracking features (player speeds, distances, accelerations from NGS), filtering easy negatives before the neural inference and correcting systematic errors from the video model. The final submission averaged predictions from 6 checkpoint models trained on different fold splits.

**Key techniques:**
1. CSN (Channel-Separated Networks) 3D CNN with ResNet50 backbone, initialized from Kinetics-400 mmaction2 pretrained weights for video understanding transfer learning
2. Helmet bounding box crop as the primary video region of interest — focus the 3D CNN on the relevant spatial region around each player pair being evaluated for contact
3. XGBoost pre-screening and post-processing: first filter "easy negative" pairs using tracking features (inter-player distance threshold), then refine 3D CNN scores with a tracking-feature XGBoost model
4. Separate PP (Player-Player) and PG (Player-Ground) XGBoost models trained on fold 0–4, with 3D CNN trained on folds 5–9 for full training data utilization
5. Multi-checkpoint ensemble: 6 trained model checkpoints averaged at inference, combined with XGBoost post-processing predictions

**How to Reuse:**
- Kinetics-400 pretrained 3D CNNs (via mmaction2, torchvision, or pytorchvideo) are the standard transfer learning starting point for any sports video analysis task
- Combining a neural video model with an XGBoost post-processor that uses tabular sensor/tracking features is a robust fusion pattern when both modalities carry complementary signal
- Pre-screen "easy negatives" with cheap heuristics (distance threshold) before neural inference — dramatically reduces compute and can also reduce false positive rate
- For contact/interaction detection, crop the video around the relevant body regions (helmets, hands, feet) rather than using full-frame input; attention is not reliable enough to focus itself

---

## Cross-Cutting Patterns

| Pattern | Competitions Where It Won |
|---|---|
| Transformer fine-tuning (BERT / DeBERTa / EVA02) | #1 TF-QA, #12 Feedback Prize, #4 ISIC 2024 |
| 2D CNN + RNN/GRU sequence model for 3D/temporal data | #2 RSNA PE, #6 Birdcall, #14 DFL Bundesliga, #15 NFL Contact |
| OOF stacking (image model → GBDT second stage) | #4 ISIC 2024, #12 Feedback Prize, #15 NFL Contact |
| Ensemble voting / model averaging (5+ models) | #6 Birdcall (13 models), #9 Landmark, #3 Indoor, #7 Plant Pathology |
| Metric learning + ArcFace loss | #9 Google Landmark Retrieval |
| Physics / geometry aware decoding | #3 Indoor (Kalman filter), #13 PKU Baidu (camera intrinsics), #11 NFL BDB (spatial features) |
| Tool-integrated reasoning / code execution in loop | #8 AIMO (Python sandbox TIR) |
| Pretrained large-scale video models (Kinetics) | #15 NFL Contact (CSN mmaction2) |
| Pretrained audio models (PANNs) | #6 Cornell Birdcall |
| Soft labels / knowledge distillation | #7 Plant Pathology |
| Structural biology priors as auxiliary features | #5 RNA Folding (BPP matrices) |
| Hybrid rule-based + learned agent | #10 Halite |
| Counterfactual / ghost modeling for distribution prediction | #11 NFL Big Data Bowl 2026 |
| High-resolution progressive training | #9 Landmark (512→736), #2 RSNA (576×576) |
| Post-processing with domain constraints | #3 Indoor (Kalman), #14 DFL (peak detection), #15 NFL (XGBoost) |

---

**Notes on data availability:** Kaggle discussion pages require authentication and JavaScript rendering; technical details for competitions #1 (TF-QA), #7 (Plant Pathology), #13 (PKU Baidu), and #14 (DFL Bundesliga) were reconstructed from indexed GitHub repositories, arXiv papers, and third-party writeups. All other competitions have publicly indexed solution repositories or blog posts that confirm the details above. The NFL Big Data Bowl 2026 (#11) winner details come from the NFL Football Operations official announcement rather than the Kaggle discussion thread.agentId: aad0cc96470148354 (use SendMessage with to: 'aad0cc96470148354' to continue this agent)
<usage>total_tokens: 95099
tool_uses: 68
duration_ms: 444202</usage>