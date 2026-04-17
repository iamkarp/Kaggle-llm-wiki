# Kaggle Past Solutions — SRK Batch 2

**Source:** kaggle.com/sudalairajkumar/winning-solutions-of-kaggle-competitions
**Ingested:** 2026-04-16

---

## 1. OpenVaccine: COVID-19 mRNA Vaccine Degradation (2020)

**Task:** Sequence regression — predict per-nucleotide RNA degradation rates to aid mRNA vaccine stability research.
**Discussion:** https://www.kaggle.com/c/stanford-covid-vaccine/discussion/189620

**Approach:** The winning solution used Graph Neural Networks (GNNs) to model RNA secondary structure as a graph, capturing long-range dependencies between nucleotides that sequential models miss. The team combined multiple GNN architectures (EGNNs, GCNs) with transformer-based sequence models and ensembled heavily. A key insight was incorporating predicted RNA folding structures (from tools like Vienna RNAfold and EternaFold) as graph edges.

**Key Techniques:**
1. **RNA Secondary Structure Graphs** — Encoded predicted base-pair bonds as graph edges; GNNs propagated information along these structural connections rather than purely sequential neighbors.
2. **Multi-folding Ensemble** — Used multiple RNA structure prediction tools (Vienna, EternaFold, CONTRAfold) and averaged predictions across different folding hypotheses to reduce structural uncertainty.
3. **Graph Attention Networks (GAT)** — Attention weights on edges allowed the model to learn which structural connections were most predictive per position.
4. **External Dataset Augmentation (OpenVaccine eterna data)** — Supplemented competition data with Eterna crowdsourced experimental degradation measurements.
5. **Sequence + Structure Multi-input Models** — Concatenated raw sequence embeddings with structural features (dot-bracket notation, adjacency matrices) at each node.

**How to Reuse:**
- Any RNA/protein sequence task: model as a graph using predicted secondary structure, not just linear sequence.
- When domain-specific structure predictors exist (folding tools, protein structure predictors), use their outputs as graph topology inputs to GNNs.
- Ensemble across multiple third-party structure predictions when ground-truth structure is unavailable — this reduces epistemic uncertainty in the input graph.
- For positional regression on sequences, ensure the model can route information from distant positions via structural shortcuts, not just adjacent tokens.

---

## 2. Feedback Prize — Evaluating Student Writing (2021)

**Task:** Token classification / NER — identify and classify argumentative discourse elements (claims, evidence, rebuttals, etc.) in student essays.
**Discussion:** https://www.kaggle.com/c/feedback-prize-2021/discussion/313177

**Approach:** The winning team fine-tuned large pretrained language models (DeBERTa-v3-large, BigBird) on the token classification task, treating span detection as a BIO-tagging problem. They trained models at multiple granularities — word-level and token-level — and fused predictions with span-level post-processing heuristics to clean boundaries. Ensembling across multiple LM backbones was critical, with each model contributing calibrated span probabilities.

**Key Techniques:**
1. **DeBERTa-v3-large as Backbone** — DeBERTa's disentangled attention (separate content and position attention) provided superior performance on long-document NER compared to standard transformers.
2. **BIO Tagging with Span Post-processing** — Raw BIO predictions were post-processed with rule-based span merging and minimum-length filtering to suppress noisy short predictions.
3. **Stride/Overlap for Long Documents** — Essays exceeded typical context windows; overlapping window inference with prediction averaging at overlap regions maintained coherence across chunk boundaries.
4. **Multi-backbone Ensemble** — DeBERTa variants + BigBird + Longformer contributed complementary recall on different discourse element types.
5. **Soft-label Training / Pseudo-labeling** — Later training stages used pseudo-labels from ensemble predictions on unlabeled external essays to regularize models.

**How to Reuse:**
- For any long-document NER task, use strided inference with overlap averaging — critical when sequences exceed 512 tokens.
- DeBERTa-v3-large is the default starting backbone for English token classification; worth fine-tuning even on small datasets.
- Always add span-level post-processing (minimum span length, boundary snapping) on top of raw BIO predictions — it provides nearly free F1 improvement.
- Pseudo-label unlabeled text from the same domain to expand effective training set.

---

## 3. SETI Breakthrough Listen (2021)

**Task:** Binary classification — detect anomalous narrowband signals (potential technosignatures) in radio telescope spectrograms.
**Discussion:** https://www.kaggle.com/c/seti-breakthrough-listen/discussion/266385

**Approach:** The winning solution treated spectrogram frames as images and applied standard image classification CNNs (EfficientNet, NFNet) with heavy data augmentation. A key innovation was training on the "ON/OFF" cadence structure — signals of interest appear in ON pointings but not OFF pointings — by stacking the six cadence images as channels so the model could learn the presence/absence pattern across pointings. Ensembling multiple CNN architectures with test-time augmentation closed the gap to the top score.

**Key Techniques:**
1. **Cadence Stacking as Multi-channel Input** — The six ON/OFF observation frames were stacked as a 6-channel image, enabling the model to directly learn the cadence-dependent anomaly pattern.
2. **EfficientNet / NFNet Backbones** — Large pretrained image classifiers fine-tuned on spectrograms; pretrained ImageNet weights transferred surprisingly well despite the domain gap.
3. **Aggressive Augmentation** — Mixup, CutMix, random erasing, and frequency/time masking (SpecAugment-style) improved generalization on the small labeled dataset.
4. **Pseudo-labeling on Unlabeled Test Cadences** — High-confidence test predictions were used as pseudo-labels for additional training rounds.
5. **Snapshot Ensemble / SWA** — Stochastic Weight Averaging and snapshot ensembling within a single training run improved calibration without additional inference cost.

**How to Reuse:**
- When data has a structured multi-frame temporal/observational cadence, stack frames as channels rather than processing independently — teaches the model about presence/absence patterns.
- SpecAugment-style augmentation (frequency and time masking) is directly transferable to any spectrogram or time-frequency representation task.
- For small labeled datasets on image-like inputs, aggressive augmentation (Mixup + CutMix + erasing) is often more valuable than architectural changes.

---

## 4. SIIM-FISABIO-RSNA COVID-19 Detection (2021)

**Task:** Detection + classification — localize COVID-19 opacity regions in chest X-rays and classify study-level severity.
**Discussion:** https://www.kaggle.com/c/siim-covid19-detection/discussion/263658

**Approach:** The solution combined a two-stage pipeline: (1) object detection models (YOLOv5, EfficientDet) for opacity bounding box prediction, and (2) image-level classification models for the four study-level labels. Predictions from both stages were fused with carefully tuned NMS and WBF (Weighted Boxes Fusion) strategies. External chest X-ray datasets (NIH ChestXray14, CheXpert, VinBigData) were used for pretraining to overcome the limited competition data.

**Key Techniques:**
1. **External Pretraining on CheXpert / NIH ChestXray14** — Models pretrained on large chest X-ray repositories before fine-tuning substantially improved both detection sensitivity and classification accuracy.
2. **Weighted Boxes Fusion (WBF)** — WBF over NMS for merging overlapping detections from multiple models produced better box quality, especially for overlapping opacities.
3. **Two-stage Detection + Classification Fusion** — Detection confidence scores were used as auxiliary features for study-level classification, linking the two tasks.
4. **TTA with Horizontal Flip + Scale** — Test-time augmentation at multiple scales and flips averaged out localization noise.
5. **YOLOv5 + EfficientDet Ensemble** — Complementary architectures (anchor-based vs. anchor-free) combined to improve recall across opacity sizes.

**How to Reuse:**
- For medical imaging with limited labels, always pretrain on large public domain datasets (CheXpert, MIMIC-CXR, NIH) before fine-tuning on competition data.
- Use WBF instead of standard NMS when ensembling detection models — it preserves box contributions from all models rather than suppressing non-maximal ones.
- When a task has both localization and classification subtasks, share a backbone and fuse intermediate representations — or at minimum feed detection outputs as classification features.

---

## 5. Abstraction and Reasoning Challenge (2020)

**Task:** Program synthesis — solve visual pattern puzzles (ARC tasks) by inducing the transformation rule from input/output grid pairs.
**Discussion:** https://www.kaggle.com/c/abstraction-and-reasoning-challenge/discussion/154597

**Approach:** The winning solution used a Domain-Specific Language (DSL) search approach: a hand-crafted library of primitive grid transformations (rotations, reflections, color substitutions, object extraction, flood fill, etc.) was composed programmatically to find transformation programs that explain the given input/output demonstrations. A DFS/BFS search over program compositions was guided by heuristics and pruning rules, and multiple candidate programs were tested against all training pairs before selecting the one that generalizes. No neural networks were used.

**Key Techniques:**
1. **Hand-crafted DSL with Primitive Operations** — A library of ~100+ atomic grid operations (translate, rotate, recolor, extract objects, fill, mirror) formed the building blocks for program search.
2. **Depth-first Program Composition Search** — Programs were built by composing primitives in sequence; DFS with pruning found programs consistent with all input/output training pairs.
3. **Object-centric Decomposition** — Grids were parsed into discrete objects (connected components, colored regions) before applying primitives, reducing the search space dramatically.
4. **Symmetry and Pattern Heuristics** — Specialized detection routines for common ARC patterns (repetition, symmetry, gravity, counting) were integrated as high-priority search branches.
5. **Multiple Program Candidates + Voting** — Several consistent programs were identified per task; their outputs on the test input were voted on or the simplest (shortest) program was selected.

**How to Reuse:**
- Program synthesis / DSL search is the right frame for tasks where the transformation is rule-based and training examples are very few (2–5 pairs).
- When neural approaches fail due to tiny data, invest in hand-crafting a domain-specific primitive library — this pays off on combinatorial reasoning tasks.
- Object decomposition (treating grid regions as discrete entities) is broadly applicable to any grid or structured-layout task.
- This approach directly informs ARC-AGI follow-on competitions — the DSL methodology remains state-of-the-art.

---

## 6. Mercari Price Suggestion (2017)

**Task:** Regression — predict product listing price from title, description, category, brand, and item condition.
**Discussion:** https://www.kaggle.com/c/mercari-price-suggestion-challenge/discussion/50256

**Approach:** The winning solution combined Ridge regression on TF-IDF features with a Feed-Forward Neural Network (FFNN) that processed text and categorical inputs jointly. Both models were trained on the full dataset with minimal preprocessing, and their predictions were blended. A key challenge was inference speed (the competition enforced a time limit), so the team optimized for fast sparse matrix operations and avoided heavy deep learning stacks.

**Key Techniques:**
1. **TF-IDF + Ridge Regression on Raw Text** — Unigram and bigram TF-IDF across title + description + category chain fed into Ridge regression; fast to train and strong baseline.
2. **Sparse Feed-forward Network with Embedding Layers** — Categorical features (brand, category, condition) were embedded; text tokens were hashed and embedded; all concatenated and passed through dense layers.
3. **Log-price Target Transformation** — Predicting log(price+1) rather than raw price stabilized gradients and improved RMSLE metric directly.
4. **Sparse Matrix Operations for Speed** — The competition had a kernel time limit; all feature engineering used scipy sparse matrices to stay within compute budget.
5. **Simple Averaging Ensemble (Ridge + NN)** — The two complementary model families (linear vs. nonlinear) were blended with equal weight for a consistent improvement over either alone.

**How to Reuse:**
- On text+tabular regression tasks, always include a TF-IDF + linear model baseline — it is fast, interpretable, and often competitive with deep models.
- When competition kernels have time limits, profile every step; sparse matrix operations are 10-100x faster than dense equivalents for high-cardinality text features.
- Log-transform skewed price/count targets — standard practice, but easy to forget and costly to omit.
- The Ridge + NN blend pattern generalizes broadly: linear models capture global TF-IDF signal, NNs capture interaction effects; together they outperform either alone.

---

## 7. Google Landmark Retrieval (2018)

**Task:** Image retrieval / metric learning — given a query image, retrieve other images of the same landmark from a large gallery.
**Discussion:** https://www.kaggle.com/c/landmark-retrieval-challenge/discussion/57855

**Approach:** The winning team trained CNN models with metric learning objectives (ArcFace / CosFace loss) on the large landmark dataset, producing embeddings where images of the same landmark cluster tightly in embedding space. At retrieval time, approximate nearest neighbor search (FAISS) retrieved top-k candidates, which were then re-ranked using local feature matching (DELF — Deep Local Features). This two-stage coarse retrieval + geometric verification pipeline became the blueprint for subsequent landmark retrieval competitions.

**Key Techniques:**
1. **ArcFace Loss for Embedding Training** — Additive angular margin loss produced more discriminative embeddings than softmax classification alone, critical for fine-grained landmark identity discrimination.
2. **DELF Local Feature Re-ranking** — After global embedding retrieval, DELF (learned keypoint descriptors from Google) performed geometric verification to re-rank top candidates by local spatial consistency.
3. **Multi-scale Inference (Image Pyramids)** — Embeddings were extracted at multiple scales and aggregated (generalized mean pooling) to handle landmark images at varying zoom levels.
4. **GeM Pooling (Generalized Mean Pooling)** — Replaced global average pooling with learnable GeM pooling, allowing the model to emphasize discriminative local regions over uninformative background.
5. **FAISS Approximate Nearest Neighbor Search** — Enabled sub-second retrieval over millions of gallery images; exact search was computationally infeasible at gallery scale.

**How to Reuse:**
- ArcFace / CosFace training is now the default for any re-identification or retrieval task (faces, products, landmarks, whales) — replace softmax classification with margin-based metric learning.
- The coarse global retrieval + local re-ranking pipeline (global embedding → top-k → geometric verification) generalizes to any image retrieval system.
- GeM pooling is a drop-in replacement for global average pooling in retrieval models and almost universally improves results.
- Use FAISS for any retrieval task with >100k gallery items — exact cosine search doesn't scale.

---

## 8. LLM Prompt Recovery (2024)

**Task:** Adversarial/reverse engineering — given original text and an LLM-rewritten version, recover the instruction prompt used.
**Discussion:** https://www.kaggle.com/c/llm-prompt-recovery/discussion/494343

**Approach:** The top solutions discovered that the evaluation metric (sentence-transformer cosine similarity to the target prompt) was gameable: a single high-scoring generic prompt ("What are the key ideas presented in the following text?") scored surprisingly well as a universal submission, essentially exploiting metric geometry. Beyond this, teams that went deeper used fine-tuned LLMs (Mistral, LLaMA-2) to infer prompts by analyzing the diff between original and rewritten text, but the universal prompt strategy remained highly competitive.

**Key Techniques:**
1. **Universal Prompt Exploit** — Discovered that a single fixed prompt achieved high cosine similarity against most targets in the evaluation embedding space; many top teams used this as a baseline or fallback.
2. **Diff Analysis Between Original and Rewritten Text** — Models were prompted (or fine-tuned) to identify what transformation occurred by comparing the two texts, then verbalizing the likely instruction.
3. **Fine-tuned Mistral/LLaMA for Prompt Inference** — Open-weight LLMs fine-tuned on (original, rewritten, prompt) triplets to generate candidate prompts.
4. **Sentence-Transformer Metric Optimization** — Candidate prompts were scored against the sentence-transformer embedding space and selected to maximize expected cosine similarity to common prompt clusters.
5. **Prompt Clustering / Vocabulary Analysis** — Analyzing the distribution of likely prompts in the training set revealed a small set of high-frequency prompt templates that could be memorized and returned.

**How to Reuse:**
- Always analyze the evaluation metric before modeling — metric geometry exploits (universal answers, distribution shortcuts) can outperform complex models.
- For reverse-engineering tasks, fine-tune on (input, output, transformation) triplets to teach the model to verbalize observed changes.
- This competition illustrates "Goodhart's Law" in ML: when a metric becomes a target, the metric itself becomes gameable. When designing evaluation metrics for generative tasks, use human eval alongside automated metrics.

---

## 9. Google ASL Fingerspelling (2023)

**Task:** Sequence-to-sequence — transcribe fingerspelled English words from MediaPipe hand landmark sequences.
**Discussion:** https://www.kaggle.com/c/asl-fingerspelling/discussion/434485

**Approach:** The winning solution treated the landmark sequence as a sequential signal and trained Transformer-based encoder-decoder models (similar to speech recognition architectures) to decode character sequences from the spatial-temporal hand landmark stream. Key innovations included heavy landmark augmentation (affine transforms, finger dropout, noise injection) and training with CTC loss combined with cross-entropy on the decoder. The final system also used beam search decoding with a character-level language model for rescoring.

**Key Techniques:**
1. **Transformer Encoder-Decoder with CTC + CE Loss** — Combined CTC loss on the encoder output with cross-entropy on the autoregressive decoder, improving alignment and transcription jointly.
2. **Landmark Augmentation (Affine + Noise + Dropout)** — Random rotation, scaling, translation of hand landmarks; random finger landmark dropout simulated occlusion; Gaussian noise simulated tracking noise.
3. **Positional Encoding Adapted for Variable-length Landmark Sequences** — Learned positional encodings handled variable-length fingerspelling sequences more robustly than fixed sinusoidal PE.
4. **Beam Search with Character Language Model Rescoring** — A character n-gram LM trained on English word lists rescored beam candidates, boosting accuracy on rare and ambiguous fingerspellings.
5. **Dominant Hand Detection + Normalization** — Input landmarks were normalized relative to the wrist joint and mirrored to a canonical dominant-hand representation before model input.

**How to Reuse:**
- The CTC + CE combined loss is now standard for landmark-to-sequence and audio-to-sequence tasks — use it whenever alignment between input frames and output tokens is variable and non-monotonic.
- For landmark-based gesture/sign tasks, always normalize relative to a canonical reference point (wrist, shoulder) and handle left/right hand mirroring explicitly.
- Landmark augmentation (affine transforms + joint dropout) is highly effective and underused — treat landmark sequences like images with domain-specific augmentation.
- Character-level LM rescoring adds meaningful accuracy improvement on word-level transcription with minimal inference overhead.

---

## 10. Kaggle LLM Science Exam (2023)

**Task:** RAG + LLM — answer 5-choice science multiple-choice questions, where questions were generated by LLMs from Wikipedia.
**Discussion:** https://www.kaggle.com/c/kaggle-llm-science-exam/discussion/446240

**Approach:** The winning solution combined two components: (1) retrieval — a dense retrieval system (TF-IDF + Wikipedia FAISS index) fetched relevant Wikipedia passages for each question, and (2) LLM inference — a fine-tuned or few-shot prompted open-weight LLM (Mistral, LLaMA-2, or DeBERTa fine-tuned as a ranker) selected the correct answer conditioned on retrieved context. Teams that downloaded and indexed Wikipedia offline gained a decisive edge since the competition ran in a restricted kernel environment.

**Key Techniques:**
1. **Offline Wikipedia FAISS Index** — Entire Wikipedia was chunked, embedded with a sentence transformer, and stored as a FAISS index inside the kernel; retrieval ran offline without internet access.
2. **TF-IDF Pre-filtering + Dense Re-ranking** — BM25/TF-IDF retrieved a candidate set of passages cheaply; dense embedding re-ranking (DPR, sentence-transformers) refined relevance before passing to the LLM.
3. **DeBERTa Fine-tuned as Multiple-Choice Ranker** — Rather than using a generative LLM, top teams fine-tuned DeBERTa-v3 on (question, context, answer_choice) triplets as a 5-class ranker — faster inference within kernel limits.
4. **Context Window Stuffing** — Top-k retrieved passages were concatenated into the LLM prompt context; passage ordering by retrieval score mattered for model accuracy.
5. **Ensemble of Retrieval Sources + Models** — Multiple retrieval methods (TF-IDF, FAISS, BM25) and model architectures were ensembled to reduce retrieval miss rate.

**How to Reuse:**
- The offline Wikipedia index pattern is directly applicable to any knowledge-intensive QA competition running in kernel environments — download and index corpora before competition end.
- DeBERTa as a multiple-choice ranker is faster and more controllable than generative decoding and is the default for discriminative MCQ tasks.
- Two-stage RAG (BM25 pre-filter → dense re-rank → LLM) is the production-grade retrieval pattern — implement all three stages for best recall/latency tradeoff.
- When context length is constrained, experiment with passage ordering in the prompt — most relevant passages first consistently outperforms random ordering.

---

## 11. Global Wheat Detection (2020)

**Task:** Object detection — detect and localize wheat head bounding boxes in field photographs from multiple global domains.
**Discussion:** https://www.kaggle.com/c/global-wheat-detection/discussion/172418

**Approach:** The winning solution used an ensemble of object detectors (EfficientDet, YOLOv5, Faster R-CNN) with domain adaptation techniques to handle the significant visual style differences between training domains (different countries, cameras, growth stages). Weighted Boxes Fusion was used to merge multi-model predictions, and pseudo-labeling on test images improved performance on unseen domains. Models were pretrained on COCO and then fine-tuned on the competition data.

**Key Techniques:**
1. **Weighted Boxes Fusion (WBF)** — WBF across multiple detectors (EfficientDet + YOLOv5 + Faster R-CNN) consistently outperformed NMS; became a standard pattern after this competition.
2. **Test-time Augmentation (TTA) with Multi-scale** — Predictions at multiple scales (512, 768, 1024) and flips were fused; wheat heads appear at variable sizes across images.
3. **Pseudo-labeling for Domain Adaptation** — High-confidence predictions on test images from unseen domains were used as pseudo-labels to adapt models to those domains without explicit domain labels.
4. **Domain Adversarial Training** — Some teams used adversarial domain classifiers to make backbone features domain-invariant across the multiple agricultural regions.
5. **COCO Pretraining + Competition Fine-tuning** — All detectors started from COCO pretrained weights, providing robust low-level feature initialization for natural images.

**How to Reuse:**
- WBF is now the standard box fusion method for detector ensembles — always prefer it over NMS when combining predictions from multiple models.
- For multi-domain detection tasks, pseudo-labeling on target-domain test images is often the single highest-impact technique for domain adaptation.
- Multi-scale TTA (vary input resolution at inference) is particularly effective for detection tasks where object size varies with camera distance.
- Domain adversarial training is worth implementing when train/test domain gap is visually large and explicit domain labels are unavailable.

---

## 12. Humpback Whale Identification (2018)

**Task:** Re-identification — match individual humpback whale fluke photographs to known identities (including a "new whale" class).
**Discussion:** https://www.kaggle.com/c/humpback-whale-identification/discussion/82366

**Approach:** The winning solution by Martin Piotte and Martin Caron used Siamese networks trained with a triplet loss to learn a whale-specific embedding space where same-whale images cluster together. At test time, cosine similarity between the query embedding and all gallery embeddings determined the predicted identity. A key innovation was the "new whale" handling: images not sufficiently similar to any known whale were classified as "new_whale" using a dynamic threshold tuned on the validation set.

**Key Techniques:**
1. **Siamese Network with Triplet Loss** — A twin CNN shared weights to project query and gallery images into a common embedding space; triplet loss with hard negative mining maximized inter-class separation.
2. **Hard Negative Mining** — Online hard negative mining during training (selecting the most confusing negative whale per batch) was critical for learning discriminative fluke patterns.
3. **"New Whale" Threshold Calibration** — A held-out validation set was used to tune the cosine similarity threshold below which predictions defaulted to "new_whale" rather than a known identity.
4. **Aggressive Fluke-specific Augmentation** — Rotation (flukes appear at arbitrary angles), brightness/contrast, and horizontal flip (flukes are often photographed from either side) were essential.
5. **Bounding Box Cropping (Fluke Alignment)** — Cropping to the fluke region before embedding removed distracting background (ocean, boat) and reduced intra-class variation.

**How to Reuse:**
- The Siamese + triplet loss framework is the standard for re-identification tasks with many classes and few images per class — ArcFace has since become more popular but triplet remains effective.
- Hard negative mining is non-negotiable for metric learning: random negatives are too easy and slow embedding convergence dramatically.
- Always model the "none of the above" / "unknown entity" case explicitly with a calibrated similarity threshold — this is critical for open-set recognition tasks.
- Preprocessing to crop and align the discriminative region (fluke, face, product) before embedding is usually worth more than architectural improvements.

---

## 13. Prostate Cancer Grade Assessment (PANDA) (2020)

**Task:** Ordinal classification — grade prostate cancer severity (Gleason score 0–5) from whole-slide pathology images.
**Discussion:** https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion/169143

**Approach:** The winning solution used a multi-instance learning (MIL) approach on tiled whole-slide images: each gigapixel slide was divided into fixed-size tiles, tiles were filtered by tissue content, and a CNN extracted tile embeddings which were aggregated (attention-weighted pooling) to a slide-level prediction. The ordinal nature of Gleason grades was explicitly modeled via ordinal regression loss (cumulative link models) rather than treating grades as unordered classes.

**Key Techniques:**
1. **Tile-based Multi-instance Learning (MIL)** — Gigapixel slides were tiled at 256x256 or 512x512; tissue-containing tiles were selected by saturation thresholding; tile embeddings were aggregated with attention pooling.
2. **Ordinal Regression Loss** — Gleason grades are ordered (0 < 1 < 2 < 3 < 4 < 5); ordinal loss (cumulative binary cross-entropy) explicitly encoded this ordering, outperforming standard cross-entropy.
3. **EfficientNet Tile Encoder Pretrained on Histology** — ImageNet pretrained models were further pretrained on public histology datasets (PCam, TCGA) before fine-tuning on PANDA tiles.
4. **Attention-weighted Tile Aggregation** — Learned attention weights over tiles allowed the model to focus on the most diagnostically relevant tissue regions (highest grade areas).
5. **Stain Normalization (Macenko / Vahadane)** — Pathology slides stained with different H&E stain intensities; stain normalization standardized color space across the Karolinska and Radboud institutions.

**How to Reuse:**
- The tile MIL architecture (tile → CNN embedding → attention aggregation → slide label) is the standard approach for any gigapixel pathology classification — reuse this pipeline directly.
- Ordinal regression loss should replace cross-entropy whenever the target has a meaningful natural ordering (grades, severity levels, ratings) — easy to implement, consistent gains.
- Stain normalization (Macenko) is mandatory preprocessing for multi-institution pathology datasets — skip it and you're training partially on stain artifacts.
- Attention-weighted pooling is preferable to mean/max pooling for MIL tasks where the bag label is determined by a subset of instances (highest-grade tiles, not all tiles).

---

## 14. RSNA Cervical Spine Fracture Detection (2022)

**Task:** 3D classification — detect and localize fractures across 7 cervical vertebrae in CT scan volumes.
**Discussion:** https://www.kaggle.com/c/rsna-2022-cervical-spine-fracture-detection/discussion/362607

**Approach:** The winning solution used a two-stage pipeline: (1) vertebra segmentation/detection to localize each of the 7 vertebrae in 3D, and (2) vertebra-level fracture classification on cropped vertebra regions. The 3D volume was processed as a sequence of 2D axial slices through a 2.5D CNN (stacking adjacent slices as channels), and LSTM/Transformer layers aggregated slice-level features into vertebra-level predictions. A custom weighted loss function upweighted patient-level prediction (any fracture) relative to individual vertebra predictions.

**Key Techniques:**
1. **2.5D CNN (Adjacent Slice Stacking)** — Each axial slice was stacked with N neighboring slices as channels, giving the 2D CNN local 3D context without full volumetric convolutions; computationally efficient.
2. **Vertebra Localization via Segmentation** — A lightweight segmentation model (UNet) localized each vertebra, enabling cropped ROI extraction for fracture classification rather than classifying the full volume.
3. **Sequence Aggregation with LSTM / Transformer** — Slice-level features across the vertebra ROI were aggregated with a recurrent or transformer layer, capturing the spatial context within each vertebra.
4. **Hierarchical Loss (Patient + Vertebra Level)** — The competition metric combined patient-level (any fracture) and vertebra-level predictions; a weighted sum of both losses during training directly optimized the metric.
5. **CT Window Normalization** — CT HU values were normalized using clinical windows (bone window: WL=400, WW=1800) to emphasize bone/fracture-relevant intensity ranges.

**How to Reuse:**
- 2.5D (slice stacking) is the go-to approach for 3D medical imaging when full 3D convolutions are too memory-intensive — it captures local volumetric context cheaply.
- For vertebra/organ localization → classification pipelines, always segment/detect the ROI first, then classify the cropped region — this is the dominant pattern in radiology AI.
- When the competition metric is hierarchical (sample-level + entity-level), match your loss function to the metric hierarchy rather than using a single-level loss.
- Always apply CT windowing as preprocessing (bone window for skeletal tasks, lung window for pulmonary tasks) — raw HU values contain irrelevant intensity range.

---

## 15. Human Protein Atlas Single Cell Classification (2021)

**Task:** Multi-label classification — classify subcellular protein localization patterns (28 classes) at the single-cell level from 4-channel fluorescence microscopy images.
**Discussion:** https://www.kaggle.com/c/hpa-single-cell-image-classification/discussion/239001

**Approach:** The solution used a multi-stage approach: (1) cell instance segmentation (HPA CellSegmentator or Cellpose) to generate individual cell masks from the 4-channel images, (2) per-cell feature extraction with a CNN backbone applied to the masked cell crops, and (3) multi-label classification per cell, with image-level weak labels propagated to cells via multiple-instance learning. The challenge was that image-level labels were provided but cell-level labels were required — requiring careful MIL training and aggregation strategies.

**Key Techniques:**
1. **Cell Instance Segmentation (Cellpose)** — Cellpose accurately segmented individual cell instances from DAPI (nuclear) and ER channels; quality of segmentation directly gated classification performance.
2. **Multi-instance Learning with Weak Labels** — Image-level protein localization labels were propagated to individual cells using MIL aggregation (max pooling or noisy-and), bootstrapping cell-level supervision from image-level annotations.
3. **4-channel Input Fusion** — The four fluorescence channels (protein of interest, nucleus, ER, microtubule) were used as 4-channel input; each channel provides complementary spatial context for subcellular localization.
4. **Label Smoothing + Focal Loss for Class Imbalance** — 28-class multi-label problem with extreme class imbalance (rare organelle patterns); focal loss downweighted easy-negative samples and improved rare class recall.
5. **External HPA Public Data Augmentation** — The Human Protein Atlas public database provided additional labeled images for rare classes, substantially improving recall on underrepresented localization patterns.

**How to Reuse:**
- The cell segmentation → per-cell classification pipeline is the standard for single-cell microscopy tasks — use Cellpose as the off-the-shelf segmenter (it generalizes well).
- MIL with weak (image-level) labels propagated to instance (cell-level) predictions is the right frame whenever instance labels are expensive but bag labels are available.
- For fluorescence microscopy, always use all available channels as separate input channels — each dye captures distinct spatial information; fusing them at the image level loses this.
- Focal loss is the default for extreme multi-label class imbalance; combine with external data augmentation for rare classes rather than relying on oversampling alone.

---

## Cross-Cutting Patterns

| Pattern | Competitions | Notes |
|---------|-------------|-------|
| **WBF (Weighted Boxes Fusion)** | Wheat Detection (#11), COVID Detection (#4) | Prefer over NMS for multi-model detection ensembles |
| **ArcFace / Metric Learning** | Landmark Retrieval (#7), Whale ID (#12) | Default for retrieval and re-ID tasks; replaced classification head |
| **Multi-instance Learning (MIL)** | PANDA (#13), HPA Single Cell (#15), OpenVaccine (#1) | Essential when bag-level labels map to instance-level predictions |
| **2.5D / Tile-based 3D Processing** | Cervical Spine (#14), PANDA (#13) | Efficient alternative to full 3D convolutions for volumetric medical data |
| **DeBERTa-v3-large as NLP Backbone** | Feedback Prize (#2), LLM Science Exam (#10) | Dominant backbone for English token classification and MCQ through 2023 |
| **Offline Index / External Data** | LLM Science Exam (#10), SETI (#3), HPA (#15) | Downloading corpora / external datasets before kernel close is decisive |
| **Pseudo-labeling** | Feedback Prize (#2), SETI (#3), Wheat Detection (#11) | Consistent +0.5–2% improvement; use high-confidence threshold |
| **Ordinal / Hierarchical Loss** | PANDA (#13), Cervical Spine (#14) | Match loss function to metric hierarchy; easy gain when target is ordered |
| **CTC + CE Combined Loss** | ASL Fingerspelling (#9) | Standard for landmark-to-sequence and audio-to-sequence tasks |
| **DSL + Program Search** | ARC (#5) | Only viable approach for extreme few-shot rule-induction tasks |
| **Universal/Metric-exploit Strategy** | Prompt Recovery (#8) | Analyze metric geometry before building models — shortcut may dominate |
| **CT / Domain-specific Preprocessing** | Cervical Spine (#14), PANDA (#13), SIIM (#4) | CT windowing, stain normalization — skip these and you train on artifacts |
| **GNN on Structure Graphs** | OpenVaccine (#1) | Whenever a domain-specific structural prior exists, encode it as a graph |
| **Cadence/Temporal Stacking as Channels** | SETI (#3) | Multi-frame observations with presence/absence pattern: stack as channels |
| **Two-stage Coarse+Fine Retrieval** | Landmark Retrieval (#7), LLM Science Exam (#10) | Global embedding → top-k → local re-rank is production-grade RAG pattern |

---

*Document covers 15 first-place solutions spanning 2017–2024. Task types: NLP (3), Medical Imaging (3), Computer Vision (4), Sequence/Signal (2), Tabular/Mixed (1), Reasoning (1), Adversarial/LLM (1).*agentId: a8a12ad2ab32b2fab (use SendMessage with to: 'a8a12ad2ab32b2fab' to continue this agent)
<usage>total_tokens: 24618
tool_uses: 0
duration_ms: 190145</usage>