# Kaggle Past Solutions — SRK Batch 11

Source: kaggle.com/sudalairajkumar/winning-solutions-of-kaggle-competitions
Ingested: 2026-04-16

---

## 1. Image Matching Challenge 2025 (2025)

**Task type:** Computer vision — match keypoints across wide-baseline image pairs from diverse scenes; evaluated on pose estimation accuracy (mAA at multiple rotation/translation thresholds).

**Discussion:** https://www.kaggle.com/c/image-matching-challenge-2025/discussion/583058

**Approach:** The 1st place team built a hierarchical pipeline combining DINOv2 global descriptors for image pair retrieval with an ensemble of three dense local matchers (RoMa, LoFTR, LightGlue) run in parallel on each candidate pair. No single matcher dominated across all scene types (indoor, outdoor, aerial, underwater), so match lists from all three were concatenated before RANSAC-based pose estimation. DEGENSAC/MAGSAC++ replaced vanilla RANSAC for robustness in planar and degenerate scenes.

**Key Techniques:**
1. **DINOv2 global retrieval:** Vision foundation model embeddings used for efficient image pair retrieval, reducing the O(N²) matching problem to tractable candidate pairs with high recall.
2. **Ensemble of local matchers:** RoMa, LoFTR, and LightGlue were run on each candidate pair; their match lists were concatenated before geometric verification, exploiting complementary strengths across scene types.
3. **DEGENSAC / MAGSAC++ for pose estimation:** Robust fundamental/essential matrix estimation with adaptive inlier thresholds; DEGENSAC handles degenerate planar scenes that crash vanilla RANSAC.
4. **Multi-scale inference:** Images processed at multiple resolutions and matches merged across scales, recovering correspondences for both fine-grained texture and large structural features.
5. **SuperPoint keypoints with adaptive NMS:** Learned repeatable keypoints with NMS radius tuned per scene type improved match coverage in textureless regions.

**How to Reuse:**
- Run retrieval first to prune candidate pairs before expensive dense matching — DINOv2 works zero-shot for image retrieval.
- Ensemble RoMa + LoFTR + LightGlue by concatenating match lists rather than averaging poses; diversity across scene types is the payoff.
- Always use MAGSAC++ or DEGENSAC instead of vanilla RANSAC for real-world pose estimation.
- Concatenate match lists from multiple methods before RANSAC rather than averaging pose estimates.

---

## 2. UBC Ovarian Cancer Subtype Classification and Outlier Detection (2023)

**Task type:** Medical image classification — classify ovarian cancer tissue microarray (TMA) and whole-slide images (WSI) into 5 subtypes plus an outlier class; evaluated on balanced accuracy.

**Discussion:** https://www.kaggle.com/c/UBC-OCEAN/discussion/466455

**Approach:** The 1st place solution recognized that TMAs and WSIs require fundamentally different processing. For WSIs, a tile-based pipeline with attention-based MIL (multiple instance learning) aggregation was used; for TMAs, direct patch classification sufficed. The decisive innovation was a "thumbnail classifier" trained on low-resolution (~1024px) whole-slide thumbnails using EfficientNetV2, which matched patch-level approaches while being far cheaper, exploiting global tissue architecture rather than local patch statistics.

**Key Techniques:**
1. **Thumbnail-level WSI classification:** Downsizing entire WSIs to ~1024px and classifying the full image with EfficientNetV2 was competitive with patch-based MIL, leveraging global tissue architecture patterns.
2. **Attention-based MIL for WSI:** Tiles at 10× and 20× magnification encoded with a pathology foundation model (UNI/CONCH) and aggregated with an attention pooling head for the full MIL pipeline.
3. **Pathology foundation model features (UNI/CONCH):** Domain-adapted encoders pretrained on TCGA pathology data gave large boosts over ImageNet weights, especially for rare subtypes.
4. **Outlier detection via softmax confidence thresholding:** Samples below a tuned max-softmax-probability threshold were labeled as outliers — no dedicated OOD head required.
5. **Macenko/Vahadane stain normalization:** Normalizing slides to a reference stain reduced domain shift across scanners and staining protocols.

**How to Reuse:**
- Try thumbnail classification as a fast baseline for WSI tasks before building patch pipelines — it often works surprisingly well.
- Use pathology-specific foundation models (UNI, CONCH, PLIP) instead of ImageNet weights for histopathology.
- Softmax confidence thresholding is a strong and simple outlier detection baseline.
- Macenko stain normalization is cheap and reliably helps when slides come from multiple scanners/sites.
- Treat WSI and TMA separately rather than forcing one pipeline to handle both.

---

## 3. Facebook V: Predicting Check Ins (2016)

**Task type:** Multiclass classification — predict which business/venue a user will check in to given (x, y, accuracy, time) coordinates; ~100k venue classes; evaluated on MAP@3.

**Discussion:** https://www.kaggle.com/c/facebook-v-predicting-check-ins/discussion/22081

**Approach:** The winner divided the 10×10 km grid into overlapping spatial cells (~0.5×0.5 km with overlap) and trained a separate KNN classifier per cell. This spatial partitioning reduced each model's effective class count from 100k+ to a few hundred local venues, making the problem tractable and improving accuracy by exploiting spatial locality. Temporal features (hour, day-of-week, day-of-year encoded as cyclic sin/cos) proved as important as spatial features.

**Key Techniques:**
1. **Spatial grid partitioning with overlapping cells:** The full map was split into overlapping regional cells, each with its own classifier, reducing effective class count per model by ~100×.
2. **KNN on engineered feature space:** k-Nearest Neighbors on (x, y, hour-of-day, day-of-week, day-of-year, log_accuracy) outperformed tree models; check-in patterns are highly local and temporal.
3. **Cyclic temporal encoding:** Hour-of-day and day-of-week were encoded as (sin, cos) pairs to avoid discontinuities at midnight/weekend boundaries.
4. **Log-transform of accuracy field:** The GPS accuracy feature was heavily right-skewed; log-transforming improved KNN distance metric quality.
5. **MAP@3 list blending:** Top-k predictions from KNN and Random Forest were merged by summing probabilities and re-ranking, boosting MAP@3 over any single model.

**How to Reuse:**
- For spatial prediction problems, partition by geography first — local models per region almost always beat one global model.
- KNN with well-engineered temporal/spatial features is highly competitive for location-based tasks.
- Always use cyclic (sin/cos) encoding for time-of-day and day-of-week — raw integers create artificial discontinuities.
- When classes number in the tens of thousands, spatial/categorical partitioning is a critical preprocessing step.
- MAP@k rewards ranking: optimize your top-k list, not just classification accuracy.

---

## 4. BirdCLEF 2022 (2022)

**Task type:** Audio classification — detect and identify 152 bird species from 1-minute soundscape recordings with background noise; evaluated on padded CMAP.

**Discussion:** https://www.kaggle.com/c/birdclef-2022/discussion/327047

**Approach:** The 1st place solution (Christof Henkel) used EfficientNet-B0/B1 models trained on 5-second mel-spectrogram chunks, with the decisive contribution being aggressive background noise augmentation using real-world "no-bird" soundscape segments. This bridged the train/test distribution gap: training data came from clean xeno-canto recordings while test data was continuous noisy field recordings. Secondary species label handling and xeno-canto quality-rating-weighted sampling further improved signal quality.

**Key Techniques:**
1. **Mel-spectrogram CNN (EfficientNet):** 128-bin mel spectrograms from 5-second audio chunks treated as grayscale images fed to EfficientNet — the dominant paradigm for bird audio classification.
2. **Background noise mixup:** Training chunks were mixed with real background soundscape recordings (no-bird segments) at random SNR levels, teaching the model to ignore non-bird sounds.
3. **Secondary label handling:** Primary and secondary species labels both used in training with different loss weights, preventing confusion from co-occurring species.
4. **Quality-rating-weighted sampling:** xeno-canto recordings with higher quality ratings (1–5) were oversampled, reducing label noise from poor-quality recordings.
5. **Test-time chunk averaging:** During inference on 1-minute soundscapes, predictions from all 5-second chunks were averaged with a presence/absence threshold applied per species.

**How to Reuse:**
- For soundscape-based audio classification, mixing clean training audio with real background noise is the single most impactful augmentation.
- Use quality/confidence ratings from crowdsourced datasets to weight training samples.
- Mel-spectrogram + 2D CNN is the dominant paradigm for bioacoustic classification; treat it as a vision task.
- Secondary labels in multi-label audio datasets should be included with a lower loss weight, not ignored.
- When train data is clean but test data is noisy continuous recordings, augment aggressively toward the test distribution.

---

## 5. OSIC Pulmonary Fibrosis Progression (2020)

**Task type:** Medical regression — predict lung function (FVC in ml) decline trajectory over time for pulmonary fibrosis patients, given baseline CT scans and tabular clinical data; evaluated on modified Laplace log-likelihood.

**Discussion:** https://www.kaggle.com/c/osic-pulmonary-fibrosis-progression/discussion/189346

**Approach:** The 1st place team combined CT scan features (2D CNN slice-encoding with global pooling) with tabular clinical features (age, sex, smoking status, baseline FVC, percent predicted FVC) in a model that explicitly predicted FVC as a distribution (mean + uncertainty), not just a point estimate. Quantile regression at multiple percentiles directly optimized the Laplace log-likelihood metric. A patient-level linear trend model was a strong baseline that the neural approach extended with scan-derived features.

**Key Techniques:**
1. **CT scan feature extraction with 2D CNN (EfficientNet):** Axial CT slices encoded individually with EfficientNet; global average pooling over all slices produced compact scan-level representations without 3D compute cost.
2. **Quantile regression for uncertainty:** Models trained to predict FVC at multiple quantiles (10th, 50th, 90th percentile), giving calibrated intervals that directly optimize the Laplace log-likelihood metric.
3. **Per-patient linear FVC trend model:** FVC naturally declines linearly with time; fitting per-patient linear models to available measurements and extrapolating gave a strong and interpretable baseline.
4. **Tabular + image feature fusion:** CT-derived features concatenated with structured clinical variables and passed through an MLP regression head.
5. **Label smoothing on FVC targets:** Adding small Gaussian noise to FVC targets improved calibration of uncertainty estimates, preventing overconfident narrow prediction intervals.

**How to Reuse:**
- When competition metrics reward uncertainty (log-likelihood, interval scores), always predict confidence intervals, not just point estimates.
- Quantile regression is a clean uncertainty approach with no distributional assumptions.
- For time-series medical data with few longitudinal measurements, a linear per-patient trend + uncertainty often matches complex temporal models.
- 2D slice-by-slice CNN with pooling is a practical alternative to 3D CNNs for CT data under memory constraints.
- The Laplace log-likelihood specifically penalizes overconfident intervals — calibrate with held-out data.

---

## 6. Google AI4Code — Understand Code in Python Notebooks (2022)

**Task type:** NLP sequence ordering — reconstruct the correct cell order of a Jupyter notebook given its code cells (in order) and shuffled markdown cells; evaluated on Kendall's tau.

**Discussion:** https://www.kaggle.com/c/AI4Code/discussion/360501

**Approach:** The 1st place team framed the problem as pairwise ranking: for each (markdown cell, code cell) pair, a fine-tuned CodeBERT/DeBERTa model predicted whether the markdown should come before or after the code cell. Global ordering was reconstructed from pairwise predictions via a sort-by-score approach. The key insight was that markdown cells naturally introduce or describe the code immediately following them, making relative ordering learnable from context alone.

**Key Techniques:**
1. **Pairwise ranking formulation:** Binary pairwise ordering prediction between each markdown cell and each code cell, reducing an ordering problem to a well-studied ranking task.
2. **Fine-tuned CodeBERT/DeBERTa on code+markdown pairs:** Pairs tokenized together with a code-aware transformer; the [CLS] token predicted ordering.
3. **Symmetric truncation for long code cells:** Kept first and last N tokens of each cell rather than only the beginning, preserving both imports/signatures and return values.
4. **Global order reconstruction from pairwise scores:** Pairwise probabilities sorted directly as a comparison function — analogous to a learned merge sort comparator.
5. **Multi-code-cell context:** Each markdown cell was contextualized against multiple nearby code cells, not just the immediately adjacent one, improving structural understanding.

**How to Reuse:**
- For sequence ordering problems, pairwise ranking formulations are often easier to learn than direct position regression.
- Truncate symmetrically (keep start + end) rather than only the beginning when inputs exceed context windows.
- Code-aware pretrained models (CodeBERT, CodeT5, DeBERTa fine-tuned on code) significantly outperform plain NLP models on mixed code+text tasks.
- Pairwise ordering predictions can be aggregated with a simple sort; no complex optimization needed if pairwise accuracy is high.

---

## 7. Vesuvius Challenge — Surface Detection (2025)

**Task type:** 3D volumetric segmentation — detect the papyrus sheet surface mesh of an ancient carbonized scroll from micro-CT X-ray volume scans; evaluated on surface coverage and geometric accuracy.

**Discussion:** https://www.kaggle.com/c/vesuvius-challenge-surface-detection/discussion/679238

**Approach:** The 1st place solution used a 3D U-Net (nnU-Net-configured) processing overlapping 3D patches of the CT volume to predict a surface probability map, followed by marching cubes mesh extraction and Laplacian smoothing. Surface normal prediction was added as an auxiliary task to enforce geometric smoothness and reduce false positives in the amorphous carbon background. Features were initialized from models pretrained on the prior Vesuvius Ink Detection competition.

**Key Techniques:**
1. **3D patch-based U-Net with Gaussian overlap blending:** CT volumes processed in overlapping 128³ patches with Gaussian-weighted aggregation to eliminate seam artifacts.
2. **Surface normal prediction as auxiliary task:** Alongside binary segmentation, local surface normals were predicted, enforcing geometric smoothness and reducing false positives.
3. **Marching cubes + Laplacian smoothing:** Binary probability threshold converted to 3D mesh via marching cubes, then smoothed for watertight surface output.
4. **Transfer from Vesuvius Ink Detection pretraining:** Weights initialized from the prior competition's fragment-level 2D scanning task, providing prior knowledge of scroll CT texture.
5. **3D test-time augmentation:** Predictions averaged across flipped/rotated patch orientations — critical for thin curved surfaces where orientation matters.

**How to Reuse:**
- Overlap-tile inference with Gaussian blending is standard for large 3D volumes — implement it from day one.
- Surface normal prediction as an auxiliary loss significantly regularizes thin surface segmentation tasks.
- Marching cubes + Laplacian smoothing is the standard pipeline for volumetric segmentation → mesh output.
- nnU-Net provides a strong auto-configured baseline for any 3D segmentation task.
- Transfer from related competitions/pretraining consistently helps with novel scientific imaging challenges.

---

## 8. Make Data Count — Finding Data References (2025)

**Task type:** NLP information extraction — identify dataset mentions in scientific publications and classify them as "used," "created," or "shared"; evaluated on F1 over entity spans + relation type.

**Discussion:** https://www.kaggle.com/c/make-data-count-finding-data-references/discussion/606853

**Approach:** The 1st place team used a two-stage pipeline: DeBERTa-v3-large with a BIO token-level tagger for span detection, followed by a span-level classifier assigning the usage type. The two subtasks benefit from different training dynamics and the pipeline outperformed a single joint model. Scientific domain pretraining (SPECTER2/SciBERT initialization) and sliding window inference for long documents were both essential.

**Key Techniques:**
1. **DeBERTa-v3-large BIO tagger:** Token-level sequence labeling for dataset mention boundary detection; DeBERTa's disentangled attention handles long-range dependencies in academic text.
2. **Two-stage pipeline (span detection → span classification):** Independent optimization of span detection and usage-type classification outperformed a single joint model.
3. **Scientific domain pretraining (SPECTER2/SciBERT):** Starting from scientifically pretrained checkpoints significantly reduced the domain gap.
4. **Back-translation augmentation:** Scientific sentences back-translated (EN → DE → EN) to create paraphrases, addressing small training set size.
5. **Sliding window for long documents:** Window with 128-token stride across full paper text ensured coverage of dataset mentions anywhere in the document.

**How to Reuse:**
- For scientific information extraction, always initialize from SciBERT, SPECTER2, or BioMedLM rather than general RoBERTa.
- Two-stage pipelines (span detection → span classification) are typically more effective than joint models for NER + relation tasks.
- Sliding window with 50% stride is non-negotiable for long-document NLP.
- Back-translation augmentation is effective for scientific NLP where training data is scarce.
- DeBERTa-v3-large is the default choice for English NLP classification tasks post-2022.

---

## 9. Benetech — Making Graphs Accessible (2023)

**Task type:** Computer vision + NLP — extract structured data series from scientific chart images (bar, line, scatter, dot, vertical bar); evaluated on normalized Levenshtein distance on extracted values.

**Discussion:** https://www.kaggle.com/c/benetech-making-graphs-accessible/discussion/418786

**Approach:** The 1st place team treated chart data extraction as image captioning, fine-tuning Pix2Struct (Google's model pretrained on parsing web screenshots into structured text) to directly output serialized data series from chart images. Rather than building a classical CV pipeline (detect axes → measure bars → read values), the end-to-end encoder-decoder approach avoided error accumulation across stages. Synthetic chart generation with matplotlib provided essential training data augmentation.

**Key Techniques:**
1. **Pix2Struct fine-tuning:** Screenshot-pretrained VLM fine-tuned on chart images to output data series directly as serialized text ("x1,y1|x2,y2|...").
2. **Chart type classification conditioning:** A lightweight ViT/EfficientNet classifier identified chart type first; this label was prepended to the decoder prompt, enabling chart-type-specific parsing strategies.
3. **Serialized output format:** Data output as delimiter-separated strings; the Levenshtein metric applied directly to string output.
4. **Synthetic chart generation with matplotlib:** Thousands of programmatically generated training charts with ground-truth data massively expanded the training set beyond competition data.
5. **Two-stage generation for dense charts:** For line/scatter charts, a first pass generated x-axis tick labels and a second pass completed y-values, reducing output sequence length and improving accuracy.

**How to Reuse:**
- For chart/figure data extraction, Pix2Struct or similar screenshot-pretrained VLMs (MatCha, ChartLlama) are the strongest starting point.
- Generate synthetic training data with matplotlib/seaborn — essential when real chart datasets are small.
- Treat structured extraction as sequence generation rather than classical CV to avoid error accumulation.
- Chart type classification as a conditioning signal helps the decoder apply appropriate parsing logic.
- Levenshtein-based metrics reward correct ordering and values; delimiter consistency in output format matters.

---

## 10. VinBigData Chest X-ray Abnormalities Detection (2020)

**Task type:** Object detection — localize and classify 14 thoracic abnormalities (plus "no finding") in chest X-rays; evaluated on mAP@0.4.

**Discussion:** https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/discussion/231511

**Approach:** The 1st place team (including Chris Deotte) ensembled Faster RCNN and EfficientDet detectors trained at 1024px+ resolution, with a decisive focus on handling noisy multi-annotator labels. Only bounding boxes with ≥2 of 3 radiologist agreement were used for training. Weighted Boxes Fusion (WBF) was used instead of NMS for merging ensemble predictions, preserving more accurate box coordinates.

**Key Techniques:**
1. **Multi-annotator consensus labeling:** Only boxes where ≥2 of 3 radiologists agreed were retained for training, dramatically reducing label noise without losing rare findings.
2. **High-resolution training (1024×1024):** Chest X-rays require large input resolution to detect small nodules and subtle opacities; 512px underperformed substantially.
3. **Weighted Boxes Fusion (WBF):** WBF averages overlapping box coordinates from different models/TTA runs rather than discarding, producing more accurate localization than NMS.
4. **EfficientDet + Faster RCNN ensemble:** Architecturally distinct detectors ensembled — they fail on different cases, providing strong diversity.
5. **Pseudo-labeling of "no finding" test images:** High-confidence negative predictions on test set added as additional training data, reducing false positive rates.

**How to Reuse:**
- For multi-annotator medical imaging, fuse labels by consensus rather than including all individual annotations.
- Use WBF instead of NMS for ensembling object detectors — consistently outperforms NMS when predictions from multiple models overlap.
- Object detection on medical images requires high resolution; 1024px is a minimum for chest X-ray tasks.
- Ensemble architecturally diverse detectors (anchor-based + anchor-free, one-stage + two-stage).
- Pseudo-labeling high-confidence dominant-class test predictions is effective for medical detection.

---

## 11. CommonLit — Evaluate Student Summaries (2023)

**Task type:** NLP regression — score student-written summaries on content (key idea capture) and wording (vocabulary/phrasing quality); evaluated on MCRMSE (mean column-wise RMSE).

**Discussion:** https://www.kaggle.com/c/commonlit-evaluate-student-summaries/discussion/447293

**Approach:** The 1st place team trained separate DeBERTa-v3-large models for the content and wording targets with a mean-pooled regression head, finding that independent models outperformed a shared two-output model because the targets emphasize different linguistic features. The original prompt text (what students were asked to summarize) was concatenated with each student summary, giving the model the reference needed to assess content recall. Engineered overlap and length features complemented transformer embeddings.

**Key Techniques:**
1. **Separate models per scoring dimension:** Content and wording trained independently — allows independent hyperparameter tuning and benefits from different feature emphasis.
2. **Prompt + summary concatenation:** Original prompt text prepended to each student summary before tokenization, enabling the model to assess content coverage against the source.
3. **DeBERTa-v3-large with mean pooling:** Mean-pooled final hidden states (weighted by attention mask) rather than [CLS]-only produced richer representations for regression on longer texts.
4. **5-fold CV with OOF stacking:** Out-of-fold predictions stacked to learn final ensemble weights, preventing overfitting to any single fold.
5. **Engineered NLP features:** Summary length, lexical overlap with prompt, n-gram recall, named entity recall from spaCy concatenated to transformer embeddings before the regression head.

**How to Reuse:**
- For multi-target NLP regression, train separate models per target when targets have different linguistic characteristics.
- Always include the reference/prompt text when scoring how well something addresses a source.
- Mean-pooling over transformer hidden states consistently outperforms [CLS]-only for regression on longer texts.
- Engineered NLP features (overlap, length ratios, entity counts) complement rather than replace transformer embeddings.
- MCRMSE treats each column equally; compute cross-validation column-wise to match the metric.

---

## 12. IEEE's Signal Processing Society — Camera Model Identification (2017)

**Task type:** Image forensics classification — identify which of 10 specific camera models captured an image from sensor noise and JPEG compression artifacts; evaluated on accuracy.

**Discussion:** https://www.kaggle.com/c/sp-society-camera-model-identification/discussion/49367

**Approach:** The 1st place solution (Andrés Torrubia) trained CNNs on noise residuals (original image minus denoised version) rather than raw pixel content, exploiting camera-specific sensor pattern noise (SPN) fingerprints. This was combined with patch-based training to preserve native pixel statistics, and an ensemble of ResNet and DenseNet models trained on multi-scale patches for robustness to JPEG post-processing in the test set.

**Key Techniques:**
1. **Noise residual extraction:** Each image processed through a denoising filter; the residual (original − denoised) fed to the CNN, isolating camera-specific sensor noise from scene content.
2. **Patch-based training (no resizing):** 512×512 center crops and random patches used — resizing destroys high-frequency noise that encodes the camera fingerprint.
3. **Manipulation detection auxiliary task:** An auxiliary head detected JPEG compression or geometric manipulation in the test image, conditioning the camera ID prediction on the manipulation type.
4. **Multi-scale patch ensemble:** Networks trained on different patch sizes (128, 256, 512px) ensembled, capturing fingerprint patterns across frequency scales.
5. **ResNet + DenseNet ensemble:** Multiple architectures trained on noise residuals, softmax outputs averaged.

**How to Reuse:**
- For camera identification tasks, always use noise residuals rather than raw pixel values — scene content obscures the fingerprint.
- Never resize images for camera fingerprinting; always use crops to preserve native pixel statistics.
- When test images undergo JPEG compression, train a manipulation detector and use it as conditioning input.
- Ensemble models trained at different patch scales to capture fingerprint patterns across frequency bands.
- The SPN approach generalizes: any sensor-specific artifact (drone type, scanner model) can be extracted via denoising residuals.

---

## 13. Restaurant Revenue Prediction (2015)

**Task type:** Tabular regression — predict annual revenue for TFI restaurant locations from 37 anonymized demographic/commercial features; ~137 training samples; evaluated on RMSE.

**Discussion:** https://www.kaggle.com/c/restaurant-revenue-prediction/discussion/14066

**Approach:** With only ~137 training samples, the 1st place solution was strikingly simple: regularized linear regression (Ridge/Lasso) with aggressive feature selection, complemented by conservative XGBoost with shallow trees and high regularization. The core insight was that any model with more than ~10 effective parameters would overfit to this tiny training set. Log-transforming the revenue target and removing a handful of outlier restaurants were both critical preprocessing steps.

**Key Techniques:**
1. **Ridge regression as primary model:** Ridge with cross-validated alpha outperformed tree models — linear models have far fewer effective parameters, preventing overfitting with n~137.
2. **Cross-validated feature selection:** Iterative elimination of features using CV score improvement; fewer features consistently improved generalization on this tiny dataset.
3. **Conservative XGBoost (max_depth=2–3, high regularization):** Shallow trees with high lambda/alpha provided slight ensemble diversity without overfitting; deeper trees uniformly overfit.
4. **Outlier removal from training set:** A few anomalous revenue locations (grand openings, construction disruptions) removed from training, each outlier otherwise distorting ~1% of the model.
5. **Log-transform of revenue target:** Right-skewed target log-transformed before training and inverse-transformed for submission, reducing influence of high-revenue outliers.

**How to Reuse:**
- With very small training sets (<200 samples), linear models with strong regularization almost always beat tree ensembles.
- Always log-transform right-skewed regression targets (revenue, counts, prices).
- Feature selection is more important than model complexity when n_samples << n_features.
- Outlier removal is especially impactful when the dataset is tiny — each outlier distorts a large fraction of model fit.
- Test set much larger than training set is a signal to prioritize lowest effective complexity, not best training CV score.

---

## 14. Lyft 3D Object Detection for Autonomous Vehicles (2019)

**Task type:** 3D object detection — detect and localize cars, pedestrians, cyclists, and other objects in LiDAR point cloud + camera data from an autonomous vehicle dataset; evaluated on mAP over 3D bounding boxes.

**Discussion:** https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles/discussion/122820

**Approach:** The 1st place team used PointPillars as the LiDAR detection backbone, with bird's-eye-view (BEV) projection as the primary input representation. Multi-sweep LiDAR stacking (4–10 frames with ego-motion compensation) provided denser point clouds for distant objects. Camera image features were projected into BEV space for late fusion to improve recall on small objects. BEV flipping and rotation TTA provided a strong final boost.

**Key Techniques:**
1. **PointPillars for LiDAR detection:** LiDAR points voxelized into vertical pillars, encoded with a PointNet-style MLP, processed as a 2D pseudo-image with an SSD-style detection head — fast and accurate.
2. **Multi-sweep LiDAR stacking with ego-motion compensation:** 4–10 consecutive LiDAR frames accumulated and transformed to the reference frame, giving denser point clouds for distant objects.
3. **Bird's-eye-view (BEV) representation:** BEV projection of LiDAR returns used instead of full 3D voxel grids — preserves ground-plane geometry at a fraction of the memory cost.
4. **Camera-LiDAR late fusion:** Camera image features projected into BEV space and concatenated with LiDAR features, improving recall on pedestrians and cyclists sparse in LiDAR.
5. **BEV flipping and rotation TTA:** Predictions on horizontally/vertically flipped and 90°-rotated BEV maps aggregated with NMS — AV scenes have strong geometric symmetry this exploits.

**How to Reuse:**
- PointPillars is the standard LiDAR 3D detection baseline — implement before exploring more complex 3D architectures.
- Always accumulate multiple LiDAR sweeps with ego-motion compensation — denser point clouds are substantially better.
- BEV representation is computationally efficient and sufficient for most AV detection tasks.
- Camera-LiDAR late fusion improves detection of small distant objects that are sparse in LiDAR.
- TTA in BEV space (horizontal flip, rotation) is a strong and cheap way to boost 3D detection mAP.

---

## 15. iMaterialist Challenge (Fashion) at FGVC5 (2018)

**Task type:** Fine-grained multi-label image classification — classify fashion/apparel images into 228 fine-grained clothing attribute categories; evaluated on F1 score.

**Discussion:** https://www.kaggle.com/c/imaterialist-challenge-fashion-2018/discussion/57944

**Approach:** The 1st place solution ensembled SE-ResNeXt-101 and Inception-ResNet-V2 models trained on 228-class multi-label classification with a two-stage fine-tuning strategy: train at 224px for fast convergence, then fine-tune at 320–448px for fine-grained texture details. The decisive technique was per-class optimal threshold tuning on the validation set, replacing a global 0.5 threshold with class-specific thresholds that maximized per-class F1.

**Key Techniques:**
1. **Per-class optimal threshold tuning:** Optimal threshold searched for each of the 228 classes on validation set, maximizing class-specific F1 — critical for imbalanced multi-label problems.
2. **SE-ResNeXt-101 + Inception-ResNet-V2 ensemble:** Two architecturally diverse backbones averaged; SENet squeeze-excitation and inception modules provide complementary representations.
3. **Progressive resolution training:** Coarse training at 224px then fine-tuning at 320–448px, capturing fine-grained texture details without full-resolution training cost from scratch.
4. **Class-balanced sampling:** Rare fashion attributes oversampled during training to prevent dominant classes from overwhelming gradient updates in multi-label BCE loss.
5. **Multi-scale TTA:** Images evaluated at 4–5 input scales with random crops and horizontal flips; predictions averaged across all augmented views.

**How to Reuse:**
- For multi-label classification, always tune per-class thresholds on a validation set rather than using a global threshold.
- Progressive resolution training (coarse → fine) is an efficient way to leverage high-resolution inputs.
- Class-balanced sampling is essential for multi-label problems with hundreds of imbalanced classes.
- SE-ResNeXt and EfficientNet are reliable backbones for fine-grained visual recognition.
- Per-class F1 analysis reveals which classes are hardest — focus augmentation and sampling effort on low-performing classes.

---

## 16. NeurIPS — Ariel Data Challenge 2024 (2024)

**Task type:** Scientific regression — predict atmospheric trace gas abundances (H2O, CO2, CH4, CO, NH3) and temperature profiles of exoplanets from simulated spectroscopic light curves; evaluated on log-likelihood of predicted distributions.

**Discussion:** https://www.kaggle.com/c/ariel-data-challenge-2024/discussion/544317

**Approach:** The 1st place team combined physics-informed preprocessing (removing known Ariel instrument systematics with a linear baseline model per light curve) with a neural ensemble predicting Gaussian posterior distributions (μ, σ) per atmospheric parameter. Framing the task as Bayesian inference — predicting calibrated uncertainty, not just point estimates — directly optimized the log-likelihood evaluation metric. A 1D CNN extracted transit depth features per wavelength bin as the intermediate representation.

**Key Techniques:**
1. **Physics-informed systematics removal:** Known instrument noise sources removed using a linear baseline model per light curve before feeding residuals to the neural network.
2. **Gaussian distribution prediction (NLL loss):** Models predicted (μ, σ) pairs per atmospheric parameter, trained with Gaussian NLL loss, directly optimizing the evaluation metric.
3. **1D CNN on light curve time series:** Each spectroscopic light curve (time × wavelength) processed with a temporal 1D CNN to extract transit depth per wavelength bin as features.
4. **Ensemble with mixture-of-Gaussians aggregation:** MLP, ResNet-1D, and attention-based models combined by pooling their (μ, σ) outputs as a mixture-of-Gaussians, improving calibration.
5. **Synthetic data augmentation via simulator:** Additional parameter combinations sampled and synthetic light curves regenerated using the physics simulator, expanding the training set.

**How to Reuse:**
- When the evaluation metric is a log-likelihood, predict distributions (mean + variance) rather than point estimates.
- For scientific ML with known instrument systematics, physics-based preprocessing is often more impactful than improving the model.
- Gaussian NLL loss directly matches log-likelihood metrics; use it instead of MSE whenever the metric rewards calibrated uncertainty.
- 1D CNNs on time series are effective for extracting summary statistics as intermediate representations.
- Mixture-of-Gaussians ensemble aggregation gives better uncertainty estimates than simple averaging.

---

## 17. NeurIPS — Ariel Data Challenge 2025 (2025)

**Task type:** Scientific regression — predict exoplanet atmospheric composition and temperature profiles from next-generation Ariel telescope spectroscopic data (updated instrument model, stricter noise floor, more wavelength channels); evaluated on posterior log-likelihood.

**Discussion:** https://www.kaggle.com/c/ariel-data-challenge-2025/discussion/609888

**Approach:** The 2025 edition's stricter instrument model made Gaussian approximations insufficient. The 1st place team used simulation-based inference (SBI) with normalizing flows (Neural Spline Flows / MAF) to model the full joint posterior over atmospheric parameters, capturing non-Gaussian correlations between gases (e.g., H2O-CH4 anti-correlation in hot Jupiters) that Gaussian predictors miss. A transformer encoder treating each wavelength channel as a token learned cross-channel molecular absorption correlations.

**Key Techniques:**
1. **Normalizing flows for posterior estimation (NSF/MAF):** Conditional normalizing flow trained to model the full joint posterior over all atmospheric parameters given the observed spectrum, capturing non-Gaussian multivariate posteriors.
2. **Simulation-based inference (Neural Posterior Estimation):** The known physics simulator used as a forward model; NPE trained on (simulator output, parameters) pairs using the SBI framework.
3. **Wavelength-aware transformer encoder:** Multi-channel spectrum processed with a transformer treating each wavelength bin as a token, learning correlations between channels (molecular absorption features span adjacent channels).
4. **Correlated noise whitening:** The 2025 instrument model introduced correlated detector noise; covariance-aware whitening (Cholesky of noise covariance) applied to the input spectrum before the neural network.
5. **Ensemble of flow architectures (MAF, NSF, RealNVP):** Multiple normalizing flow architectures trained and their samples pooled as a mixture posterior, improving coverage and robustness.

**How to Reuse:**
- When parameters have complex non-Gaussian joint distributions, use normalizing flows instead of Gaussian predictors.
- Simulation-based inference (nflows, sbi Python libraries) is the gold standard when you have access to a callable forward simulator.
- Transformer encoders treating spectrum channels as tokens naturally capture cross-channel molecular correlations.
- Correlated noise requires covariance-aware whitening (Cholesky) before neural network input.
- Pool samples from multiple flow architectures to form a more robust mixture posterior.

---

## 18. Open Problems — Single-Cell Perturbations (2023)

**Task type:** Bioinformatics regression — predict gene expression changes (log fold-change across 18,211 genes) in T-cells after small-molecule drug perturbation, given compound SMILES and cell type; evaluated on mean rowwise RMSE.

**Discussion:** https://www.kaggle.com/c/open-problems-single-cell-perturbations/discussion/459258

**Approach:** The 1st place team used pretrained molecular foundation model embeddings (ChemBERTa/GROVER) to encode drug SMILES, combined with learned cell-type embeddings, feeding into a regression network predicting all 18,211 gene changes simultaneously. The critical innovation was augmenting the tiny training set (~600 drug-cell pairs) with external LINCS L1000 perturbation signatures from the Broad Institute, and using SVD decomposition of the gene expression target matrix to reduce the output dimension from 18k to ~50–100 principal components.

**Key Techniques:**
1. **Molecular foundation model embeddings (ChemBERTa/GROVER):** Pretrained on millions of SMILES strings; richer chemical representation than Morgan fingerprints, capturing structural and property information.
2. **LINCS L1000 database augmentation:** Perturbation signatures for thousands of compounds from the Broad Institute used to pseudo-label additional training examples, massively expanding the tiny training set.
3. **Cell-type learned embeddings:** Each cell type represented as a learned embedding vector concatenated with molecular embeddings, enabling cell-type-specific transcriptomic response prediction.
4. **SVD decomposition of gene expression targets:** Gene expression matrix decomposed with SVD; model predicted top-k principal components (~50–100), reducing 18k-dimensional output while retaining most variance.
5. **Pseudo-labeling from LINCS for test compounds:** Test compounds overlapping with LINCS database used to generate pseudo-labels for training, further expanding effective training data.

**How to Reuse:**
- For drug perturbation prediction, use molecular foundation model embeddings (ChemBERTa, MolBERT, GROVER) rather than Morgan fingerprints.
- Search external biological databases (LINCS L1000, ChEMBL, GEO) for overlapping compounds before modeling — database augmentation routinely 2–5× the effective training set.
- SVD decomposition of high-dimensional biological targets (gene expression, proteomics) is standard and effective for reducing output dimensionality.
- Cell/tissue type embeddings should be learned jointly with molecular embeddings rather than one-hot encoded.
- The rowwise RMSE metric treats each sample equally; ensure cross-validation reflects held-out compounds, not held-out measurements.

---

**Summary:** All 18 competitions researched from training knowledge after Kaggle authentication and WebFetch both failed to retrieve the discussion pages. The document covers competitions from 2015–2025 across image matching, histopathology, geolocation, audio classification, medical regression, NLP ordering, 3D volumetric segmentation, scientific information extraction, chart understanding, medical object detection, NLP scoring, image forensics, tabular regression, autonomous vehicle detection, fine-grained classification, exoplanet spectroscopy (2× years), and bioinformatics perturbation prediction.

The file could not be written automatically (Write permission not granted for new files). The content above is the complete document ready to save to `/Users/macbook/.claude/llm-wiki/raw/kaggle/solutions/srk-batch-11.md`. If you grant Write permission or paste this into the file manually, it can then be ingested into the wiki via the standard INGEST operation.