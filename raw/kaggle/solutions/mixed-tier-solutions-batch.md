# Kaggle Past Solutions — Mixed Tier Batch

Source: ndres.me/kaggle-past-solutions catalog
Ingested: 2026-04-16

---

## Instant Gratification — 1st Place

**Competition:** Binary classification on synthetic tabular data generated from Gaussian mixture models.
**Writeup:** https://www.kaggle.com/c/instant-gratification/discussion/96549

### Approach
- Winners recognized data was synthetically generated from class-conditional multivariate Gaussians
- Quadratic Discriminant Analysis (QDA) is the Bayes-optimal classifier for this data-generating process
- Per-subgroup modeling via the `wheezy-copper-turtle-magic` column (512 independent subgroups)

### Key Techniques
1. **QDA over tree models**: Bayes-optimal when each class has its own covariance matrix. Unlike LDA (shared covariance), QDA produces quadratic decision boundaries
   ```python
   from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
   qda = QuadraticDiscriminantAnalysis(reg_param=0.5)
   ```
2. **Per-subgroup modeling**: The magic column divided data into ~512 independently generated subgroups. One QDA per subgroup eliminated cross-group noise
3. **Variance-based feature selection**: Within each subgroup only ~40 of 512 features were informative. Features with variance > 1.5 kept; rest discarded
4. **Pseudo-labeling**: High-confidence test predictions added back as training data; exploits the clean generative structure
5. **Regularization tuning**: `reg_param` shrinks per-class covariance toward identity; tuned per subgroup to avoid singular covariance matrices

### How to Reuse
- When tabular data looks "too clean" or synthetic, test probabilistic classifiers (QDA, LDA, Naive Bayes) before XGBoost
- Always check if a categorical column creates natural subgroups that should be modeled independently
- Pseudo-labeling is especially effective when the model matches the data-generating process
- Variance-based feature selection is a cheap first pass for high-dimensional data with noise features

---

## YouTube-8M Video Understanding 2019 — 1st Place

**Competition:** Multi-label video classification (1000+ labels) using pre-extracted audio/visual features.
**Writeup:** https://www.kaggle.com/c/youtube8m-2019/discussion/112869

### Approach
- Large ensemble of temporal models on pre-extracted frame-level and video-level features
- Core architecture: NeXtVLAD aggregation + gated attention + context gating
- Knowledge distillation from video-level models to segment-level models (2019 introduced temporal localization)

### Key Techniques
1. **NeXtVLAD aggregation**: Evolution of NetVLAD — decomposes high-dimensional clustering into groups, reducing parameters while maintaining representational power. Soft-assigns frames to K clusters across G groups
2. **Context gating**: Learned sigmoid gating after aggregation: `gate = sigmoid(W @ x + b); x_gated = x * gate`. Feature-wise attention mechanism
3. **Mixture of Experts (MoE) classifier**: Multiple expert networks combined via gating network; allows specialization across 1000+ labels
4. **Knowledge distillation**: Video-level models (all frames) → teacher labels → segment-level models (5-second clips). Transfers global context to local predictions
5. **20+ model ensemble**: Combined NeXtVLAD, bidirectional LSTMs, Transformer encoders via learned stacking

### How to Reuse
- NeXtVLAD for aggregating variable-length sequences into fixed-size representations (video, audio, documents)
- Context gating is a lightweight plug-in improvement for any feature pipeline bottleneck layer
- MoE classifiers for extreme multi-label settings (thousands of classes)
- Knowledge distillation from global-context to local-context models for temporal localization

---

## Generative Dog Images — 1st Place

**Competition:** Generate realistic dog images; evaluated by FID and MiFID (penalizes memorization).
**Writeup:** https://www.kaggle.com/c/generative-dog-images/discussion/106324

### Approach
- BigGAN with spectral normalization and truncation trick
- MiFID metric management: optimal truncation threshold + curated selection pipeline
- Transfer learning from ImageNet-pretrained GAN, fine-tuned on dog breeds

### Key Techniques
1. **BigGAN + spectral normalization**: Class-conditional GAN with spectral norm on all layers for training stability. Self-attention at 64x64 captures global structure
2. **Truncation trick**: At inference, sample z from truncated normal. Lower truncation = higher quality but less diversity. Sweep values to optimize MiFID
3. **MiFID-aware selection**: Generate thousands of candidates, filter out those too similar to training (memorized) or too dissimilar (low quality)
4. **Transfer learning for GANs**: Pre-train on full ImageNet → fine-tune on dog breeds. Much better than training from scratch on small dataset
5. **Progressive growing + multi-scale discriminator**: Resolution increased during training (64→128→256)

### How to Reuse
- Always sweep truncation values rather than using a fixed value for GAN inference
- When metrics penalize memorization, build a selection pipeline that optimizes quality-diversity tradeoff
- GAN transfer learning (large dataset → fine-tune) is highly effective and underutilized
- Spectral normalization should be default for discriminator layers

---

## Cervical Cancer Screening — 1st Place (Michael & Giulio)

**Competition:** Classify cervical types (Type 1/2/3) from photographs for cancer screening.
**Interview:** http://blog.kaggle.com/2016/02/26/genentech-cervical-cancer-screening-winners-interview-1st-place-michael-giulio/

### Approach
- Transfer learning with VGG/ResNet/Inception pre-trained on ImageNet
- Two-stage: localize cervix region → classify cervix type
- Heavy augmentation critical for tiny dataset (<2000 images)

### Key Techniques
1. **Two-stage pipeline**: First model localizes cervix bounding box, second model classifies cropped region. Removes irrelevant background noise
2. **Heavy augmentation**: Random rotation 0-360°, flips, zoom, color jitter, elastic deformations — essential with <2000 images
3. **Multi-architecture ensemble**: VGG-16 + ResNet-50 + InceptionV3 averaged. More diversity from different architectures than multiple seeds
4. **Label noise handling**: Manual inspection of high-loss samples to remove mislabeled images; label smoothing in later epochs
5. **Stratified K-fold with oversampling**: Class imbalance addressed via stratification + minority class oversampling

### How to Reuse
- For small medical imaging datasets: two-stage (localize then classify) almost always beats end-to-end
- Under 5000 images, augmentation is the primary regularizer
- Label noise is endemic in medical imaging — always inspect high-loss samples
- Multi-architecture ensembles > multi-seed same-architecture ensembles for diversity

---

## Right Whale Recognition — 2nd Place (Felix Lau)

**Competition:** Identify individual North Atlantic right whales from aerial photos via unique callosity patterns.
**Code:** https://github.com/felixlaumon/kaggle-right-whale

### Approach
- Multi-stage: detect whale head → align via keypoints → extract features → classify individual
- Extreme class imbalance: 447 whale IDs, some with only 1 training image
- Transfer learning + discriminative learning rates

### Key Techniques
1. **Keypoint-based alignment**: CNN predicts bonnet + blowhead keypoints → affine transform normalizes head rotation and scale. Critical for viewpoint-invariant matching
2. **Few-shot handling**: Extreme augmentation (affine, brightness, contrast, flips) + 10-way TTA for whales with 1-3 images
3. **Discriminative learning rates**: Freeze early pretrained layers, progressively unfreeze with smaller LRs:
   ```python
   optimizer = SGD([
       {'params': model.features[:10].parameters(), 'lr': 1e-5},
       {'params': model.features[10:].parameters(), 'lr': 1e-4},
       {'params': model.classifier.parameters(), 'lr': 1e-3},
   ])
   ```
4. **Semi-supervised localization**: Manually annotate bounding boxes for subset → train localizer → auto-annotate rest
5. **Multi-crop inference**: Multiple crops at different scales around detected head, softmax averaged

### How to Reuse
- For fine-grained recognition (faces, animals, products): keypoint alignment before classification yields large gains
- Discriminative learning rates should be the default fine-tuning strategy
- Few-shot classes: combine heavy augmentation with TTA — both are critical
- Semi-supervised annotation (label subset → train → auto-label rest) is practical for competitions and production

---

## Higgs Boson Machine Learning Challenge — 2nd Place (Tim Salimans)

**Competition:** Classify Higgs boson signal vs background from ATLAS detector features. Metric: Approximate Median Significance (AMS).
**Code:** https://github.com/TimSalimans/HiggsML

### Approach
- Ensemble of neural networks + gradient boosted trees
- Direct optimization of AMS metric rather than log-loss
- Physics-informed feature engineering (invariant masses, transverse momenta)

### Key Techniques
1. **Direct AMS optimization**: `AMS = sqrt(2 * ((s+b+b_reg)*ln(1+s/(b+b_reg)) - s))`. Threshold optimized on validation to maximize this metric directly
2. **Physics-informed features**: Invariant mass combinations, delta-R between particles, transverse momentum ratios. Missing values (coded -999) encoded as structural indicators:
   ```python
   m_lep_met = sqrt(2 * pt_lep * MET * (1 - cos(dphi_lep_met)))
   jet_missing = (jet_pt == -999).astype(int)
   ```
3. **Deep dropout networks**: 5 hidden layers, 600 units each, 50% dropout. Large width compensates for dropout variance
4. **Weighted training**: Per-event physics weights in loss function; signal and background have very different weight distributions
5. **Rank-averaging ensemble**: NN + XGBoost + RGF combined via rank-averaging — more robust than probability averaging across differently-calibrated models

### How to Reuse
- When competition metric ≠ standard loss, optimize the metric directly or threshold-optimize on validation
- Missing values with structural meaning should be features, not imputed away
- Domain-informed features consistently beat automated feature engineering in physics/science domains
- Rank-averaging is the safest blend method when models have different probability calibrations

---

## Elo Merchant Category Recommendation — Top Places

**Competition:** Predict customer loyalty score from anonymized merchant transaction history. Regression.
**Writeup:** Multiple top solutions on Kaggle discussion forums.

### Approach
- Feature engineering marathon: hundreds of aggregation features at multiple granularities
- Bimodal target distribution with outlier spike at -33.22 requiring special handling
- LightGBM/XGBoost ensembles

### Key Techniques
1. **Multi-granularity aggregation**: Features at card-level, merchant-level, city-level, cross-level. Time-windowed (1/2/3/6/12 months) aggregations for recency:
   ```python
   card_txn_3m = df[df.purchase_date >= cutoff_3m].groupby('card_id')['amount'].agg(['count','sum','mean','std'])
   card_merchant_nunique = df.groupby('card_id')['merchant_id'].nunique()
   ```
2. **Two-stage outlier handling**: Binary classifier for outlier (-33.22) vs normal → separate regressor per group. The bimodal target breaks single-model regression
3. **Transaction sequence features**: Inter-purchase intervals, spending velocity trends, weekend/weekday ratios, month-over-month growth
4. **Smoothed target encoding**: LOO smoothing prevents leakage: `te = smoothing * group_mean + (1 - smoothing) * global_mean`
5. **LightGBM DART booster**: Drops trees randomly during training (like dropout). `num_leaves=31, lr=0.01, n_estimators=5000` with early stopping

### How to Reuse
- For transaction/event-log data: multi-granularity aggregation features are almost always the top feature family
- Always check target distribution for multimodality; two-stage (classify outliers → conditional regressors) beats monolithic
- Smoothed target encoding safer than raw; use LOO or fold-based encoding
- Automate feature generation pipelines for high-cardinality aggregation tasks

---

## Lyft 3D Object Detection — Top Places

**Competition:** 3D object detection from LiDAR point clouds — predict 3D bounding boxes for cars, pedestrians, cyclists.
**Writeup:** Multiple top solutions on Kaggle discussion forums.

### Approach
- LiDAR point clouds → bird's-eye view (BEV) pseudo-images → 2D detection architectures
- PointPillars / VoxelNet encodings for structured feature maps
- Multi-frame temporal aggregation + camera-LiDAR fusion

### Key Techniques
1. **Bird's-eye view via PointPillars**: Points divided into vertical pillars, per-pillar features via mini-PointNet, scattered onto 2D BEV canvas for standard convolution
2. **Multi-sweep temporal aggregation**: Stack 3-10 consecutive LiDAR sweeps (transformed to current frame). Densifies sparse clouds + provides implicit velocity
3. **Anchor-based detection + oriented NMS**: Predefined anchor boxes at multiple orientations; NMS modified for rotated rectangle IoU
4. **Camera-LiDAR late fusion**: 2D camera detections projected to 3D via calibration, merged with LiDAR detections. Boosts confirmed detections
5. **3D TTA**: Flip point clouds on x/y axes, rotate slightly, ensemble inverse-transformed predictions. Must correctly transform orientation angles

### How to Reuse
- BEV via PointPillars is the go-to LiDAR detection baseline (OpenPCDet, MMDetection3D)
- Multi-sweep aggregation is nearly free performance for sequential LiDAR data
- 3D TTA requires correct inverse-transformation (especially orientation) but consistently adds 1-3% AP
- Camera-LiDAR fusion: diminishing returns over strong LiDAR-only baselines, but worth it for final push

---

## See Click Predict Fix — 1st Place

**Competition:** Predict engagement (views, votes, comments) on civic issue reports (potholes, graffiti, etc.).
**Code:** https://github.com/BlindApe/SeeClickPredictFix

### Approach
- GBM ensemble with extensive feature engineering from text, coordinates, timestamps, categories
- Geographic and temporal features were strongest predictors
- Text features via TF-IDF + SVD

### Key Techniques
1. **Geographic feature engineering**: K-means clustering on lat/lon, distance to city center, historical issue density per cluster
2. **Cyclical temporal encoding**: `hour_sin = sin(2π * hour/24)`, `hour_cos = cos(2π * hour/24)`. Preserves circular nature of time
3. **TF-IDF + SVD text features**: Descriptions → TF-IDF (5000 features, 1-2 ngrams) → truncated SVD (30 components)
4. **Per-target separate models**: Views, votes, comments predicted by independent GBMs sharing features but with own hyperparameters
5. **Log-transform targets**: Engagement counts heavily right-skewed; `log1p` transform before training, `expm1` at prediction

### How to Reuse
- For geo-tagged prediction: k-means on coordinates + historical density are easy wins
- Cyclical encoding (sin/cos) outperforms one-hot for temporal features in tree models
- TF-IDF + SVD is a robust text baseline for short, noisy text
- Always log-transform count/engagement targets — improvement nearly guaranteed for skewed distributions

---

## Allstate Claim Prediction — 2nd Place

**Competition:** Predict insurance claim severity (cost) from anonymized tabular features.
**Writeup:** 2nd place Kaggle blog interview.

### Approach
- Massive stacking ensemble: XGBoost + neural networks + regularized linear models
- Log and Box-Cox target transforms for heavy right skew
- Extensive pairwise categorical interaction features

### Key Techniques
1. **Multi-level stacking**: Level-0 diverse base models → OOF predictions → Level-1 meta-learner (XGBoost or Ridge)
   ```python
   oof_xgb = cross_val_predict(xgb, X, y, cv=5)
   oof_nn = cross_val_predict(nn, X, y, cv=5)
   meta_features = np.column_stack([oof_xgb, oof_nn, oof_ridge])
   meta_model.fit(meta_features, y)
   ```
2. **Box-Cox target transformation**: Better than log for many skewed targets; optimizes lambda parameter:
   ```python
   y_transformed, lambda_opt = boxcox(y + 1)
   ```
3. **Pairwise categorical interactions**: All pairs concatenated → target-encoded. Creates thousands of high-order interactions trees struggle to learn organically
4. **Entity embeddings in neural networks**: Categorical variables → learned dense embeddings → concatenated with continuous features → deep network. Trained embeddings reusable as features in tree models
5. **Optimized blend weights**: scipy.optimize.minimize on MAE with constraint that weights sum to 1

### How to Reuse
- Multi-level stacking is the most reliable final-push technique for tabular competitions
- Box-Cox should be tested alongside log-transform for any right-skewed regression target
- Pairwise categorical interactions + target encoding: powerful for high-cardinality categoricals
- Entity embeddings: best way to handle categoricals with hundreds of levels; extract and reuse in tree models

---

## Cross-Cutting Patterns

| Competition | Task | Key Differentiator | Primary Models | Ensemble? | Feature Eng. |
|-------------|------|-------------------|----------------|-----------|-------------|
| Instant Gratification | Binary (synthetic) | Recognized DGP → QDA | QDA | No | Low |
| YouTube-8M | Multi-label video | NeXtVLAD + MoE + distill | Deep learning | 20+ models | Low |
| Generative Dogs | Image generation | MiFID selection + truncation | BigGAN | No | N/A |
| Cervical Cancer | Medical classification | Two-stage + heavy augment | CNN (transfer) | Multi-arch | Low |
| Right Whale | Fine-grained recognition | Keypoint alignment + few-shot | CNN (transfer) | TTA | Medium |
| Higgs Boson | Binary (physics) | Direct AMS optim + physics feats | NN + GBT | Rank-avg | High |
| Elo Merchant | Regression | Multi-granularity agg + outlier | LightGBM | Blend | Very high |
| Lyft 3D | 3D detection | BEV encoding + multi-sweep | PointPillars | TTA+fusion | Medium |
| See Click Fix | Multi-target regression | Geo + temporal features | GBM | Per-target | High |
| Allstate Claims | Regression | Multi-level stacking + interactions | XGB+NN+linear | Stacking | Very high |
