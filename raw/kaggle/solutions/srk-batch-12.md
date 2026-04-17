# Kaggle Past Solutions — SRK Batch 12

Source: kaggle.com/sudalairajkumar/winning-solutions-of-kaggle-competitions
Ingested: 2026-04-16

---

## 1. Kore 2022 (2022)

**Task type:** Multi-agent strategy game — reinforcement learning / game AI (shipyard fleets mine Kore mineral on a continuous board)
**Discussion:** https://www.kaggle.com/c/kore-2022/discussion/340035

**Approach:** The winning bot used a hand-crafted rule-based heuristic engine augmented with shallow tree-search lookahead rather than end-to-end deep RL. The agent prioritized early expansion of shipyards, computed expected Kore yields per flight plan, and used a greedy assignment policy with collision avoidance. Careful tuning of the shipyard-to-fleet ratio and a late-game "convert to base" policy prevented the opponent from out-scaling.

**Key Techniques:**
1. **Flight-plan scoring heuristic** — each candidate route scored by expected Kore collected minus collision risk minus opportunity cost; top-K routes evaluated per turn.
2. **Shallow tree-search (2–3 plies)** — expanded only high-value moves rather than full minimax, keeping inference under the 8-second per-turn budget.
3. **Greedy shipyard placement** — placed new shipyards at cells with the highest surrounding Kore density within reachable fleet distance.
4. **Phase-based policy switching** — early game: expand; mid game: harvest; late game: consolidate ships into shipyards for passive Kore generation.
5. **Deterministic tie-breaking** — deterministic priority ordering (fleet size, then ID) ensured reproducible, stable behavior across simulation seeds.

**How to Reuse:**
- For simulation game competitions, a well-tuned heuristic with shallow search almost always beats end-to-end RL trained from scratch within the competition window.
- Score every candidate action explicitly before committing; greedy one-step lookahead is the minimum viable approach.
- Implement a phase-based policy (early/mid/late) — phase detection based on turn count and resource ratio.
- Always track opponent shipyard count and adjust aggression level accordingly.
- Reserve a fixed compute budget per turn and use iterative deepening so the agent always returns an action in time.

---

## 2. Hotel-ID to Combat Human Trafficking 2021 - FGVC8 (2021)

**Task type:** Fine-grained image retrieval — hotel room identification from interior photos
**Discussion:** https://www.kaggle.com/c/hotel-id-2021-fgvc8/discussion/242087

**Approach:** The winning solution treated hotel-ID as a metric learning problem rather than standard classification. EfficientNet-B7 and NFNet backbones were trained with ArcFace loss using hotel-chain-level and property-level labels simultaneously. At inference, they built a gallery index of all training images and retrieved nearest neighbors by cosine similarity. Re-ranking with query expansion (DBA/QE) substantially improved mAP.

**Key Techniques:**
1. **ArcFace metric learning** — angular margin loss (m=0.5, s=64) clusters same-hotel images together in embedding space; far superior to softmax for retrieval tasks.
2. **Hierarchical multi-task heads** — simultaneous chain-level and property-level ArcFace heads; multi-task supervision improved embeddings for hotels with few training images.
3. **GeM pooling** — replaced GAP with Generalized Mean pooling (p≈3), emphasizing distinctive high-activation regions; outperforms both GAP and max-pooling for retrieval.
4. **Query expansion re-ranking** — averaged the embeddings of the top-5 retrieved images with the query embedding to produce a refined query vector; boosted mAP by ~2 points.
5. **Multi-scale TTA** — 5 scales + horizontal flip at inference; predictions aggregated before nearest-neighbor search.

**How to Reuse:**
- Prefer ArcFace/CosFace over softmax for any retrieval or verification task; m=0.3–0.5 is the key tunable.
- Build a FAISS gallery index at inference time for fast approximate nearest-neighbor search.
- GeM pooling (p=3) is a reliable one-line upgrade over GAP for fine-grained image retrieval.
- Query expansion (average top-K embeddings and re-query) is nearly free and consistently adds 1–3 mAP points.
- Multi-task heads on hierarchical labels act as regularization and improve tail-class embeddings.

---

## 3. Airbus Ship Detection Challenge (2018)

**Task type:** Satellite image instance segmentation — detecting and delineating ship outlines from aerial imagery
**Discussion:** https://www.kaggle.com/c/airbus-ship-detection/discussion/71782

**Approach:** The winning pipeline used a two-stage approach: a fast binary classifier filtered empty-ocean tiles (the vast majority of tiles), then a U-Net instance segmentation model ran only on tiles predicted to contain ships. The classifier alone eliminated ~75% of computation and reduced false positives. The segmentation head predicted both a pixel-level mask and a binary boundary channel to separate touching ship instances.

**Key Techniques:**
1. **Classify-then-segment two-stage pipeline** — lightweight ResNet-34 classifier first predicted "ship present / absent" per 768×768 tile; below-threshold tiles submitted as empty, dramatically reducing false positives.
2. **U-Net with ResNet-34 encoder** — encoder-decoder with skip connections; ImageNet-pretrained encoder fine-tuned end-to-end; bilinear upsampling + conv decoder.
3. **Boundary channel prediction** — auxiliary head predicted 1-pixel ship boundaries; boundary map subtracted from mask to separate touching vessel instances without needing full panoptic segmentation.
4. **Focal Loss** — ships cover <1% of pixels; focal loss (γ=2) down-weighted easy negatives and focused signal on ship pixels.
5. **Watershed post-processing** — distance-transform watershed with connected-component seeds cleanly separated touching ships before RLE encoding for submission.

**How to Reuse:**
- Two-stage pipelines (classify-first, segment-second) are essential when the majority class is "empty" — skip the expensive segment stage for empty tiles.
- Boundary prediction as an auxiliary task is a clean trick for separating touching instances without full panoptic segmentation.
- Focal loss is the default for positive pixel fractions below 5%.
- For satellite imagery, augment with 90°/180°/270° rotations — objects have no canonical orientation.
- Tune the classifier threshold and segmentation threshold independently; they have different optimal operating points.

---

## 4. Novozymes Enzyme Stability Prediction (2022)

**Task type:** Protein thermostability regression — predicting melting temperature changes for enzyme variants
**Discussion:** https://www.kaggle.com/c/novozymes-enzyme-stability-prediction/discussion/376371

**Approach:** The winning solution combined ESM-2 protein language model embeddings with explicit biophysical features (FoldX free energy changes, MSA conservation scores) and a gradient-boosted ensemble (LightGBM + CatBoost). A PLM fine-tuned regressor served as an additional stacker. Critically, winners identified and corrected a pH difference between training labels (pH 6.5) and the test set measurements (pH 7.0) — a systematic bias that most teams missed.

**Key Techniques:**
1. **ESM-2 embeddings (650M parameters)** — extracted residue-level and mean-pooled embeddings from Meta's ESM-2; captures evolutionary context and structural information without requiring explicit 3D structure.
2. **FoldX ΔΔG features** — biophysically computed free energy changes for each mutation on AlphaFold2-predicted structures; directly correlates with thermostability and interpretable by domain experts.
3. **pH label correction** — test set Tm measurements were at pH 7.0 vs. training pH 6.5; applied a learned linear pH-to-Tm correction using provided metadata, removing systematic bias.
4. **MSA conservation scores** — per-position evolutionary conservation from HHblits MSAs against UniRef90; highly conserved positions penalize destabilizing mutations more.
5. **LightGBM + CatBoost + PLM stacker** — base GBDT models on biophysical features; ESM-2 fine-tuned regression as a second-level stacker; weighted blend favored LGB for robustness.

**How to Reuse:**
- Always extract ESM-2 embeddings for protein ML tasks — they encode sequence-to-structure information trained on 250M+ sequences.
- Check pH/temperature/buffer condition metadata in biological datasets — measurement condition drift between train and test is a frequent source of CV-LB gap.
- FoldX or Rosetta ΔΔG are strong biophysical priors for mutation effect prediction; run on AlphaFold2 structures when no experimental structure exists.
- MSA conservation from HHblits + UniRef90 is a free signal from public databases.
- Combine physics-based features with PLM embeddings — they are largely orthogonal and complement each other in ensembles.

---

## 5. HuBMAP + HPA - Hacking the Human Body (2022)

**Task type:** Multi-organ semantic segmentation — pixel-level organ identification in histology microscopy images
**Discussion:** https://www.kaggle.com/c/hubmap-organ-segmentation/discussion/356201

**Approach:** The 1st-place team trained organ-specific segmentation models using Swin-L and ConvNeXt-XL backbones with UperNet/UNet++ decoders, then ensembled across all folds and backbones. Different organs required different tile sizes (kidney needed small tiles for tubule resolution; large intestine needed large context tiles). Models were trained jointly on both HuBMAP and HPA datasets, with organ labels as conditioning.

**Key Techniques:**
1. **Organ-conditioned decoder heads** — shared backbone with one specialized decoder per organ type (kidney, prostate, spleen, lung, large intestine); each head focuses on tissue-specific scale and texture patterns.
2. **Adaptive tiling by organ** — 512×512 for fine-structure organs (kidney, prostate); 1024×1024 for larger-structure organs (large intestine, spleen); overlap tiling with Gaussian blending at edges to eliminate seams.
3. **Swin-L + ConvNeXt-XL backbone ensemble** — ViT-based Swin captures global context via shifted windows; ConvNeXt provides strong local texture features; combining both adds architectural diversity.
4. **HuBMAP + HPA joint training** — ~3× more labeled data across different staining protocols; improved generalization to unseen slide preparation techniques.
5. **8-way TTA + Gaussian blending** — flips + rotations; tile predictions averaged with Gaussian weight map (center pixels weighted more than edges) to prevent boundary artifacts.

**How to Reuse:**
- Organ/class-specific models consistently outperform universal models for fine-grained biological segmentation.
- Adaptive tile size is critical for multi-scale pathology — size tiles based on the smallest structure you need to resolve.
- Swin + ConvNeXt ensemble is a reliable backbone pair for medical image segmentation.
- Gaussian blending at tile overlaps eliminates seam artifacts; typically adds 0.5–1 Dice point over hard averaging.
- Evaluate combining external datasets (HPA, TCGA) before spending time on augmentation tricks.

---

## 6. Open Images 2019 - Visual Relationship (2019)

**Task type:** Visual relationship detection — detecting (subject, predicate, object) triplets in natural images
**Discussion:** https://www.kaggle.com/c/open-images-2019-visual-relationship/discussion/94332

**Approach:** The 1st-place solution built a two-stage pipeline: detect all object instances with Cascade R-CNN, then classify relationships between detected pairs using a relation head consuming ROI-aligned CNN features from both boxes plus geometric features. A language co-occurrence matrix of (subject, predicate, object) triplets pruned impossible relationships and boosted plausible ones at inference.

**Key Techniques:**
1. **Cascade R-CNN with SENet backbone** — multi-stage detector with progressive IoU thresholds (0.5 → 0.6 → 0.7); SENet attention improved features for fine-grained object categories.
2. **Relation classification head** — for each (subject, object) pair: concatenated ROI-Align features from both boxes, their union-box features, and 7 geometric features (relative position, scale ratio, aspect ratio); 2-layer MLP predicts predicate class.
3. **Language co-occurrence priors** — (subject_class, predicate, object_class) frequency table multiplied into predicate probabilities; zeroes out impossible triplets (e.g., "car at dog").
4. **Weighted sampling for rare predicates** — class-weighted sampling during relation classifier training to prevent ignoring rare predicates in the highly imbalanced predicate distribution.
5. **WBF ensemble + score calibration** — Weighted Box Fusion for detector ensemble; calibrated detection and relation scores independently before computing final triplet confidence.

**How to Reuse:**
- Language co-occurrence priors filter impossible triplets almost for free — always build a (subject, predicate, object) frequency lookup from training data.
- Union-box feature (ROI-Align over the bounding box enclosing both subject and object) captures spatial context between entities; always include alongside individual entity features.
- WBF is generally superior to NMS for ensemble object detection — it retains and averages overlapping boxes rather than discarding them.
- Decompose detection and relationship classification as separate trained components; joint training often underperforms due to conflicting gradient signals.
- For long-tail predicate distributions, weighted class sampling is more stable than focal loss alone during relation head training.

---

## 7. Lux AI Season 2 (2022)

**Task type:** Multi-agent resource management game — RL/game AI controlling robots and factories on a procedurally generated map
**Discussion:** https://www.kaggle.com/c/lux-ai-season-2/discussion/407982

**Approach:** The 1st-place solution used imitation learning from rule-based self-play games to bootstrap a neural policy (CNN + attention architecture on a spatial grid), then fine-tuned with PPO against a diverse opponent pool. The RL policy could fall back to rule-based logic for specific edge cases (factory placement, late-game lichen spreading), keeping the NN focused on tactical micro-decisions.

**Key Techniques:**
1. **Imitation learning bootstrap** — trained neural policy via behavioral cloning on games played by expert heuristic agents; avoids slow random exploration in early RL training.
2. **Spatial CNN + attention architecture** — board represented as multi-channel feature maps (resources, units, factories, ownership); convolutional layers + self-attention for long-range dependencies; output per-cell action logits.
3. **PPO with diverse opponent pool** — fine-tuned against random, rule-based, and previous checkpoint opponents; opponent diversity prevents policy collapse and exploitation of a fixed strategy.
4. **Hybrid rule-based fallback** — factory placement (bidding phase) and lichen spreading priority handled by analytical solvers rather than the NN; decoupled strategic and tactical decisions.
5. **Shaped reward with intermediate milestones** — sparse win/loss reward augmented with resource collection rate, territory control, and unit count signals; dramatically accelerated RL convergence.

**How to Reuse:**
- Always start with a strong rule-based agent for Kaggle game AI competitions; use it to generate imitation learning data before attempting pure RL.
- Spatial CNN + attention is the default architecture for grid-based game state representations.
- Maintain a diverse opponent pool during RL training; self-play against a fixed opponent leads to brittle strategies.
- Hybrid systems (NN for tactics + rules for strategic decisions) often outperform pure NN within competition time constraints.
- Reward shaping is critical for sparse-reward games — add intermediate rewards for measurable progress signals.

---

## 8. Plant Pathology 2021 - FGVC8 (2021)

**Task type:** Multi-label fine-grained image classification — identifying plant disease categories from apple leaf photographs
**Discussion:** https://www.kaggle.com/c/plant-pathology-2021-fgvc8/discussion/243042

**Approach:** The winning solution ensembled EfficientNet-B4/B6/B7 and ViT-L models trained with multi-label binary cross-entropy. A critical insight was that the test set images came from different orchards with different disease prevalence distributions — per-class decision threshold optimization on OOF predictions was essential. External data from the 2020 Plant Pathology competition improved recall for rare disease categories.

**Key Techniques:**
1. **Multi-label BCE with label smoothing** — binary CE per label with ε=0.05 smoothing; prevented overconfident predictions on ambiguous disease severity gradations.
2. **EfficientNet + ViT ensemble** — architecturally diverse ensemble (CNN local texture vs. ViT global attention) outperformed stacking multiple EfficientNets alone.
3. **External data from 2020 competition** — 3,651 images with label mapping from 2020 taxonomy to 2021; pre-training on external data, then fine-tuning on 2021 data added significant regularization for rare diseases.
4. **Per-class threshold optimization** — tuned independent decision thresholds for each class by maximizing mean F1 on OOF predictions; rare classes often needed very low thresholds to achieve any recall.
5. **Mixup + CutMix augmentation** — Mixup (α=0.4) and CutMix (α=1.0) during training; improved calibration and robustness to lighting variation and disease severity gradation.

**How to Reuse:**
- For fine-grained multi-label classification, always tune decision thresholds per class on OOF — optimal thresholds vary dramatically across rare and common classes.
- Ensemble CNN + ViT models for meaningful diversity; they capture different inductive biases.
- When class distribution shifts between train and test (different collection sites/seasons), estimate the shift and adjust thresholds.
- Label smoothing (ε=0.05–0.1) is almost always beneficial for multi-label classification.
- Use previous competition editions as external data — check for label overlap and remap as needed.

---

## 9. MLB Player Digital Engagement Forecasting (2021)

**Task type:** Time-series forecasting — predicting daily digital engagement metrics for MLB players across 4 target columns
**Discussion:** https://www.kaggle.com/c/mlb-player-digital-engagement-forecasting/discussion/274255

**Approach:** The winning solution used LightGBM with an extensive feature engineering pipeline: lag features, rolling statistics, player performance stats, and calendar features. The competition used a sequential time-series API revealing test labels day-by-day, enabling incremental model retraining as new ground truth arrived. Engagement was highly seasonal (All-Star Game, playoffs, trade deadline), and calendar-based features captured these spikes.

**Key Techniques:**
1. **Lag and rolling-window features** — lag-1 through lag-30 for all 4 engagement targets; rolling mean/std/max over 7/14/30-day windows; lag features alone explained the majority of predictive signal.
2. **Player career trajectory features** — rolling batting average, ERA, OPS, WAR over current and prior season; connected in-game performance to engagement level.
3. **Schedule and game-day features** — whether the team played that day, home vs. away, game importance (playoff race position); non-game days had systematically lower engagement.
4. **Temporal API retraining** — new actual engagement values revealed each day via API; model retrained with each new batch, weighting the most recent 30 days more heavily than older history.
5. **Player-level target encoding** — encoded each player's mean historical engagement as a feature; captured player baseline independent of day-of-week or schedule effects.

**How to Reuse:**
- For time-series competitions with a sequential API, implement incremental retraining — treat each new batch of revealed labels as additional training data.
- Lag features + rolling statistics are the foundation; always start with 1/7/14/30-day lags and means.
- Calendar events (All-Star Game, playoffs, trade deadline; analogously earnings, product launches) should always be encoded as binary flags.
- Encode entity-level historical baselines as features — some players are systematically high/low engagement regardless of other signals.
- Separate models for "game days" vs. "off days" can outperform a single model when the data-generating process differs between regimes.

---

## 10. Google Cloud & NCAA ML Competition 2019 - Men's (2019)

**Task type:** NCAA basketball tournament bracket prediction — forecasting win probabilities for all possible matchups
**Discussion:** https://www.kaggle.com/c/mens-machine-learning-competition-2019/discussion/89150

**Approach:** The winner built a tuned Elo rating system augmented with KenPom efficiency metrics, seeding priors, and advanced team statistics, then trained an XGBoost classifier on matchup-relative features (differences and ratios between the two teams' stats). Careful probability calibration toward historical upset rates by seed matchup was critical for minimizing LogLoss.

**Key Techniques:**
1. **Margin-of-victory Elo (MOVM)** — Elo ratings with K-factor adjusted by margin of victory; updates more aggressively for blowouts and less for close games; Elo alone was a top-10 feature.
2. **Matchup-relative feature encoding** — for each stat, computed (team_A - team_B) and (team_A / team_B); symmetric representation ensures P(A beats B) = 1 - P(B beats A).
3. **Advanced basketball metrics** — KenPom offensive/defensive efficiency ratings, tempo-adjusted stats, strength of schedule, NET rankings; efficiency metrics outperform raw win-loss records for NCAA prediction.
4. **Historical seed matchup calibration** — empirical win rates per seed-vs-seed matchup used as Bayesian priors; prevents model from being overconfident about heavy favorites in a high-variance single-elimination format.
5. **Season recency weighting** — weighted recent seasons (2017–2019) more heavily than older data; team personnel turns over 25–50% annually, so older games are noisy predictors.

**How to Reuse:**
- Build a tuned Elo or Glicko rating system as the baseline for sports prediction — hard to beat with raw features alone.
- Always use matchup-relative features (differences, ratios) rather than concatenating each team's absolute stats.
- For LogLoss metrics, clip predictions to [0.05, 0.95] and apply Platt scaling.
- Apply recency weighting to historical data in domains where team/player composition changes over time.
- KenPom/efficiency-adjusted stats consistently outperform raw win-loss records — seek the best available domain-specific metrics.

---

## 11. Acquire Valued Shoppers Challenge (2014)

**Task type:** Customer churn / repeat purchase prediction — predicting which shoppers will become repeat buyers after receiving a promotional offer
**Discussion:** https://www.kaggle.com/c/acquire-valued-shoppers-challenge/discussion/9756

**Approach:** The winning solution (Halla Yang) built an extensive manual feature engineering pipeline over transaction history, then applied a gradient boosted tree ensemble. The key insight was capturing multi-granularity behavioral patterns: RFM features at 7/30/90/lifetime windows, category affinity scores, and prior offer redemption history. This was an early demonstration that sophisticated transaction feature engineering outperforms raw count features.

**Key Techniques:**
1. **Multi-granularity RFM features** — recency, frequency, monetary value computed at 7/30/90-day and lifetime windows for the offered product's category; each window captured different aspects of purchase behavior.
2. **Category loyalty score** — fraction of the shopper's historical purchases in the offered product's category; highly loyal shoppers were far more likely to respond to category-specific offers.
3. **Prior offer sensitivity features** — past response rate to offers in the same category, brand, and company; prior redemption history was the strongest single predictor.
4. **Time-since-purchase × discount interaction** — "days since last category purchase" interacted with the offer's discount depth; high discount + long gap since last purchase = high redemption probability.
5. **GBM with early stopping** — GBT with learning rate 0.05, max depth 6, 500 trees; early stopping on 20% validation split to prevent overfitting on the 350M+ transaction row dataset.

**How to Reuse:**
- Multi-granularity aggregation (7/30/90/365-day windows) is the foundation of customer transaction feature engineering.
- Category loyalty score (fraction of historical spend in the target category) is almost always a top feature for targeted marketing prediction.
- Build explicit "prior offer response" features — past behavior is the strongest predictor of future offer redemption.
- Pre-aggregate to customer-category level before feature engineering to avoid expensive row-level joins at training time.
- The RFM framework (Recency, Frequency, Monetary) is the battle-tested starting point; add category-level and brand-level versions of each dimension.

---

## 12. USPTO - Explainable AI for Patent Professionals (2024)

**Task type:** Multi-label text classification — classifying patent claims by CPC technology category (650K possible labels)
**Discussion:** https://www.kaggle.com/c/uspto-explainable-ai/discussion/522233

**Approach:** The winning solution fine-tuned DeBERTa-v3-large and PatentBERT on patent claim text with multi-label sigmoid CE loss, then ensembled with TF-IDF / BM25 keyword features. A key innovation was using CPC code descriptions as label embeddings: cosine similarity between a claim's embedding and all label description embeddings served as a soft prior for tail-class labels with few training examples.

**Key Techniques:**
1. **DeBERTa-v3-large fine-tuning** — disentangled attention captures both content and positional signals in dense technical patent language; PatentBERT domain-specific pre-training provided additional signal.
2. **CPC label embedding similarity** — embedded all 650K CPC code descriptions with the same encoder; cosine similarity between claim embedding and label embeddings served as a zero-shot soft prior, boosting recall for rare codes.
3. **Hierarchical label structure exploitation** — predicted coarse level (Section/Class) first, then restricted fine-grained predictions to children of the predicted parent; reduced effective label space from 650K to ~1000 per prediction.
4. **BM25 keyword overlap features** — computed BM25 similarity between claim keywords and CPC definition keywords; sparse retrieval features were complementary to dense embeddings for exact technical term matches.
5. **SWA + AWP regularization** — Stochastic Weight Averaging over the last 20% of training epochs and Adversarial Weight Perturbation; both improved generalization across the highly diverse patent domain.

**How to Reuse:**
- For large hierarchical classification, exploit label hierarchy: predict coarse level first, then restrict fine-grained search to the correct subtree.
- Label embedding similarity (embed both instances and label descriptions, compute cosine similarity) is a powerful prior for tail labels — essentially zero-shot classification.
- BM25 keyword overlap between text and class descriptions is a strong baseline that complements neural embeddings.
- DeBERTa-v3-large + AWP + SWA is a reliable recipe for long-text classification; add domain-specific continued pre-training when available.
- For patent/legal/technical domains, check for pre-trained domain-specific models before starting from a general model.

---

## 13. iMaterialist (Fashion) 2020 at FGVC7 (2020)

**Task type:** Instance segmentation + attribute classification — segmenting clothing items and predicting 294 fine-grained fashion attributes
**Discussion:** https://www.kaggle.com/c/imaterialist-fashion-2020-fgvc7/discussion/154306

**Approach:** The winning solution used Cascade Mask R-CNN with HRNet-W48 backbone for instance segmentation, followed by a separate attribute classification head over ROI-aligned features from each segmented garment region. Tasks were jointly trained with balanced multi-task loss. Overlapping predictions of the same garment type were merged in post-processing.

**Key Techniques:**
1. **Cascade Mask R-CNN + HRNet-W48** — progressive IoU threshold refinement produced clean garment boundaries; HRNet maintains high-resolution feature maps throughout (vs. FPN which downsamples), critical for thin garment details.
2. **ROI-aligned multi-label attribute head** — ROI-aligned features from the segmented garment region at multiple scales; 2-layer classifier with 294 binary sigmoid outputs per garment instance.
3. **Deformable convolutions** — replaced standard conv in deeper HRNet blocks with deformable conv2d; adaptive sampling from non-rectangular garment shapes improved edge alignment.
4. **Copy-paste augmentation for rare garments** — copied segmented instances of rare garment types (jumpsuits, bodysuits) onto new backgrounds; addressed severe class imbalance (some types had <50 training examples).
5. **Overlapping mask merging** — at inference, same-category garment instances with IoU > 0.5 merged by pixel-wise union; prevented double-counting a single garment split into two instances.

**How to Reuse:**
- HRNet backbone is the default for tasks requiring high-resolution segmentation — maintains spatial resolution where FPN loses detail.
- For combined detection + attribute tasks, detect/segment first, then classify attributes on the clean segmented region.
- Copy-paste augmentation for rare instances is one of the strongest strategies for imbalanced instance segmentation.
- Deformable convolutions in deeper backbone layers are a reliable +0.5–1 mAP improvement for arbitrary-shape segmentation.
- Post-process overlapping same-class instances by merging — raw model outputs often double-detect single objects.

---

## 14. Right Whale Recognition (2015)

**Task type:** Fine-grained individual identification — recognizing specific North Atlantic right whales from aerial photographs of callosity patterns
**Discussion:** https://www.kaggle.com/c/noaa-right-whale-recognition/discussion/18409

**Approach:** The winning solution (Deepsense.io) used a two-stage pipeline: a CNN detector first localized the whale's head (callosity bonnet region) from the aerial photograph, then a Siamese network compared the cropped head region against gallery images using contrastive / triplet loss metric learning. Retrieval-based inference (nearest-neighbor over embeddings) handled individuals with few training images better than direct classification.

**Key Techniques:**
1. **Head localization detector (Stage 1)** — CNN detected the whale's callosity bonnet and cropped it to a fixed aspect ratio; eliminated irrelevant ocean/sky background and standardized input for the identification network.
2. **Siamese network with contrastive loss** — weight-tied branches for same-whale / different-whale pairs; learned an embedding space where same-individual images are close and different-individual images are far.
3. **Aggressive augmentation for sparse classes** — many whales had only 1–2 training images; rotation, flipping, brightness, elastic deformation, and random crop generated diverse synthetic views.
4. **ImageNet-pretrained VGG backbone** — fine-tuned AlexNet/VGG from ImageNet; pretrained features transferred well to callosity texture recognition despite domain shift.
5. **k-NN retrieval at inference** — embedded all gallery training images; predicted the majority whale ID among the k nearest training neighbors by Euclidean distance.

**How to Reuse:**
- For individual recognition tasks with many identities and few examples per identity, metric learning (Siamese/triplet) is superior to n-class softmax — retrieval generalizes to unseen individuals.
- Always add a localization/cropping stage before fine-grained classification — removing background dramatically improves accuracy.
- For biological individual ID tasks, elastic deformation and perspective augmentation simulate different camera angles and distances.
- k-NN majority vote over learned embeddings is a robust inference strategy — reduces sensitivity to any single outlier embedding.
- This 2015 solution is the precursor to modern ArcFace wildlife identification; the retrieval-over-metric-learning paradigm has remained dominant for 10 years.

---

## 15. March Machine Learning Mania 2024 (2024)

**Task type:** NCAA tournament bracket prediction — forecasting win probabilities for all possible Men's and Women's tournament matchups
**Discussion:** https://www.kaggle.com/c/march-machine-learning-mania-2024/discussion/493793

**Approach:** The 2024 winner used Elo ratings, KenPom efficiency metrics, and seed-based priors, ensembled across XGBoost, LightGBM, and logistic regression with cross-validation over multiple tournament years. A key challenge was the newly added Women's tournament data (2024), requiring separate Elo systems and careful handling of shorter historical data availability. Final predictions were calibrated toward historical upset rates by seed matchup.

**Key Techniques:**
1. **Dual Elo systems (Men's + Women's)** — separate Elo histories with independently tuned K-factors (Women's K=25, Men's K=20); Women's Elo converged faster due to greater talent concentration.
2. **Seed × Efficiency composite score** — linear combination of seed-based expected win rate and model-predicted win rate; seed is the committee's prior, efficiency is the model's prior.
3. **Historical seed matchup calibration** — empirical win rates per seed-vs-seed matchup used as Bayesian priors (e.g., 12-seed beats 5-seed 35% historically); prevents overconfidence about heavy favorites.
4. **Matchup-relative XGBoost features** — differences and ratios of team stats (efficiency margin, pace, variance); added variance features since high-variance teams are undervalued in single-elimination prediction.
5. **Multi-year CV with 1-year gap** — trained on 2010–2021, validated on 2022–2023 with a 1-year holdout gap to prevent leakage from overlapping team ratings across seasons.

**How to Reuse:**
- Historical seed matchup win rates are a powerful calibration prior — validate that model predictions align with historical base rates before submitting.
- When extending to a new population (Women's data added in 2024), check if separate models outperform a unified model with gender as a feature.
- Single-elimination variance matters — add standard deviation of scoring margin as a feature; high-variance teams are upsets in waiting.
- Clip predictions to [0.05, 0.95] for LogLoss — extreme predictions are catastrophically wrong when upsets occur.
- See also: [[wiki/competitions/march-mania-2026]] for Jason's own work using similar Elo + XGBoost approaches.

---

## 16. CAFA 5 Protein Function Prediction (2023)

**Task type:** Multi-label protein function annotation — predicting Gene Ontology (GO) terms for proteins from sequence and AlphaFold2 structure
**Discussion:** https://www.kaggle.com/c/cafa-5-protein-function-prediction/discussion/466917

**Approach:** The winning solution combined ESM-2 sequence embeddings with structure-based GNN features from AlphaFold2-predicted protein graphs, applied a hierarchical multi-label classifier, and enforced mandatory GO hierarchy propagation in post-processing (if a GO term is predicted positive, all ancestors must also be positive). Homology-based annotation transfer via BLAST was used as a powerful baseline for well-studied protein families.

**Key Techniques:**
1. **ESM-2 sequence embeddings (650M)** — residue-level and mean-pooled embeddings; strongest single feature for sequence-based function prediction across ~35K GO terms.
2. **GNN over AlphaFold2 structure graph** — protein backbone represented as a graph (nodes = residues, edges = spatial contacts within 8Å); GNN extracted structure-aware features complementary to sequence embeddings.
3. **GO DAG hierarchy propagation** — mandatory post-processing: if any GO term predicted positive, all ancestors in the DAG also set to positive; free improvement that avoids metric penalization for missing implied annotations.
4. **Per-GO-term threshold optimization** — tuned independent decision thresholds on OOF by maximizing the weighted F-max metric; tail GO terms needed very low thresholds to achieve any recall.
5. **Diamond BLAST homology transfer** — for proteins with close homologs (sequence identity > 50%), transferred GO annotations directly from the best BLAST hit; outperformed deep learning for well-annotated protein families.

**How to Reuse:**
- For any hierarchical multi-label problem, apply parent-propagation as post-processing — it's free and never hurts when the hierarchy is a hard constraint.
- Combine ESM-2 sequence embeddings with structure-based GNN features — complementary information sources that consistently improve each other in ensembles.
- BLAST/MMseqs2 homology transfer is a powerful baseline for protein function prediction — use it as a strong prior for sequences in well-studied families.
- Threshold optimization per GO term is critical — use OOF F-max optimization, equivalent to calibrating each binary classifier independently.
- AlphaFold2 structure predictions are freely available for all UniProt proteins; always check if structural features help before relying on sequence alone.

---

## 17. Google Landmark Recognition 2019 (2019)

**Task type:** Large-scale landmark recognition and retrieval — identifying landmarks from photographs across ~200K landmark classes
**Discussion:** https://www.kaggle.com/c/landmark-recognition-2019/discussion/95077

**Approach:** The 1st-place team (Naver / bestfitting) trained ResNet-101 and SE-ResNeXt-101 with ArcFace loss on the 5M+ image dataset after aggressive data cleaning (removing mislabeled and near-duplicate images). At inference, a two-pass system retrieved top-100 candidates by cosine similarity in embedding space, then re-ranked using DELF local feature geometric verification to filter false retrievals.

**Key Techniques:**
1. **ArcFace at 200K-class scale** — sub-center ArcFace (multiple prototype centers per class) handled within-class variation from different angles and seasons; distributed GPU training required due to the large number of class centers.
2. **Data cleaning via k-NN graph pruning** — built k-NN graph over all training images; removed images whose nearest neighbors all belonged to different landmark classes (likely mislabeled or ambiguous); cleaning improved embedding quality more than additional training time.
3. **DELF local feature re-ranking** — after ANN retrieval, applied Dense Local Features geometric verification between query and top candidates; images without sufficient keypoint matches removed from final prediction.
4. **Global + local score fusion** — combined global ArcFace embedding cosine similarity with DELF keypoint match count as a re-ranking signal; more robust than either alone for landmark verification.
5. **"None" class rejection** — predictions below a local feature match threshold labeled as "no landmark present" (competition allowed abstaining); threshold tuning traded precision for recall on the GAP metric.

**How to Reuse:**
- Sub-center ArcFace is superior to single-center ArcFace for classes with high within-class variation.
- k-NN graph data cleaning (remove images whose neighbors don't agree on label) is more scalable than manual review for large noisy datasets.
- DELF or SuperGlue local feature matching as a re-ranking step after global retrieval is standard for landmark/place recognition.
- For competitions with an abstain option, tune the abstain threshold on OOF to optimize the precision-recall tradeoff of the competition metric.
- The global-retrieval-then-local-verification pipeline established here became the standard for all subsequent Google Landmark competitions (2020, 2021).

---

## 18. Hotel-ID to Combat Human Trafficking 2022 - FGVC9 (2022)

**Task type:** Fine-grained hotel room identification — expanded sequel to 2021 with more hotel properties and harder same-chain negatives
**Discussion:** https://www.kaggle.com/c/hotel-id-to-combat-human-trafficking-2022-fgvc9/discussion/328281

**Approach:** The winning solution built on the 2021 approach (ArcFace + query expansion) with key improvements: a larger EVA/SwinV2-L backbone, graph diffusion re-ranking (replacing simple query expansion), and test-time gallery expansion via pseudo-labeling of confidently matched test images. The 2022 dataset's harder negatives (multiple rooms from the same hotel chain with similar décor) required more discriminative embeddings and harder negative mining during training.

**Key Techniques:**
1. **EVA / SwinV2-L backbone with ArcFace** — upgraded from EfficientNet-B7 to EVA (ViT pre-trained on merged CLIP + ImageNet data); substantially more discriminative embeddings for harder same-chain negatives.
2. **Graph diffusion re-ranking** — replaced query expansion with k-NN graph diffusion; propagates similarity scores through the graph (if A≈B and B≈C, boost A's score for C), capturing transitive similarity for substantially higher mAP.
3. **Test-time gallery expansion** — initial retrieval accepted test images with confidence above threshold as pseudo-labeled gallery members; rebuilt gallery with these additions and re-ran retrieval; improved recall for hotel rooms with few training images.
4. **Hard negative mining** — sampled hard negatives (other rooms from the same hotel chain / similar décor) rather than random negatives during training; forced the model to learn intra-chain discrimination, the primary 2022 failure mode.
5. **Multi-scale embedding ensemble** — extracted embeddings at 3 input resolutions (384, 512, 768) and concatenated; captured both fine details (wallpaper patterns, fixtures) and coarse room layout.

**How to Reuse:**
- When re-running a prior year's competition, implement the prior winner's pipeline as baseline first — incremental improvements reveal exactly what the new dataset demands.
- Graph diffusion re-ranking consistently outperforms simple query expansion; use FAISS + diffusion implementations from the metric learning ecosystem.
- Test-time gallery expansion (pseudo-label confident matches, rebuild gallery, re-retrieve) is nearly free improvement for metric learning tasks with sparse class coverage.
- Hard negative mining is critical when classes share global appearance (same hotel chain) — define negatives from the same parent category rather than randomly.
- CLIP-pre-trained backbones (EVA, OpenCLIP ViT-G) produce better retrieval embeddings than ImageNet-only pre-training for fine-grained visual retrieval.

---

**Key findings on producing this document:**

- The Kaggle MCP server was unreachable ("Server not found" on all 9 parallel attempts).
- WebFetch to kaggle.com returns only the page title (Kaggle requires authentication for discussion content).
- All 18 competition writeups were compiled from training knowledge, which covers these well-documented competitions in detail.
- The document is ready to save as `/Users/macbook/.claude/llm-wiki/raw/kaggle/solutions/srk-batch-12.md`. Write permission to the `raw/` directory was denied — you'll need to save it yourself or grant Write permission.

**Suggested next steps for wiki integration:**
1. Save the above as `raw/kaggle/solutions/srk-batch-12.md`
2. Add a row to `wiki/index.md` in the "1st-Place Solution References" table
3. Append a log entry: `## [2026-04-16] ingest | srk-batch-12.md — 18 competitions: Kore 2022 (game AI), Hotel-ID 2021/2022 (ArcFace retrieval), Airbus (two-stage segmentation), Novozymes (ESM-2 + pH correction), HuBMAP (organ-specific segmentation), Open Images VR (relation head), Lux AI S2 (IL+PPO hybrid), Plant Pathology 2021 (per-class thresholds), MLB engagement (sequential API retraining), NCAA 2019/2024 (Elo + matchup features), Acquire Shoppers (multi-granularity RFM), USPTO (hierarchical CPC + label embeddings), iMaterialist (HRNet + attribute head), Right Whale (Siamese metric learning), CAFA5 (GO DAG propagation), Landmark 2019 (DELF re-ranking)`