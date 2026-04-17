# Kaggle Past Solutions — SRK Batch 9

Source: kaggle.com/sudalairajkumar/winning-solutions-of-kaggle-competitions
Ingested: 2026-04-16

---

## 1. The Learning Agency Lab - PII Data Detection (2024)

**Task type:** Token-level named entity recognition (NER) to detect and remove personally identifiable information from student essays.
**Discussion:** https://www.kaggle.com/c/pii-detection-removal-from-educational-data/discussion/497374

**Approach:** The winning team ("fold-zero") built an ensemble of diverse DeBERTa architectures combined with custom postprocessing rules. Multiple DeBERTa variants (base, large, xlarge, v3 variants) were trained on the provided data augmented with synthetic PII examples, then combined via soft-voting ensemble. Postprocessing heuristics corrected edge cases — phone numbers, emails, and URLs missed by the neural models — and per-entity confidence thresholds were tuned separately to balance precision-recall on the micro-F5 evaluation metric.

**Key Techniques:**
1. **DeBERTa ensemble diversity** — Combined DeBERTa-v3-base, DeBERTa-v3-large, and DeBERTa-large with different tokenization strides (128, 256, 512); soft-voting across models smoothed per-token probability estimates.
2. **Stride-based sliding window inference** — Long essays exceeded model context limits; overlapping windows with stride-based merging aggregated predictions without truncation artifacts.
3. **Synthetic data augmentation** — Additional PII examples (names, addresses, phone patterns) were synthetically generated and injected into training to improve recall on rare PII types.
4. **Rule-based postprocessing** — Regex patterns for emails, URLs, and phone numbers caught structured PII the model missed; false-positive suppression was applied to common English words misclassified as names.
5. **Per-entity threshold calibration** — Separate confidence thresholds were tuned for each BIO label (NAME, EMAIL, PHONE, etc.) to balance precision-recall independently per entity class.

**How to Reuse:**
- Use DeBERTa-v3-large as the default backbone for document-level NER on English text; it consistently tops leaderboards.
- Apply stride-based windowed inference for texts longer than 512 tokens — never truncate.
- Always complement neural NER with regex rules for structured entities (emails, phone numbers, SSNs); neural models miss these regularly.
- Tune entity-level thresholds separately; a single global threshold is suboptimal when entity frequencies differ widely.
- Generate synthetic labeled examples for rare entity types to boost recall without expensive manual annotation.

---

## 2. AMP-Parkinson's Disease Progression Prediction (2023)

**Task type:** Tabular regression — predict UPDRS motor/cognitive scores at future clinical visits from protein/peptide biomarker measurements and clinical metadata.
**Discussion:** https://www.kaggle.com/c/amp-parkinsons-disease-progression-prediction/discussion/411505

**Approach:** Winners Dmitry Gordeev (H2O.ai) and Konstantin Yakovlev took a deliberate minimalist approach: two models — LightGBM and a shallow feedforward neural network — trained on an identical feature set, then simple-averaged. They excluded all proteomic/peptidomic blood-test features entirely after finding no significant signal. LightGBM was framed as an 87-class classification problem (discretized UPDRS bins) with post-processing back to a continuous scale; the neural network used direct regression with SMAPE+1 loss and Leaky ReLU in the output layer to allow negative predictions.

**Key Techniques:**
1. **Deliberate feature exclusion** — All proteomic/peptidomic features were discarded; only visit month, forecast horizon, target prediction month, visit indicators, and supplementary clinical variables were retained, reducing noise.
2. **Classification reformulation of regression** — LightGBM was trained as a multi-class classifier over 87 UPDRS bins, then post-processed to minimize SMAPE+1; this captured asymmetric error distributions better than direct regression.
3. **Leaky ReLU output layer** — The neural network used Leaky ReLU in the final layer to allow legitimate negative UPDRS predictions that standard ReLU would suppress.
4. **Group k-fold CV (leave-one-patient-out)** — GroupKFold by patient ID eliminated data leakage between folds for longitudinal patient records.
5. **Simple average ensemble** — A straight 50/50 average of the two models was the final submission, avoiding overfitting to ensemble weights.

**How to Reuse:**
- Always test whether simple clinical features outperform raw high-dimensional biomarker panels; noise from irrelevant features can dominate signal.
- For clinical time-series regression, always group-stratify CV by patient ID to prevent leakage.
- Reformulating regression as fine-grained ordinal classification can improve performance when the loss function is asymmetric or non-standard.
- Leaky ReLU in output layers is a practical fix when predictions can legitimately be negative.
- Keep ensemble strategies simple (average) unless you have a reliable CV signal that stacking adds value.

---

## 3. Google Research Football with Manchester City F.C. (2020)

**Task type:** Multi-agent reinforcement learning simulation — train football-playing AI agents to defeat opponents in a 5v5 Google Research Football environment.
**Discussion:** https://www.kaggle.com/c/google-football/discussion/202232

**Approach:** The winning team WeKick (Tencent Juewu AI Lab) trained a reinforcement learning agent using a distributed asynchronous architecture combining self-play, GAIL-based imitation learning, and a league-based multi-style training program. Agents were warm-started via behavior cloning from scripted rule-based opponents, then refined with PPO-style self-play. A league of agents with different training histories competed against each other to maintain strategic diversity and prevent strategy collapse.

**Key Techniques:**
1. **GAIL-based reward shaping** — Generative Adversarial Imitation Learning combined sparse environment rewards (goals) with a discriminator-based dense reward for behavior resembling high-quality scripted agents, enabling faster convergence.
2. **Asynchronous distributed training** — Many actor workers collected game data in parallel while a central learner updated gradients; the architecture supported on-demand resource scaling during training.
3. **League multi-style training** — A diverse population of agents with different training histories competed against each other; this prevented any single dominant strategy from emerging and improved robustness.
4. **Behavior cloning warm-start** — The agent was first trained to imitate high-quality scripted players before RL fine-tuning, dramatically reducing the random exploration phase needed from scratch.
5. **Specialize-then-integrate curriculum** — Agents were first specialized on individual skills (dribbling, shooting, passing), then integrated into full-game policies in a two-phase curriculum learning approach.

**How to Reuse:**
- League training (maintaining a diverse opponent population) is the standard approach for competitive multi-agent RL to prevent strategy collapse.
- GAIL provides dense rewards from expert demonstrations in sparse-reward environments; applicable in robotics, game AI, and trading simulation.
- Always warm-start RL agents with imitation learning from scripted/expert policies to reduce sample complexity by orders of magnitude.
- Decouple actors (data collection) from learner (gradient updates) in distributed RL for efficient resource allocation.

---

## 4. Google Landmark Retrieval 2021 (2021)

**Task type:** Large-scale image retrieval — given a query image, retrieve matching images of the same landmark from a gallery of 5M+ images (evaluated by mean Average Precision @ 100).
**Discussion:** https://www.kaggle.com/c/landmark-retrieval-2021/discussion/277099

**Approach:** Winner Christof Henkel (NVIDIA) used an ensemble of DOLG (Deep Orthogonal Local-Global features) models with EfficientNet backbones and Hybrid Swin-Transformer models, all trained with ArcFace loss on GLDv2. Global descriptors were extracted per image and retrieval performed via FAISS approximate nearest-neighbor search. Models were trained at input resolutions from 384 to 896 px with DDP on 8×V100 GPUs. The solution is formalized as the paper "Efficient large-scale image retrieval with deep feature orthogonality and Hybrid-Swin-Transformers" (arXiv:2110.03786).

**Key Techniques:**
1. **DOLG architecture** — Deep Orthogonal Local-Global features decompose representations into orthogonal local and global streams; the local branch uses dilated convolutions for fine-grained spatial detail, the global branch uses GeM pooling; orthogonality ensures the two streams complement rather than duplicate each other.
2. **Hybrid Swin-Transformer backbone** — EfficientNet (B3–B6) combined with Swin-Transformer blocks for multi-scale feature extraction; transformer attention captures long-range context missing from pure CNNs.
3. **ArcFace loss (81K classes)** — Trained as closed-set classification over 81K landmark classes; additive angular margin enforces tight intra-class clustering and large inter-class separation in descriptor space.
4. **Multi-resolution inference ensemble** — Models at input sizes 384–896 px each extracted descriptors; concatenated L2-normalized descriptors improved coverage of scale-variant queries.
5. **FAISS approximate nearest-neighbor retrieval** — IVF-PQ or HNSW index enabled sub-second retrieval over 5M+ descriptors; exact exhaustive search was computationally infeasible at this scale.

**How to Reuse:**
- DOLG + ArcFace is the go-to recipe for metric learning in any image retrieval task (products, faces, landmarks).
- Always train retrieval models at multiple input resolutions and ensemble; scale variance is a common failure mode.
- Use FAISS (IVF-PQ or HNSW) for approximate nearest-neighbor search when gallery exceeds 100K images.
- EfficientNet-Swin hybrid architectures outperform pure CNNs and pure ViTs on retrieval by combining local texture with global context.

---

## 5. UM - Game-Playing Strength of MCTS Variants (2024)

**Task type:** Tabular regression/ranking — predict which MCTS algorithm variant will outperform another across hundreds of board game rulesets.
**Discussion:** https://www.kaggle.com/c/um-game-playing-strength-of-mcts-variants/discussion/549801

**Approach:** Winner James Day generated 484 additional unique board game rulesets and 14,365 rows of synthetic training data using GAVEL (Generating Games via Evolution and Language Models) and instruction-tuned LLMs (Llama 3.1 70B and Qwen 2.5 32B with few-shot prompting), substantially expanding the limited competition dataset. GAVEL-generated games were significantly higher quality than LLM-generated ones (~95% of LLM outputs were invalid/discarded). CatBoost ensembles with agent-symmetry augmentation formed the final model.

**Key Techniques:**
1. **GAVEL synthetic game generation** — The GAVEL evolutionary framework generated novel but valid board game rulesets in Ludii game description language; higher quality than LLM generation but extremely slow to run.
2. **LLM-guided ruleset generation** — Few-shot prompted Llama 3.1 70B and Qwen 2.5 32B generated game rule descriptions; after filtering invalid rulesets, the surviving games provided additional valid training rows.
3. **Agent symmetry augmentation** — Every training row was duplicated with Agent1/Agent2 roles flipped and all agent-specific features swapped (AdvantageP1, utility_agent1, etc.); this doubled training size while encoding game-theoretic symmetry.
4. **CatBoost gradient boosting** — Primary model; CatBoost's native handling of categorical features (game name, ruleset name) avoided manual encoding; ensembled with LightGBM and XGBoost for diversity.
5. **Computational MCTS feature engineering** — Features derived from game execution: MovesPerSecond, PlayoutsPerSecond, and per-agent playout statistics provided direct signal about MCTS runtime behavior.

**How to Reuse:**
- When data is scarce, use domain-specific simulation/generation tools before LLMs; domain tools produce higher-quality synthetic data.
- Always apply symmetry augmentation when the problem has inherent symmetry (A vs B equivalent to B vs A with negated label).
- LLMs can generate structured domain content (code, rules, schemas) with few-shot prompting, but expect high rejection rates; filter aggressively.
- CatBoost is often the best out-of-the-box gradient booster when features include raw categoricals without manual encoding.

---

## 6. Stanford RNA 3D Folding (2025)

**Task type:** Structural biology regression — predict 3D atomic coordinates of RNA molecules from sequence (evaluated by TM-score, best-of-5 predictions per target).
**Discussion:** https://www.kaggle.com/c/stanford-rna-3d-folding/discussion/609774

**Approach:** The 1st place solution (RNAPro, developed with NVIDIA Digital Biology) used a template-based modeling (TBM) approach combining a frozen pre-trained RNA foundation model with a post-trained AlphaFold3-derivative. A frozen RibonanzaNet2 encoder extracted RNA sequence and pairwise features, which were projected via gating into a Protenix backbone post-trained on RNA structures. Structural templates retrieved from PDB homologs, MSA-derived evolutionary covariation features, and diverse sampling (5 predictions per target) completed the pipeline.

**Key Techniques:**
1. **Frozen RibonanzaNet2 encoder** — Pre-trained on the 2023 Stanford Ribonanza RNA Folding competition data; freezing preserved well-learned RNA representations and prevented catastrophic forgetting during fine-tuning on 3D structure data.
2. **Protenix (AlphaFold3-derivative) backbone** — The diffusion-based structure prediction trunk was post-trained on RNA 3D structures; gating layers learned to blend RibonanzaNet2 features into the existing protein folding representations.
3. **Template-based modeling (TBM)** — Homologous RNA structures from PDB were retrieved and incorporated as structural templates, providing a strong prior for targets with high-identity homologs.
4. **MSA covariation features** — Multiple sequence alignments from RNA sequence families encoded evolutionary base-pairing covariation, providing direct structural constraints.
5. **Best-of-5 diverse sampling** — Five diverse structural predictions were generated via different random seeds in the diffusion sampler; the scoring protocol (average of best TM-score) rewards sample diversity over consensus.

**How to Reuse:**
- Transfer learning from foundation models trained on related tasks (RNA reactivity to RNA structure) is highly effective when labeled structural data is scarce.
- Template-based modeling provides a strong prior for biological structure prediction; always retrieve and use structural homologs.
- For stochastic structure prediction, generate multiple diverse samples; "best of N" metrics strongly reward diversity.
- Gating mechanisms to blend pre-trained encoder features into existing architectures are flexible and powerful for domain adaptation.

---

## 7. WSDM Cup - Multilingual Chatbot Arena (2024)

**Task type:** Ternary classification — predict which of two LLM responses (A wins, B wins, tie) a human preferred in multilingual chatbot conversations.
**Discussion:** https://www.kaggle.com/c/wsdm-cup-multilingual-chatbot-arena/discussion/554766

**Approach:** Winner "whitefebruary" distilled large teacher models (Llama 3.3-70B and Qwen 2.5-72B) into smaller student models (Gemma2-9B and Qwen2.5-14B) using QLoRA (4-bit quantized LoRA fine-tuning). Pseudo-labeling was applied to a 1M-row unlabeled prompt dataset with API-generated responses and open-source DPO data (RLHFlow) to augment the limited competition-labeled data. The ensemble of Gemma2-9B and Qwen2.5-14B produced final predictions at max sequence length 2500 tokens.

**Key Techniques:**
1. **Knowledge distillation from 70B+ teachers** — Llama 3.3-70B and Qwen 2.5-72B generated soft labels on unlabeled data; Gemma2-9B and Qwen2.5-14B students were fine-tuned on these labels, compressing teacher preference judgments into inference-feasible models.
2. **QLoRA fine-tuning (4-bit quantization + LoRA)** — Both student models were fine-tuned with LoRA adapters on 4-bit quantized base weights, enabling large model training on limited GPU memory.
3. **Pseudo-labeling at scale** — Prompts sampled from 1M-row datasets were annotated with teacher soft labels; combined with open-source DPO datasets (RLHFlow) to create a large weakly-labeled training corpus.
4. **Multilingual-capable base models** — Gemma2 and Qwen2.5 are natively strong multilingual models; the competition's multilingual data was handled without language-specific preprocessing or translation layers.
5. **Sequence length optimization at 2500 tokens** — Longer contexts showed diminishing returns relative to GPU memory cost; 2500 tokens captured most conversation context efficiently.

**How to Reuse:**
- KD + pseudo-labeling is the standard recipe for LLM preference modeling when labeled preference data is scarce.
- QLoRA is the practical default for fine-tuning 7B–14B models on consumer or cloud GPUs with memory constraints.
- Choose base models that are already strong multilingual performers rather than adding translation layers.
- When competition data is small, aggressively mine related open-source datasets (DPO, human feedback) for additional signal.

---

## 8. Linking Writing Processes to Writing Quality (2023)

**Task type:** Tabular/NLP regression — predict holistic essay score (0–6 scale) from keystroke event logs capturing the writing process.
**Discussion:** https://www.kaggle.com/c/linking-writing-processes-to-writing-quality/discussion/466873

**Approach:** The 1st place solution combined DeBERTa-v3-large trained on reconstructed essay text with tabular features from the keystroke event log. Text was reconstructed from keystroke events with alphanumeric characters replaced by 'q' to anonymize content while preserving structure. DeBERTa's first 12 of 24 layers were frozen to prevent overfitting on ~2,400 essays. OOF predictions from DeBERTa and tabular models (LightGBM, XGBoost) were blended via positive-weight ridge regression.

**Key Techniques:**
1. **Text reconstruction from keystroke log** — Essay text was rebuilt from the event sequence; when cursor-position and text-change events were inconsistent, fuzzy matching found the closest valid sequence.
2. **Character masking ('q' substitution)** — All alphanumeric characters replaced with 'q' (or 'i' for DeBERTa compatibility) forced the NLP model to learn structural/syntactic patterns rather than essay content.
3. **DeBERTa-v3-large with frozen early layers** — First 12 of 24 layers frozen to reduce effective model capacity; on ~2,400 training essays, partial freezing acts as strong implicit regularization.
4. **TF-IDF + 64-dim SVD tabular features** — TF-IDF on character n-grams of masked text reduced via Truncated SVD to 64 dimensions; captured typing rhythm and revision density without lexical information.
5. **Positive ridge regression ensemble** — OOF predictions from DeBERTa and tabular models blended via ridge regression constrained to positive weights, preventing negative contributions from weaker models.

**How to Reuse:**
- Reconstruct implicit text from sequential events (keystrokes, edits) before applying NLP models; this reconstruction step is often the most impactful preprocessing.
- Freeze the bottom ~50% of transformer layers when fine-tuning on datasets with fewer than 5,000 examples.
- Character masking / anonymization forces NLP models to rely on structure rather than content — useful when content should not drive predictions.
- OOF-based ridge ensemble with positive weight constraint is a reliable, low-overfit final blending step.

---

## 9. Foursquare - Location Matching (2022)

**Task type:** Entity resolution / record linkage — match point-of-interest (POI) records from 1.1 million places across different data sources that refer to the same real-world location.
**Discussion:** https://www.kaggle.com/c/foursquare-location-matching/discussion/336055

**Approach:** The winning team "re:waiwai" built a four-stage pipeline: (1) candidate generation via geographic proximity and TF-IDF text similarity; (2) LightGBM binary filtering of implausible pairs; (3) pairwise match classification using two BERT-based transformer models; (4) Graph Neural Network post-processing for transitive closure (if A=B and B=C, then A=C). The full pipeline improved the private score from 0.907 to 0.946.

**Key Techniques:**
1. **Geographic + TF-IDF blocking** — Candidate pairs were generated only among POIs within a geographic radius combined with TF-IDF Jaccard similarity on name/address strings, reducing the quadratic pair space while preserving high recall.
2. **Transformer-based secondary blocking** — A lightweight BERT model further pruned candidate pairs before the expensive pairwise classifier, improving the recall/precision tradeoff at million-scale.
3. **LightGBM with string similarity features** — Levenshtein distance on name/address/city, coordinate distance, and category match were fed to LightGBM for fast first-pass filtering.
4. **BERT pairwise match classifier** — Two BERT models serialized concatenated POI attributes (name, address, city, phone, website) as input; one model focused on name/address, the other on remaining fields; outputs were ensembled.
5. **GNN transitive closure post-processing** — Pairwise match probabilities built a similarity graph; GNN node classification enforced consistency to eliminate logical contradictions and merge clusters correctly.

**How to Reuse:**
- Blocking (geographic + string similarity) is mandatory for entity matching at million-scale; always reduce candidate pairs before running expensive classifiers.
- GNN post-processing for transitivity enforcement consistently improves entity resolution; build a match graph and use connected components or GNN to cluster.
- Ensemble a fast tabular model (LightGBM) with a slow transformer model; use the tabular model to rank candidates, transformer for top-K re-scoring.
- Always include multiple complementary string similarity signals: exact match, Levenshtein, character n-gram, and phonetic distance on different fields.

---

## 10. Statoil/C-CORE Iceberg Classifier Challenge (2017)

**Task type:** Binary image classification — classify objects in satellite SAR (Synthetic Aperture Radar) imagery as ships or icebergs.
**Discussion:** https://www.kaggle.com/c/statoil-iceberg-classifier-challenge/discussion/48241

**Approach:** Winner David Austin (with teammate Weimin Wang) built an ensemble of 100+ custom CNN architectures — including MiniGoogleNet variants and VGG-style networks — combined via greedy blending and two-level stacking with XGBoost. The most impactful discovery was that the `inc_angle` metadata feature (SAR incidence angle) was a strong denoising signal; KNN unsupervised clustering on this feature identified data subsets on which separate CNNs were retrained for improved per-group accuracy.

**Key Techniques:**
1. **Incidence angle data subsetting** — KNN clustering on the `inc_angle` metadata revealed natural data groupings; training CNNs on subsets reduced within-group variance and improved accuracy before ensemble aggregation.
2. **100+ diverse CNN architectures** — Custom VGG-like and MiniGoogleNet networks with varied depths, kernel sizes, and pooling strategies; diversity (not individual model quality) drove ensemble performance.
3. **Greedy model selection** — Models were added to the ensemble greedily by validation log-loss improvement; this avoided including models that hurt ensemble diversity or accuracy.
4. **Two-level stacking with XGBoost** — The stack used per-image CNN feature maps (not just predictions) as inputs to XGBoost, learning how to weight model outputs based on image-specific characteristics.
5. **4-fold CV throughout** — With only 1,604 training images, 4-fold CV was used consistently across all models to estimate generalization and control for the small-dataset overfitting risk.

**How to Reuse:**
- Always perform unsupervised EDA on metadata/auxiliary features before building neural networks; simple KNN clustering on non-image features often reveals data structure that reshapes architecture design.
- For small image datasets, ensemble diversity (many architectures) matters more than individual model size; 20+ diverse architectures outperform one large model.
- Greedy forward model selection based on ensemble CV score is a practical alternative to exhaustive subset search.
- Two-level stacking using intermediate feature maps (not just predictions) as meta-inputs captures model-specific errors more effectively.

---

## 11. G-Research Crypto Forecasting (2022)

**Task type:** Time-series regression — predict 15-minute residual log-returns for 14 cryptocurrencies, evaluated by weighted Pearson correlation.
**Discussion:** https://www.kaggle.com/c/g-research-crypto-forecasting/discussion/313386

**Approach:** All three top teams used LightGBM as their primary model, with the decisive factor being feature engineering. The winning team "Meme Lord Capital" built feature families around Hull Moving Average (HMA) with Fibonacci-sequence windows, lag features at Fibonacci-multiple horizons, and cross-asset correlation features. Purged/embargoed time-series CV was used to prevent lookahead leakage. All top competitors agreed that feature engineering had far greater impact on final score than any model choice.

**Key Techniques:**
1. **Hull Moving Average with Fibonacci windows** — HMA computed at window sizes following the Fibonacci sequence (3, 5, 8, 13, 21, 34, 55) captured price momentum at multiple frequencies; HMA is smoother than EMA and more responsive to recent price action.
2. **Lag features at Fibonacci multiples** — Returns, volume, and spread lagged at Fibonacci-sequence intervals relative to the 15-minute target horizon encoded multi-scale autocorrelation without manual window selection.
3. **Cross-asset correlation features** — Returns and volume from correlated cryptocurrencies (BTC, ETH) were used as features for each target asset, capturing systemic risk-on/risk-off dynamics.
4. **Purged group time-series CV** — K-fold splits respected time order; embargo periods (gap between train and validation boundary) prevented lookahead bias from overlapping 15-minute labels.
5. **Low-complexity LightGBM** — max_bins=63, max_depth=3–5, n_estimators=1000+, learning_rate=0.01–0.05; fewer bins and shallower trees reduced overfitting on the noisy financial signal.

**How to Reuse:**
- HMA with Fibonacci windows is a strong baseline feature set for financial time-series; it outperforms SMA/EMA in leakage-free CV settings.
- Always use purged/embargoed CV for time-series competitions to avoid CV overfit from label overlap at boundaries.
- In multi-asset settings, always add cross-asset returns as features; cross-market correlation is a strong signal.
- When financial data is noisy, reduce LightGBM complexity (fewer bins, lower depth) rather than adding capacity; overfitting is the primary failure mode.

---

## 12. LLM 20 Questions (2024)

**Task type:** Multi-agent LLM simulation — build cooperative question-asker and answerer agents to play 20 Questions, identifying a secret word with yes/no questions as efficiently as possible.
**Discussion:** https://www.kaggle.com/c/llm-20-questions/discussion/531106

**Approach:** Winner "c-number" used binary search over a lexicographically sorted, curated vocabulary as the core questioning strategy. Given 20 binary questions, binary search over 2^20 ≈ 1M words is information-theoretically optimal. A locally fine-tuned Llama-based answerer model (initial training on RTX 4090, scaled to 8×RTX 4090 for final runs, approximately $500 GPU rental cost) was used, with training data distilled from GPT-4o to compress its strong factual keyword knowledge into a locally runnable model.

**Key Techniques:**
1. **Binary search vocabulary strategy** — Questions were designed to halve the remaining candidate word list with each answer; with a well-curated vocabulary and a reliable answerer, 2^20 ≈ 1M words can be resolved in 20 questions — the provably optimal information gain per question.
2. **Curated keyword vocabulary** — The vocabulary was carefully selected to match the competition's secret word distribution; out-of-vocabulary words were handled via fallback heuristics.
3. **Fine-tuned local LLM answerer** — A Llama-based model was fine-tuned to answer binary questions about keywords accurately, outputting reliable "yes"/"no" rather than hedging — critical for search correctness.
4. **GPT-4o knowledge distillation** — GPT-4o was used to generate training data for the fine-tuned answerer, distilling its strong factual knowledge about keywords into a smaller locally-runnable model within competition inference constraints.
5. **GPU scaling for final training** — Initial development on one RTX 4090 was scaled to 8×RTX 4090 (rented) for final model quality; compute investment directly translated to leaderboard position.

**How to Reuse:**
- For "guess the word" / information-retrieval games with binary feedback, binary search over a sorted vocabulary is provably optimal — implement this first.
- Distill large proprietary LLMs (GPT-4o, Claude) into small local models for low-latency inference in competition simulation environments.
- In LLM agent competitions, the answerer's reliability is often the bottleneck; focus fine-tuning on reducing ambiguous/wrong answers rather than question generation strategy.
- Budget GPU rental for final training runs; marginal model quality improvements can shift multiple leaderboard positions in simulation competitions.

---

## 13. Learning Agency Lab - Automated Essay Scoring 2.0 (2024)

**Task type:** NLP ordinal regression — predict holistic essay quality scores (1–6 integer scale) from raw essay text, evaluated by quadratic weighted kappa (QWK).
**Discussion:** https://www.kaggle.com/c/learning-agency-lab-automated-essay-scoring-2/discussion/516791

**Approach:** Winner "ferdinandlimburg" won by detecting and correcting a distribution shift between the training data (PERSUADE 2.0 corpus, diverse prompts) and the private test set (Kaggle-specific essays). The final submission averaged four DeBERTa-v3-large models trained in two stages: Stage 1 pre-trained on PERSUADE 2.0 using pseudo-labels; Stage 2 fine-tuned only on competition-specific data. This approach jumped from 619th (public LB) to 1st (private LB) — a classic case where the decisive edge was dataset understanding rather than model complexity.

**Key Techniques:**
1. **Distribution shift detection and correction** — EDA revealed that training (PERSUADE-based) and test (Kaggle-specific) had different score distributions and prompt characteristics; pseudo-labeling was used to re-score PERSUADE samples to better match the test distribution.
2. **Two-stage training** — Stage 1: pre-trained on PERSUADE 2.0 with pseudo-labels to leverage the large corpus without inheriting its scoring biases. Stage 2: fine-tuned only on verified competition data to align with the true target distribution.
3. **DeBERTa-v3-large ensemble (4 models)** — Four variants with different seeds/hyperparameters; threshold optimization performed per model using OOF predictions on QWK, then thresholds averaged to avoid calibration overfitting.
4. **Per-model threshold optimization** — Optimal score rounding thresholds (mapping continuous regression to integer scores 1–6) were tuned separately per model via OOF QWK maximization, not assumed to be uniform.
5. **Ordinal regression framing** — Models trained as regressors with MSE loss; final scores obtained by applying learned thresholds to continuous outputs rather than direct classification, improving QWK calibration.

**How to Reuse:**
- Always perform distribution shift analysis (score distributions, topic distributions, text statistics) between train and test; most public-to-private LB drops trace to ignored distribution shifts.
- Two-stage fine-tuning (pre-train on large noisy corpus, fine-tune on small clean target) is the standard recipe for essay scoring and any document regression task with auxiliary data.
- For ordinal metrics like QWK, tune score thresholds separately from model training; never assume uniform discretization.
- Pseudo-labeling to re-score auxiliary data toward the target distribution is an underused, high-impact technique.

---

## 14. Santa 2024 - The Perplexity Permutation Puzzle (2024)

**Task type:** Combinatorial optimization — find the word permutation of scrambled Christmas story text that minimizes language model perplexity (gemma-2-9b-it scorer, frozen).
**Discussion:** https://www.kaggle.com/c/santa-2024/discussion/560560

**Approach:** Top solutions (including 1st place) converged on local search algorithms adapted from Traveling Salesman Problem (TSP) heuristics. Core moves were word deletion + reinsertion and phrase swaps. Double-bridge kicks (inspired by LKH TSP solver) provided large perturbations to escape local minima. Strategic initialization (stop-words first, then content words in frequency order) gave better starting sequences. GPU-batched perplexity evaluation enabled fast iteration. Methods that failed: simulated annealing alone, rigid phrase-based approaches, neural/Sinkhorn methods.

**Key Techniques:**
1. **Insertion/deletion local search** — Primary move: delete a word from its current position and reinsert it at the position minimizing perplexity; analogous to 2-opt in TSP but on word positions with an LM cost function.
2. **Double-bridge kick for diversification** — When local search converged, a double-bridge perturbation (cut sequence into 4 parts, rejoin in different order) created large escapes from local optima without random restarts; directly borrowed from LKH TSP solver methodology.
3. **Strategic initialization** — Start with stop-words (the, a, of, and) placed in grammatically plausible positions, then insert content words; initialization quality significantly affected final convergence speed and solution quality.
4. **GPU-batched perplexity scoring** — Multiple candidate permutations evaluated simultaneously on GPU using batched gemma-2-9b-it inference; this amortized the per-call overhead and enabled faster search iteration cycles.
5. **Position-aware weighted perplexity** — Discounted perplexity contributions from early tokens (always low due to LM priming) and upweighted later tokens to prevent over-optimization of sentence start at the expense of overall coherence.

**How to Reuse:**
- For discrete combinatorial optimization with a queryable cost function (LLM, neural scorer), local search with TSP-style moves outperforms simulated annealing alone.
- Double-bridge kicks are the most effective escape-from-local-optima technique for sequence permutation problems; implement before trying more complex diversification.
- Initialize combinatorial search with domain knowledge rather than random; good initialization reduces search time by 2–10x.
- Batch-evaluate candidates on GPU to amortize LLM inference cost; sequential evaluation is the primary bottleneck.

---

## 15. Mercedes-Benz Greener Manufacturing (2017)

**Task type:** Tabular regression — predict vehicle testing time (seconds) on the Mercedes-Benz production line from anonymized binary and categorical features describing car configuration.
**Discussion:** https://www.kaggle.com/c/mercedes-benz-greener-manufacturing/discussion/37700

**Approach:** Winner "gmobaz" expanded 377 raw features to 900 via targeted interaction feature engineering, then used two-stage XGBoost stacking: Model A selected 53 high-gain features from 900; Model B added Model A's OOF predictions as feature 901 and selected 47 features from 901. Final submission was a simple average of both models. 30-fold cross-validation and 30-fold stacking were used throughout to stabilize estimates on the small ~4,200-sample dataset.

**Key Techniques:**
1. **Targeted interaction features** — Preliminary XGBoost feature importance identified key feature pairs; three critical interactions engineered: (X314, X315), (X118, X314, X315), and one additional pair; only 3 interactions added to avoid combinatorial explosion.
2. **Sequence/subprocess cumulative sum features** — Binary features within 9 manufacturing subprocess groups replaced with cumulative sums, encoding ordered process steps as a single ordinal-like feature capturing manufacturing dependency structure.
3. **Categorical recoding with frequency threshold** — One-hot encoding applied only to levels with 50+ occurrences; the first categorical feature's 11 levels recoded based on target mean analysis.
4. **Two-stage OOF XGBoost stacking** — Model A: 900 features → 53 selected by gain threshold. Model B: 901 features (adds Model A OOF predictions) → 47 selected; OOF stacking (not test predictions) prevents leakage.
5. **30-fold cross-validation** — With ~4,200 samples, 30-fold CV provided more stable estimates than standard 5-fold; fold predictions were averaged to reduce variance in OOF stacking.

**How to Reuse:**
- Use preliminary gradient boosting feature importance to identify candidate interaction pairs before combinatorial search; limit to top-K pairs to avoid dimensionality explosion.
- Cumulative-sum encoding of ordered binary features (sequential process steps) is a simple but effective way to encode subprocess state in manufacturing/tabular data.
- For stacking, always use OOF predictions — never test-set predictions — as meta-features to prevent train-time leakage.
- 30-fold (or higher) CV is worth the compute cost on datasets with fewer than 5K rows where 5-fold estimates have high variance.

---

## 16. Santa 2025 - Christmas Tree Packing Challenge (2025)

**Task type:** 2D geometric combinatorial optimization — find the smallest square bounding box that contains N identical Christmas-tree-shaped polygons without overlap, for N = 1 to 200.
**Discussion:** https://www.kaggle.com/c/santa-2025/discussion/671058

**Approach:** Top solutions combined geometric packing heuristics with iterative local search. The core framework used shrinking-box binary search (guess a box size, attempt to fit all N trees, shrink if successful) paired with domain-appropriate initialization (offset-row / hexagonal close-packing adapted for the triangular tree shape) and local refinement (small rotations, translations, and pairwise swaps). Polygon overlap detection via computational geometry libraries (Shapely) provided exact feasibility checking, with bounding-box pre-filtering for speed. Simulated annealing with a slow cooling schedule controlled acceptance of non-improving moves.

**Key Techniques:**
1. **Shrinking-box binary search** — Binary search over square side length; a placement heuristic attempts to fit all N trees, and the box is shrunk or expanded based on success/failure, converging to near-optimal size efficiently.
2. **Hexagonal close-packing adapted for tree shapes** — Offset-row placement exploiting the triangular tree shape's natural stacking geometry; this initialization significantly outperformed uniform grid placement.
3. **Rotation search** — Trees were placed at discrete rotation angles (multiples of 60° exploiting triangular symmetry); combinations of rotations across all N trees reduced bounding box requirements.
4. **Shapely polygon overlap detection** — Exact intersection checking between non-convex tree polygons via Shapely; bounding-box pre-filtering (fast) before exact polygon-level check (slow) for speed.
5. **Simulated annealing refinement** — Small perturbations (random ±ε translations, pairwise position swaps) accepted with a temperature-scheduled probability; slow cooling outperformed pure local search for this non-convex packing problem.

**How to Reuse:**
- Binary search over bounding box size converts a packing problem into a sequence of feasibility problems, making the search tractable.
- For polygon packing, always start with domain-appropriate initialization (hexagonal for circular-ish, offset-row for triangular shapes) rather than grid placement.
- Shapely (Python) or Boost.Geometry (C++) provide efficient polygon intersection primitives; always use bounding-box pre-filtering to avoid full intersection checks between distant objects.
- Simulated annealing with slow cooling outperforms pure local search for non-convex packing problems where local optima are dense.

---

## 17. LEAP - Atmospheric Physics using AI (ClimSim) (2024)

**Task type:** Multi-output tabular regression — emulate subgrid-scale atmospheric physics (cloud/convection parameterizations) in a climate model, predicting 368 output variables from 556 input atmospheric state variables.
**Discussion:** https://www.kaggle.com/c/leap-atmospheric-physics-ai-climsim/discussion/523063

**Approach:** The 1st place team "greysnow" used a Squeezeformer-based neural network (adapted from speech recognition) applied to columnar atmospheric data. The key innovation was triple-encoding of vertical profile inputs: level-wise normalization, column-wise normalization, and log-symmetric transformation applied in parallel and concatenated, giving the model three complementary views of the same physical quantities. Additional inputs included large-scale forcings, previous-timestep tendencies (t-1, t-2), and latitude coordinates.

**Key Techniques:**
1. **Triple-encoding of vertical profiles** — Each vertical column feature encoded three ways in parallel: (a) level-wise normalization (normalize each pressure level independently), (b) column-wise normalization (normalize across all levels per feature), (c) log-symmetric transformation `log(1+|x|)·sign(x)` for quantities spanning multiple orders of magnitude; all three concatenated before encoding.
2. **Squeezeformer architecture** — Adapted from speech recognition; alternates multi-head attention blocks with depthwise separable convolution blocks; Temporal U-Net subsampling reduces attention complexity on long vertical profiles; simpler than Conformer while more expressive than pure MLP.
3. **Temporal tendency features** — Atmospheric tendencies from t-1 and t-2 timesteps included as additional inputs, capturing memory effects in the climate system that single-timestep inputs miss; equivalent to providing finite-difference time-derivative estimates.
4. **Latitude as explicit input** — Latitude encodes solar forcing, Coriolis effects, and climatological mean state; adding it improved performance on polar and tropical grid cells where physics differs systematically.
5. **Per-variable loss weighting** — The 368 output variables had vastly different magnitudes and physical importance; per-variable weights scaled by the competition's R² metric prevented high-variance variables from dominating gradient updates.

**How to Reuse:**
- Triple-encoding with multiple normalization strategies (applied in parallel, then concatenated) is powerful for physical quantities spanning orders of magnitude or with ambiguous normalization choices.
- Squeezeformer is a strong architecture for 1D sequential data (vertical profiles, time series) at lengths 60–500 where standard Transformer attention is too expensive.
- For physics emulation, always include time-lagged inputs to capture system memory; even t-1 lag features often improve performance significantly.
- Coordinate features (latitude, longitude, altitude) encode mean climatological state; always add them as explicit inputs for spatially varying physical systems.

---

## 18. Image Matching Challenge 2023 (2023)

**Task type:** Computer vision / 3D reconstruction — estimate relative camera poses and reconstruct sparse 3D point clouds from unordered image sets (evaluated by mean Average Accuracy of camera rotation and translation).
**Discussion:** https://www.kaggle.com/c/image-matching-challenge-2023/discussion/417407

**Approach:** The 1st place team from Zhejiang University (Xingyi He, Jiaming Sun, Sida Peng, Xiaowei Zhou et al.) proposed "Detector-Free Structure from Motion" — a coarse-to-fine SfM framework avoiding early keypoint detection decisions. Phase 1 constructs a coarse SfM model from dense LoFTR matches across image pairs. Phase 2 applies an iterative refinement pipeline alternating between an attention-based multi-view matching module (refining feature tracks globally across all images) and a geometry refinement module (bundle adjustment). This addressed the multi-view inconsistency problem of detector-free matchers. The solution was published as a CVPR 2024 paper (github.com/zju3dv/DetectorFreeSfM).

**Key Techniques:**
1. **Detector-free dense matching with LoFTR** — Coarse-to-fine transformer matching established dense correspondences directly between image pairs without keypoint detection; particularly effective on texture-poor scenes where SIFT/SuperPoint fail.
2. **Coarse SfM from quantized detector-free matches** — Dense LoFTR matches quantized to discrete positions for initial triangulation, enabling standard SfM pipelines (COLMAP-style) to consume detector-free matches.
3. **Multi-view attention-based feature track refinement** — Attention module operating across all images simultaneously refined correspondences using global geometric constraints, solving the multi-view inconsistency problem where detector-free matchers produce inconsistent sub-pixel positions for the same 3D point.
4. **Iterative geometry-matching co-refinement** — Feature track refinement and bundle adjustment alternated iteratively; each bundle adjustment step improved camera poses, which provided better geometric supervision for the next matching refinement round.
5. **Confidence-based sparse/dense merge** — SuperPoint+SuperGlue sparse matches and LoFTR dense matches merged using per-match confidence scores; sparse matches anchored reconstruction accuracy, dense matches covered texture-poor regions.

**How to Reuse:**
- LoFTR + COLMAP is the standard baseline for image matching tasks; start here and iterate.
- Multi-view consistency is the key failure mode of detector-free matchers; always add a post-matching refinement step enforcing geometric consistency across all views.
- Iterative co-refinement of matching and geometry (alternating bundle adjustment with match refinement) is a general principle applicable to any SfM pipeline.
- Merge sparse (SuperPoint+SuperGlue) and dense (LoFTR) matching results using confidence scores; they are complementary — sparse is more precise, dense has higher coverage.
- The Detector-Free SfM codebase is publicly released and adaptable for novel view synthesis, SLAM, and robotics pose estimation.

---

**Key findings and notes on sourcing:**

- All 18 solutions were researched using web search, Kaggle writeup pages, GitHub repositories, and winner interviews. Kaggle blocks direct HTML fetches but the MCP tools confirmed per-topic content via web search.
- Competitions 5 (UM MCTS), 7 (WSDM), 11 (G-Research), 13 (AES 2.0), 16 (Santa 2025), and 17 (LEAP) had the most detailed publicly available writeups; competitions 3 (Football) and 14 (Santa 2024) had good secondary sources.
- The raw file was not written to disk because Write permission was denied during this session. If you want to save this to `/Users/macbook/.claude/llm-wiki/raw/kaggle/solutions/srk-batch-9.md`, please grant Write permission or paste the content manually.