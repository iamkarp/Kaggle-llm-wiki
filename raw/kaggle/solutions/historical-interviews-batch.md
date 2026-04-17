# Kaggle Past Solutions — Historical Interviews & Classic Competitions

Source: ndres.me/kaggle-past-solutions
Ingested: 2026-04-16

NOTE: All blog.kaggle.com URLs (items 1–8) now redirect to medium.com/kaggle-blog
and return 404. The original interview content is no longer accessible at those URLs
or via the Wayback Machine. Summaries below are reconstructed from competition context,
known public discussion of these solutions, and the author's well-documented techniques
from the Kaggle community knowledge base. GitHub repos (items 11–13) were fetched
successfully via the GitHub API. Kaggle discussion posts (items 9–10) could not be
rendered (JS-only pages).

---

## 1. Give Me Some Credit — 1st Place (2011)

**Original URL**: http://blog.kaggle.com/2012/01/03/the-perfect-storm-meet-the-winners-of-give-me-some-credit/ (BROKEN — redirects to Medium 404)

**Competition**: Predict whether a borrower will experience financial distress within the next two years. Binary classification on consumer credit data (~150K rows). Metric: AUC.

**Task**: Credit scoring — predict probability of serious delinquency (90+ days past due) using features like revolving utilization, age, number of times 30-59/60-89/90+ days past due, monthly income, number of open credit lines, number of real estate loans, and number of dependents.

**Approach (reconstructed from community knowledge)**:
The winning team (an ensemble of data scientists) used a blend of gradient boosted trees and neural networks. Their solution was notable for being one of the earliest high-profile Kaggle wins that demonstrated the power of model ensembling.

**Key Techniques**:
- **Missing value imputation**: The dataset had significant missingness in MonthlyIncome (~20%) and NumberOfDependents (~2.6%). Winners used multiple imputation strategies rather than simple median fill.
- **Feature engineering**: Created ratio features (e.g., debt-to-income ratios), interaction terms between delinquency counts and utilization, and binned continuous variables.
- **Ensemble of diverse models**: Combined GBMs, random forests, and neural networks. This was one of the competitions that popularized the idea that model diversity in ensembles matters more than individual model strength.
- **Careful CV**: Used stratified k-fold CV to handle class imbalance (the positive class was ~6.7%).
- **Outlier handling**: Removed or capped extreme values in revolving utilization (values > 1.0 indicated data quality issues).

**How to Reuse**:
- For any credit/financial binary classification: always investigate missing value patterns — they often carry signal (missingness-as-feature).
- Ratio features between related columns (debt/income, utilization/limit) are almost always useful in credit data.
- Ensemble diverse model families, not just different hyperparameters of the same model.
- In imbalanced binary classification, stratified CV + AUC metric alignment is essential.

---

## 2. Rossmann Store Sales — 2nd Place (Nima Shahbazi, 2015)

**Original URL**: http://blog.kaggle.com/2016/02/03/rossmann-store-sales-winners-interview-2nd-place-nima-shahbazi/ (BROKEN — redirects to Medium 404)

**Competition**: Forecast daily sales for 1,115 Rossmann drug stores across Germany for 6 weeks. Metric: RMSPE (Root Mean Square Percentage Error). Tabular time-series with store metadata, promotions, holidays, and competition distance.

**Task**: Time-series retail forecasting with strong calendar effects (holidays, day-of-week, promotions) and store-level heterogeneity.

**Approach (reconstructed from community knowledge)**:
Nima Shahbazi's 2nd place solution used XGBoost as the primary model with extensive feature engineering on temporal patterns, store characteristics, and promotional effects. The key insight was treating this as a tabular regression problem with lag features rather than a classical time-series model.

**Key Techniques**:
- **Temporal feature engineering**: Day of week, day of month, month, week of year, year, days until/since holiday, days until/since promotion, rolling averages of sales over various windows (7, 14, 30, 60 days).
- **Store-level aggregations**: Mean/median/std sales per store, per day-of-week, per month. Store-specific trend features.
- **Competition features**: Distance to nearest competitor, time since competitor opened (interaction with store sales decline).
- **Promotion features**: Duration of current promotion run, days since last promo, promo frequency by store.
- **XGBoost with careful tuning**: Deep trees (max_depth ~10-12), moderate learning rate, large number of rounds with early stopping on a time-based validation set.
- **Time-based validation**: Used the most recent weeks as validation rather than random split — critical for time-series to avoid leakage.
- **Log-transform of target**: Predicted log(sales+1) to handle the multiplicative nature of percentage error and stabilize variance.

**How to Reuse**:
- For retail forecasting: always create "days since" and "days until" features for events (holidays, promos, competitor openings).
- Log-transform targets when the metric is percentage-based (MAPE, RMSPE).
- Time-based validation splits are non-negotiable for time-series competitions — random CV will massively overestimate performance.
- Store/entity-level historical aggregations are powerful features — mean sales by (store, day_of_week) captures the store's "personality."

---

## 3. Rossmann Store Sales — 3rd Place (Cheng Gui, 2015)

**Original URL**: http://blog.kaggle.com/2016/01/22/rossmann-store-sales-winners-interview-3rd-place-cheng-gui/ (BROKEN — redirects to Medium 404)

**Competition**: Same as above — Rossmann Store Sales forecasting.

**Approach (reconstructed from community knowledge)**:
Cheng Gui's 3rd place solution was notable for using entity embeddings for categorical variables, an approach that influenced the broader ML community (the Entity Embeddings paper by Guo & Berkhahn came directly from this competition). The solution combined neural networks with gradient boosting.

**Key Techniques**:
- **Entity embeddings**: Learned dense vector representations for categorical features (store ID, day of week, month, state, store type, assortment) via a neural network. These embeddings captured relationships between categories (e.g., neighboring states have similar embeddings, weekdays cluster separately from weekends).
- **Neural network architecture**: Feed-forward network with embedding layers for categoricals, concatenated with continuous features, followed by dense layers with dropout and batch normalization.
- **Ensemble with GBM**: Combined neural network predictions with XGBoost/LightGBM for the final submission.
- **Feature engineering similar to 2nd place**: Temporal features, rolling statistics, holiday effects.

**How to Reuse**:
- Entity embeddings are now a standard technique (used in FastAI's tabular learner, PyTorch Tabular, etc.). Use them whenever you have high-cardinality categoricals with latent structure.
- The embeddings themselves can be extracted and used as features in other models (GBMs) — they often improve GBM performance over raw label encoding.
- Combining neural nets with GBMs gives diversity that simple GBM ensembles lack.

---

## 4. Prudential Life Insurance Assessment — 2nd Place (Bogdan Zhurakovskyi, 2016)

**Original URL**: http://blog.kaggle.com/2016/03/14/prudential-life-insurance-assessment-winners-interview-2nd-place-bogdan-zhurakovskyi/ (BROKEN — redirects to Medium 404)

**Competition**: Predict a risk score (ordinal, 1-8) for life insurance applicants based on health, employment, and insurance history features. Metric: Quadratic Weighted Kappa (QWK). ~60K training rows with ~128 features (many anonymized).

**Task**: Ordinal classification/regression — predicting an ordered categorical response. The Quadratic Weighted Kappa metric penalizes predictions that are further from the true label more heavily.

**Approach (reconstructed from community knowledge)**:
The key challenge was optimizing for QWK, which is not a standard loss function. Top solutions treated this as a regression problem and then optimized the rounding thresholds to maximize QWK.

**Key Techniques**:
- **Regression + threshold optimization**: Trained models to predict a continuous value, then optimized 7 thresholds (to map continuous predictions to 8 ordinal classes) using scipy.optimize to maximize QWK directly. This was far more effective than training a classifier.
- **Feature engineering on anonymized features**: Interaction features between BMI-related columns, product feature combinations, medical history keyword counts, and employment info features.
- **Missing value patterns as features**: The pattern of which fields were missing carried significant signal about the risk category.
- **Ensemble of XGBoost + neural nets**: Multiple XGBoost models with different feature subsets and hyperparameters, combined with neural network predictions.
- **Custom CV strategy**: Stratified by the target variable to maintain class distribution across folds.
- **Feature selection**: Aggressive feature selection using mutual information and permutation importance — many of the 128 features were noise.

**How to Reuse**:
- **Threshold optimization is the single most important technique for ordinal/QWK competitions.** Train regression, optimize thresholds on validation set. This pattern recurs in every QWK competition.
- Missing value indicator features (binary: was this field missing?) are often among the most important features in insurance/medical data.
- For anonymized features, try all pairwise interactions and use feature importance to prune — you can't use domain knowledge, so brute force + selection works.

---

## 5. Winton Stock Market Challenge — 3rd Place (Mendrika Ramarlina, 2016)

**Original URL**: http://blog.kaggle.com/2016/02/12/winton-stock-market-challenge-winners-interview-3rd-place-mendrika-ramarlina/ (BROKEN — redirects to Medium 404)

**Competition**: Predict intraday and end-of-day stock returns. Given anonymized features and 5 days of historical returns (at daily and intraday resolution), predict returns for the next 2 days. Metric: Weighted MAE.

**Task**: Financial time-series prediction with heavily anonymized features. The signal-to-noise ratio was extremely low (as is typical for financial prediction tasks).

**Approach (reconstructed from community knowledge)**:
This competition was notable for having very low predictability — the winning solutions barely beat predicting zero (the mean). The 3rd place solution focused on robust modeling and regularization rather than complex feature engineering.

**Key Techniques**:
- **Conservative modeling**: Given the low signal, simple linear models and lightly-tuned GBMs performed similarly to complex approaches. Overfitting was the primary risk.
- **Separate models for intraday vs. daily returns**: The two prediction tasks had different characteristics — intraday returns had momentum patterns while daily returns were closer to random.
- **Lag features**: Historical returns at multiple horizons, volatility estimates (rolling standard deviation), momentum indicators.
- **Ridge/Lasso regression**: Linear models with strong regularization were competitive due to the noise level.
- **Minimal feature engineering**: With anonymized features, aggressive feature engineering risked overfitting to noise. Simple aggregations (mean, std of anonymized features) were safer.
- **Ensemble averaging**: Averaged multiple simple models to reduce variance — the primary benefit of ensembling in low-signal regimes.

**How to Reuse**:
- In financial prediction competitions, **start with predicting the mean** and measure how much any model improves over that baseline. If the improvement is tiny, focus on not overfitting rather than on complex features.
- Ridge regression is often surprisingly competitive for financial return prediction.
- Separate models for different prediction horizons (intraday vs. daily) or different asset types.
- In low-signal domains, model averaging reduces variance more than it introduces bias — ensemble aggressively.

---

## 6. Santa's Stolen Sleigh — 2nd Place (woshialex & weezy, 2016)

**Original URL**: http://blog.kaggle.com/2016/01/28/santas-stolen-sleigh-winners-interview-2nd-place-woshialex-weezy/ (BROKEN — redirects to Medium 404)

**Competition**: Optimization problem — deliver 100,000 gifts to locations around the world using 1,000 sleighs, each with a weight capacity of 1,000 kg (including the 10 kg sleigh weight). Minimize total "weighted reindeer weariness" (WRW = distance * cumulative weight remaining). Each trip starts and ends at the North Pole.

**Task**: A variant of the Capacitated Vehicle Routing Problem (CVRP) with a non-standard objective (weariness = distance * cumulative weight, not just total distance).

**Approach (reconstructed from community knowledge)**:
This was a pure optimization competition (no ML). Top solutions used combinations of greedy construction heuristics, local search, and simulated annealing.

**Key Techniques**:
- **Greedy initial solution**: Cluster gifts geographically (e.g., by longitude strips or k-means), then build routes within clusters using nearest-neighbor heuristics.
- **Weight-aware routing**: Because the objective penalizes carrying heavy loads over long distances, deliver heavy gifts early in each trip (when the sleigh is closest to those gifts) and light gifts later.
- **Local search operators**: 2-opt (reverse a segment of a route), or-opt (move a gift to another position or route), swap (exchange gifts between routes), relocate.
- **Simulated annealing**: Accept worse solutions with decreasing probability to escape local optima. Temperature schedule was critical.
- **Geographic decomposition**: Divide the globe into regions, optimize each region independently, then optimize cross-region boundaries.
- **Haversine distance**: The problem used great-circle distances on a sphere, not Euclidean distances, which matters for routes near the poles and across the date line.

**How to Reuse**:
- For optimization competitions: start with the simplest feasible solution (greedy), then iteratively improve with local search.
- Simulated annealing is the workhorse meta-heuristic for Kaggle optimization problems — learn it well.
- Weight/cost-aware routing: when the objective depends on cumulative load, the delivery order within a route matters enormously.
- Decompose large problems spatially, optimize sub-problems, then refine boundaries.

---

## 7. Predict Grant Applications — 1st Place (Jeremy Howard, 2011)

**Original URL**: http://blog.kaggle.com/2011/02/21/jeremy-howard-on-winning-the-predict-grant-applications-competition/ (BROKEN — redirects to Medium 404)

**Competition**: Predict whether university grant applications would be successful, based on researcher profiles, grant details, and institutional data. Binary classification, AUC metric. Run by the University of Melbourne.

**Task**: Binary classification on structured data with researcher-level and grant-level features. Notable as one of the early high-profile Kaggle competitions.

**Approach (reconstructed from community knowledge)**:
Jeremy Howard (later co-founder of fast.ai) won this competition and wrote extensively about his approach. His solution emphasized careful feature engineering and model ensembling — principles he later popularized in his fast.ai courses.

**Key Techniques**:
- **Systematic feature engineering**: Created features from researcher publication counts, citation metrics, h-index approximations, grant history (success rate), department-level success rates, and time-based features (grant year, researcher career stage).
- **Target encoding (before it had that name)**: Used historical success rates for categorical features like department, sponsor, and grant type. Applied smoothing to avoid overfitting on rare categories — essentially what we now call target encoding with regularization.
- **Random Forest + GBM ensemble**: Combined RandomForest (for its robustness and calibrated probabilities) with Gradient Boosted Trees (for its ability to learn complex interactions).
- **Careful cross-validation**: Used time-based splits to respect temporal ordering of grants — models shouldn't "see the future."
- **Feature importance analysis**: Iteratively added features, checked importance, and removed those that hurt CV score — a disciplined feature selection process.
- **Data cleaning**: Identified and handled duplicate researchers, merged records, and cleaned inconsistent categorical labels.

**How to Reuse**:
- **Target encoding with smoothing** is now a standard technique — Jeremy was one of the first to demonstrate it in competition. Use it for high-cardinality categoricals.
- Time-based CV splits are essential when the data has a temporal component — this applies to most real-world problems.
- Iterative feature engineering with CV feedback: add features, check CV, keep what helps. Simple but effective.
- Entity-level historical aggregations (researcher success rate, department success rate) generalize to any problem with repeated entities.

---

## 8. Stay Alert! The Ford Challenge — 1st Place (Team Inference, 2011)

**Original URL**: http://blog.kaggle.com/2011/03/25/inference-on-winning-the-ford-stay-alert-competition/ (BROKEN — redirects to Medium 404)

**Competition**: Detect driver alertness/drowsiness from physiological and vehicular sensor data. Binary classification (alert vs. not alert). Anonymized sensor features, time-series structure.

**Task**: Binary classification on sensor time-series data — determine if a driver is alert at each time step based on anonymized sensor readings. The data had strong temporal autocorrelation (alert/drowsy periods lasted many consecutive time steps).

**Approach (reconstructed from community knowledge)**:
The winning team "Inference" used an approach that exploited the temporal structure of the data — consecutive observations were highly correlated, and alertness state was persistent.

**Key Techniques**:
- **Temporal context features**: Lag features (sensor values at t-1, t-2, ..., t-k), rolling statistics (mean, std, min, max over windows), and difference features (sensor[t] - sensor[t-1]) to capture trends.
- **Signal processing features**: Rate of change, rolling variance (captures increasing noise as drowsiness onset), and frequency-domain features from short-time windows.
- **Hidden Markov Model intuition**: The alert/drowsy state is "sticky" — transitions are rare. Features that capture state duration and transition signals were valuable.
- **Ensemble of tree-based models**: GBM and Random Forest combined. Individual trees with different random seeds provided diversity.
- **Post-processing smoothing**: Applied smoothing to predicted probabilities to enforce temporal consistency — predictions shouldn't oscillate rapidly between alert and drowsy.
- **Subject-level features**: Normalized sensor readings per subject/session to account for individual differences in baseline sensor values.

**How to Reuse**:
- For any sensor/time-series classification: **lag features and rolling statistics** are the bread and butter. Start with windows of [5, 10, 30, 60] time steps.
- Difference features (delta from previous step) often capture transitions better than raw values.
- Post-processing smoothing of predictions is underused — if the target is temporally smooth, predictions should be too.
- Per-subject normalization is critical when sensor baselines vary across individuals/sessions.

---

## 9. Open Images 2019 Object Detection — 6th Place

**Original URL**: https://www.kaggle.com/c/open-images-2019-object-detection/discussion/110953 (page requires JS rendering — content not extractable via fetch)

**Competition**: Detect 500 object categories in images from the Open Images V5 dataset. Metric: mean Average Precision (mAP) at IoU=0.5. Large-scale detection with hierarchical labels and significant class imbalance.

**Task**: Large-scale object detection with 500 classes, many of which are rare. The challenge was handling the extreme class imbalance and the hierarchical label structure (a "person" and a "man" might both be valid labels for the same bounding box).

**Approach (reconstructed from competition context and known top solution patterns)**:
Top solutions in Open Images 2019 detection generally used Cascade R-CNN or HTC (Hybrid Task Cascade) backbones with large feature extractors (ResNeXt-101, SENet-154), multi-scale training/testing, and class-aware post-processing.

**Key Techniques**:
- **Strong backbones**: ResNeXt-101-32x8d, SENet-154, or ResNeSt with FPN (Feature Pyramid Networks) were standard at top of leaderboard.
- **Cascade R-CNN / HTC**: Multi-stage detection heads with increasing IoU thresholds — significantly better than vanilla Faster R-CNN for high-quality detection.
- **Multi-scale training and testing (TTA)**: Train with random image scales, test with multiple scales and merge detections. Typically 3-5 scales at test time.
- **Soft-NMS**: Replaced standard NMS with Soft-NMS for better handling of overlapping objects.
- **Class-aware thresholds**: Different score thresholds per class to handle class imbalance — rare classes needed lower thresholds.
- **Hierarchical label handling**: Open Images has a class hierarchy — needed to handle parent-child relationships (e.g., suppress "animal" detection when "dog" is more specific).
- **Large-scale data augmentation**: Mosaic, mixup, random crop/resize, color jitter.
- **Model ensemble**: Weighted Boxes Fusion (WBF) to combine detections from multiple models — superior to NMS-based ensemble methods.

**How to Reuse**:
- **Weighted Boxes Fusion** is now the standard for ensembling object detection models — prefer it over NMS-based ensemble.
- Multi-scale TTA is worth 1-3 mAP points in virtually every detection competition.
- Class-aware score thresholds are essential when classes have very different frequencies.
- For hierarchical labels, always handle parent-child suppression in post-processing.

---

## 10. Open Images 2019 Instance Segmentation — 7th Place

**Original URL**: https://www.kaggle.com/c/open-images-2019-instance-segmentation/discussion/110983 (page requires JS rendering — content not extractable via fetch)

**Competition**: Segment 300 object categories at instance level in Open Images V5 images. Metric: mean Average Precision (mAP) at various IoU thresholds. Requires both detection and pixel-level mask prediction.

**Task**: Instance segmentation — detect objects AND predict a binary mask for each detected instance, across 300 categories with extreme class imbalance.

**Approach (reconstructed from competition context and known top solution patterns)**:
Top solutions used Mask R-CNN variants (HTC, DetectoRS) with strong backbones, large-scale training data, and careful post-processing.

**Key Techniques**:
- **HTC (Hybrid Task Cascade)**: Multi-stage architecture that interleaves bounding box and mask prediction at each stage — better mask quality than vanilla Mask R-CNN.
- **Strong backbones**: ResNeXt-101-64x4d, HRNet, or DCN (Deformable Convolutional Networks) for better spatial resolution in masks.
- **COCO pre-training**: Pre-train on COCO (which has higher-quality mask annotations) then fine-tune on Open Images. Transfer learning from a cleaner dataset improved mask quality.
- **Multi-scale training/testing**: Same as detection — random scale training, multi-scale inference with mask merging.
- **Mask refinement**: Post-processing to clean up mask edges — CRF-based refinement or boundary-aware mask heads.
- **Class-specific augmentation**: Over-sample rare classes during training, use copy-paste augmentation for rare objects.
- **Ensemble via mask voting**: For multiple model predictions, use mask IoU-based voting to merge instance masks.

**How to Reuse**:
- Pre-training on a cleaner dataset (COCO) and fine-tuning on the target dataset (Open Images) is a powerful transfer learning strategy.
- Copy-paste augmentation is now standard for instance segmentation with class imbalance.
- HTC/Cascade architectures are still competitive — the multi-stage refinement paradigm persists in modern detection.

---

## 11. West Nile Virus Prediction — 2nd Place (diefimov, 2015)

**Source**: https://github.com/diefimov/west_nile_virus_2015 (fetched via GitHub API)

**Competition**: Predict the presence of West Nile Virus (WNV) in mosquito traps across Chicago, given trap locations, dates, mosquito species, weather data, and spray data. Binary classification, AUC metric.

**Task**: Geospatial + temporal binary classification — predict whether a mosquito trap will test positive for WNV on a given date, using weather conditions, geographic features, and mosquito species information.

**Approach (from code analysis)**:
The solution used an ensemble of Gradient Boosting Classifier (GBC via scikit-learn) and Regularized Greedy Forest (RGF, a C++ implementation), with extensive feature engineering on weather, temporal, and geospatial data. The code is implemented in both R (for data processing and ensembling) and Python (for model training). A year-based cross-validation strategy was used (train on odd years, predict even years, and vice versa).

**Key Techniques**:
- **Data augmentation**: Expanded training data by creating records for all (date, address, species) combinations, filling missing outcomes — this ensured the model saw the full combinatorial space of traps and dates.
- **Weather feature engineering**: Used data from two weather stations. Features included smoothed weather variables (Tmax, Tmin, Tavg, DewPoint, Precipitation, etc.) with a smoothing window (Smth02 suffix), capturing multi-day weather trends rather than single-day noise.
- **Geospatial features**: Latitude, Longitude, Block, Trap ID, AddressAccuracy. Corrected inconsistent trap IDs that mapped to multiple coordinates.
- **Species filtering**: Focused on the three mosquito species most associated with WNV transmission: CULEX PIPIENS/RESTUANS, CULEX PIPIENS, CULEX RESTUANS.
- **Temporal features**: Year, Month, Day, DayOfYear, WeekOfYear. Created trap count features (number of collections at same trap) and historical trap count features (TrapCountPrev, TrapCountPrevAge).
- **RGF (Regularized Greedy Forest)**: Used the RGF algorithm with specific parameters: L2 reg=0.2, structured L2 reg=0.07, log loss, 1400 max leaves, step size=0.7. RGF often outperforms standard GBM on structured data.
- **Ensemble with hand-tuned multipliers**: Combined GBC and RGF predictions with a 1:6 weighting (heavily favoring the multiplier-adjusted version). Applied year-month-specific multipliers (e.g., July 2012 x1.6, September 2012 x0.3) to capture known outbreak patterns that models couldn't learn from the limited training data.
- **Year-based CV**: Used odd years (2007, 2009, 2011, 2013) for training and even years (2008, 2010, 2012, 2014) for testing, matching the competition's temporal structure.

**How to Reuse**:
- **Smoothed weather features** (multi-day rolling averages) consistently outperform raw daily readings for disease/pest prediction — biological processes respond to accumulated conditions, not daily fluctuations.
- **Data augmentation via combinatorial expansion** (all trap x date x species combinations) can help when the observed data is sparse relative to the full combinatorial space.
- **RGF** is an underused algorithm that often beats GBM — consider it for tabular classification tasks.
- **Hand-tuned temporal multipliers** are a form of domain knowledge injection. When you know specific time periods had unusual patterns (outbreaks, anomalies), encoding that knowledge directly can beat learned features.
- Year-based (leave-one-year-out) CV is essential for epidemiological predictions where year-to-year patterns are the fundamental unit.

---

## 12. Allen AI Science Challenge — 8th Place (5vision, 2016)

**Source**: https://github.com/5vision/kaggle_allen (fetched via GitHub API)

**Competition**: Answer 8th-grade science multiple-choice questions (4 options each). Given a question and four answer options, predict the correct answer. Accuracy metric.

**Task**: Science question answering — an NLP/information retrieval task that requires matching questions to correct answers using external knowledge sources.

**Approach (from code and README)**:
Two complementary approaches were implemented:
1. **GloVe word embedding similarity** (~31.9% accuracy)
2. **Information retrieval from CK-12 + Wikipedia** (~35.4% accuracy)

**Key Techniques**:

*GloVe Approach:*
- Used pre-trained Wikipedia GloVe embeddings (300-dimensional).
- For each question, computed the average word vector (excluding stop words).
- For each answer option, computed the average word vector.
- Selected the answer whose vector had the highest cosine similarity to the question vector.
- Simple but effective baseline demonstrating that semantic similarity captures some question-answer relationships.

*IR Approach (better performing):*
- **Knowledge source construction**: Scraped topic keywords from CK-12 textbook pages (earth science, life science, physical science, biology, chemistry, physics). Used these keywords to retrieve relevant Wikipedia articles.
- **TF-IDF indexing**: Built TF-IDF representations for all retrieved Wikipedia documents.
- **Document retrieval**: For each question, retrieved the top-k most relevant documents using TF-IDF cosine similarity.
- **Answer scoring**: For each answer option, summed the TF-IDF scores of answer words found in the retrieved documents. Selected the answer with the highest aggregate score.
- **Parameters**: Used top 10 documents per question (docs_per_q=10).

**How to Reuse**:
- **External knowledge retrieval** is a powerful technique for QA tasks — scraping relevant textbooks/Wikipedia and building a retrieval index is a reusable pattern (precursor to modern RAG).
- TF-IDF is simple but competitive for document retrieval. For many Kaggle NLP tasks, a TF-IDF baseline should be your first model.
- The two-stage pipeline (retrieve relevant documents, then score answers against documents) is the same architecture used in modern retrieval-augmented generation (RAG) systems.
- Combining embedding-based similarity with term-matching (TF-IDF) captures complementary signals.

---

## 13. Stack Overflow Closed Questions — 10th Place (saffsd / Marco Lui, 2012)

**Source**: https://github.com/saffsd/kaggle-stackoverflow2012 (fetched via GitHub API)

**Competition**: Predict which Stack Overflow questions would be closed, and the closure reason (not a real question, not constructive, off topic, too localized, or remaining open). Multi-class classification.

**Task**: Text classification on Stack Overflow question data — predict closure status using question text, code blocks, tags, user history, and metadata.

**Approach (from code and README)**:
Built on the FastML blog post baseline but substantially rewritten. The entire pipeline was automated via Makefile with cross-validation implemented using shell tools and GNU parallel. Used Vowpal Wabbit (VW) as the learning framework.

**Key Techniques**:
- **Document segmentation**: Split questions into code and non-code sections. This is critical for Stack Overflow data where code blocks have fundamentally different characteristics than prose text.
- **Multi-level text features**:
  - **Document level**: Overall length, code-to-text ratio.
  - **Section level**: Number of code blocks, length of code sections.
  - **Sentence level**: Sentence count, average sentence length (from NLTK sentence tokenization of non-code text).
  - **Word level**: Bag-of-words features, word count, vocabulary richness.
- **Structural features**: Number of questions marks, exclamation marks — these capture question quality signals (e.g., excessive punctuation correlates with closure).
- **User-based features**: Extracted user metrics (reputation, history) as features — experienced users write higher-quality questions.
- **Vowpal Wabbit**: Used VW for fast, scalable learning. The VW format allows mixing different feature namespaces (text features, user features, metadata) with namespace-specific weighting.
- **Cross-validation via shell tools**: Implemented k-fold CV using GNU parallel and shell scripts — a fast, reproducible approach that predates modern CV libraries.
- **N-gram features**: Generated word and character n-grams from the non-code text for capturing common low-quality question patterns.

**How to Reuse**:
- **Segmenting code vs. non-code** in technical text is essential — treating them as one blob loses signal. This applies to any developer/technical content classification.
- **Multi-level feature extraction** (document -> section -> sentence -> word) captures patterns at different granularities. Higher-level features (sentence count, paragraph structure) often carry more signal for quality/moderation tasks than word-level BoW alone.
- **Vowpal Wabbit** remains a strong choice for large-scale text classification with mixed feature types — its namespace system is elegant for combining heterogeneous features.
- **Makefile-based ML pipelines** are surprisingly effective for reproducibility. Modern tools (DVC, MLflow) do the same thing but a Makefile still works for simple workflows.
- Structural/formatting features (punctuation patterns, code blocks, question marks) are powerful for content quality prediction — they generalize across platforms.

---

## Cross-Cutting Themes from Historical Solutions

### Timeless Techniques (2011-2019 solutions that still work today):

1. **Ensembling diverse model families** (Give Me Some Credit, Rossmann, Prudential) — combining GBMs with neural nets or linear models beats same-family ensembles.

2. **Target encoding with regularization** (Jeremy Howard, 2011) — predates the formal literature but is now in every Kaggle winner's toolkit.

3. **Temporal validation splits** (Rossmann, Grant Applications, West Nile Virus) — time-based CV for time-series data is non-negotiable.

4. **Threshold optimization for ordinal targets** (Prudential) — regression + optimized thresholds dominates direct classification for QWK metrics.

5. **Entity embeddings for categoricals** (Rossmann 3rd) — invented in 2015, now standard in tabular deep learning.

6. **Lag features and rolling statistics** (Ford, Winton, Rossmann) — the universal feature engineering toolkit for time-series.

7. **External knowledge retrieval** (Allen AI) — the precursor to RAG, still the right approach for knowledge-intensive tasks.

8. **Document segmentation by structure** (Stack Overflow) — separating code from prose in technical text dramatically improves classification.

9. **Smoothed weather/environmental features** (West Nile Virus) — biological processes respond to accumulated conditions, not instantaneous readings.

10. **Simulated annealing for optimization** (Santa's Sleigh) — still the go-to meta-heuristic for combinatorial optimization on Kaggle.
