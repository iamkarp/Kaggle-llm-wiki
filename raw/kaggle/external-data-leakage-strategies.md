# External Data Usage & Leakage Detection in Kaggle Competitions

Compiled from competition discussions, Grandmaster interviews, research papers. April 2026.

---

## External Data: Approved Usage Patterns

### BirdCLEF — xeno-canto as Standard External Source

Xeno-canto (xeno-canto.org) contains millions of crowd-sourced bird audio recordings. It has become the de-facto approved external data source for Kaggle bird audio competitions.

**Workflow:**
1. Download xeno-canto recordings for all target species
2. Filter by quality rating (A and B ratings only)
3. Use as training data augmentation
4. Pseudo-label ambiguous xeno-canto recordings with trained model

**Legal status:** Permitted in competition rules; BirdCLEF organizers explicitly list xeno-canto as allowed.

### Adversarial Validation as External Data Gate

Before using any external dataset, run adversarial validation:
```python
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import cross_val_score

def check_external_data_compatibility(train_df, external_df, feature_cols):
    """
    Returns: AUC score. If < 0.6, external data distribution matches train.
    """
    train_adv = train_df[feature_cols].copy()
    train_adv['is_external'] = 0
    ext_adv = external_df[feature_cols].copy()
    ext_adv['is_external'] = 1
    
    combined = pd.concat([train_adv, ext_adv], ignore_index=True)
    X = combined[feature_cols]
    y = combined['is_external']
    
    model = lgb.LGBMClassifier(n_estimators=200, num_leaves=31)
    scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
    return scores.mean()

# If AUC > 0.65: external data has different distribution → use with caution
# If AUC < 0.55: external data is compatible → safe to append
```

---

## Leakage Detection Playbook

### 1. ID-Based Leakage — Single-Feature AUC Scan

**Concept:** If a single feature (usually an ID or timestamp) can predict the target with AUC > 0.52, there's a leak.

```python
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

def scan_for_id_leakage(X, y, threshold=0.52):
    """
    Scan every column for suspiciously high AUC when used alone.
    """
    leaky_features = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for col in X.columns:
        try:
            col_vals = X[[col]].fillna(-999)
            aucs = []
            for train_idx, val_idx in skf.split(col_vals, y):
                auc = roc_auc_score(y.iloc[val_idx], col_vals.iloc[val_idx])
                aucs.append(auc)
            
            mean_auc = np.mean(aucs)
            if mean_auc > threshold:
                leaky_features.append((col, mean_auc))
        except:
            pass
    
    return sorted(leaky_features, key=lambda x: -x[1])

leaky = scan_for_id_leakage(X_train, y_train)
print(f"Potentially leaky features: {leaky[:10]}")
```

**Laurae Method (more rigorous):**
```python
def laurae_leakage_scan(X_train, y_train, X_test, n_bins=10):
    """
    Discretize each feature → compute IV (Information Value) vs target.
    High IV on test set but not train = potential leakage.
    """
    for col in X_train.columns:
        # Bin the feature
        bins = pd.qcut(X_train[col], q=n_bins, duplicates='drop')
        train_iv = compute_information_value(bins, y_train)
        
        test_bins = pd.cut(X_test[col], bins=bins.cat.categories)
        # Can't compute test IV (no labels), but check extreme bin patterns
        bin_counts = test_bins.value_counts(normalize=True)
        
        # Red flag: test data is highly concentrated in one bin
        if bin_counts.max() > 0.8:
            print(f"Warning: {col} has >80% test data in single bin — possible leakage")
```

### 2. Extreme Bin Pattern Detection (Featexp)

[Featexp](https://github.com/abhayspawar/featexp) provides automated leakage detection via bin pattern analysis.

```python
# pip install featexp
from featexp import get_trend_stats

# Compare train vs test feature distributions
stats = get_trend_stats(
    data=train,
    target_col='target',
    data_test=test,
    features_list=feature_cols
)

# Features with high "Trend Change" are potentially leaky
leaky_features = stats[stats['Trend_changes'] > 3]['Feature'].tolist()
```

**Red flags in bin patterns:**
- Feature values in test that don't appear in training → out-of-distribution
- Feature bin with 100% positive class rate in training → overfit to specific values
- Sharp discontinuity in bin vs. target pattern → threshold artifact

### 3. File Metadata Leakage (Image Competitions)

Competition organizers sometimes forget to remove image metadata.

**EXIF data check:**
```python
from PIL import Image
import piexif

def check_image_metadata_leakage(image_paths, target_labels=None):
    """Check for target information hidden in EXIF metadata."""
    leaky_images = []
    
    for path, label in zip(image_paths, target_labels or [None]*len(image_paths)):
        try:
            img = Image.open(path)
            exif_data = img._getexif()
            
            if exif_data:
                # Check for suspicious fields
                for tag_id, value in exif_data.items():
                    tag_name = TAGS.get(tag_id, tag_id)
                    
                    # Red flags: camera model, GPS, description, comment fields
                    if isinstance(value, str) and any(
                        indicator in value.lower() 
                        for indicator in ['malignant', 'benign', 'class', 'label', 'positive', 'negative']
                    ):
                        leaky_images.append((path, tag_name, value))
        except:
            pass
    
    return leaky_images
```

**File size leakage:**
```python
import os

def check_file_size_leakage(image_dir, labels_df):
    """
    File size correlates with target? That's a leak.
    """
    sizes = []
    for _, row in labels_df.iterrows():
        path = os.path.join(image_dir, row['filename'])
        if os.path.exists(path):
            sizes.append({'filename': row['filename'], 
                         'file_size': os.path.getsize(path),
                         'label': row['label']})
    
    size_df = pd.DataFrame(sizes)
    
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(size_df['label'], size_df['file_size'])
    if abs(auc - 0.5) > 0.02:
        print(f"WARNING: File size predicts label with AUC={auc:.4f}!")
    
    return size_df
```

### 4. Oracle Probing (Whitehill 2017)

**Paper:** Whitehill & Movellan (2017) — "Approximately Correct: When Automated Grading Meets the Real World"

**Method:** If competition uses log-loss, you can determine whether a test sample belongs to class 0 or 1 by submitting carefully crafted predictions:

```python
# Oracle probing for binary classification
# NOT recommended (violates Kaggle terms of service, but important to understand)

# To determine label of sample i:
# Submit: p(i) = 0.99 (guess positive) → observe score change
# If score improves, sample i is positive
# If score worsens, sample i is negative

# Defense: Kaggle limits submissions (max 5/day prevents systematic probing)
# Kaggle uses private LB for final scoring (not probed)
```

**Competition implication:** Public LB can be gamed via oracle probing. Private LB (with different distribution) is the true test. This is why Chris Deotte and GMs trust CV > public LB.

### 5. PetFinder Cheating Scandal (Historical)

**What happened:** In PetFinder 2019, top-scoring solutions used image metadata (ELA analysis, file size) that encoded adoption likelihood. Some participants used this knowingly.

**Lessons learned:**
1. Always check file metadata before competition ends
2. If you find a leak, report it in the forum (most GMs recommend this)
3. Competition organizers now strip metadata more carefully
4. Leakage that's too obvious often gets the competition cancelled/rescored

### 6. Data Leakage Prevention Checklist

```
Before splitting:
[ ] Drop duplicate rows (same row in train and test = guaranteed leak)
[ ] Remove ID columns unless you've verified they're noise
[ ] Remove timestamp/ordering columns unless modeling temporal effects
[ ] Check for row-level overlap between train/test

Feature engineering:
[ ] GroupBy aggregations: never include test data in aggregation (do train/test separately)
[ ] Target encoding: use nested CV or CatBoost Ordered TS
[ ] Rolling features: use only past data (no future lookahead)
[ ] External data: run adversarial validation before including

Model training:
[ ] SMOTE only inside CV pipeline (never before split)
[ ] Normalization: fit scaler on train only, transform both
[ ] For time-series: purged CV with embargo gap

Submission:
[ ] Final test predictions: retrain on FULL train (never hold-out)
[ ] Check that test predictions have reasonable distribution vs CV predictions
```

---

## Approved External Data Sources by Domain

| Domain | Source | Notes |
|---|---|---|
| Audio classification | xeno-canto.org | Bird audio; must cite usage |
| NLP | Wikipedia, Common Crawl | Universal approval |
| Image classification | ImageNet, OpenImages | Pretrained weights always allowed |
| Medical imaging | NIH Open-i, TCIA | Check competition rules |
| Tabular/finance | FRED, Yahoo Finance | Case-by-case; check rules |
| Drug discovery | ChEMBL, PubChem, ZINC | Standard for molecular tasks |
| Text NLP | Pile, C4, Dolma | For pretraining auxiliary models |

---

Sources:
- Laurae leakage detection: https://github.com/Laurae2/wRobustness
- Featexp package: https://github.com/abhayspawar/featexp
- Whitehill & Movellan 2017 oracle probing: https://arxiv.org/abs/1712.01487
- PetFinder scandal discussion: https://www.kaggle.com/competitions/petfinder-adoption-prediction/discussion
- Adversarial validation tutorial: https://www.kaggle.com/code/carlmcbrideellis/what-is-adversarial-validation
