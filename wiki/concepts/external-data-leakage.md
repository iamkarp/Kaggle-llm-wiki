---
title: "External Data Usage & Leakage Detection"
tags: [leakage, external-data, adversarial-validation, id-leakage, metadata, tabular, kaggle]
date: 2026-04-15
source_count: 5
status: active
---

## Summary

External data requires adversarial validation before inclusion — if AUC > 0.6 distinguishing external from train, the distributions don't match. ID-based leakage detection via single-feature AUC scan is the primary tool. File metadata leakage (EXIF, file size) is a recurring pitfall in image competitions. SMOTE before CV is the most common self-inflicted leakage.

## What It Is

A systematic checklist and toolkit for finding and preventing data leakage in Kaggle competitions — both from external data inclusion and from the training pipeline itself.

## Key Facts / Details

### Adversarial Validation for External Data

```python
def check_external_data_compatibility(train_df, external_df, feature_cols):
    """AUC > 0.6 = external data distribution mismatch."""
    train_adv = train_df[feature_cols].copy(); train_adv['is_external'] = 0
    ext_adv = external_df[feature_cols].copy(); ext_adv['is_external'] = 1
    combined = pd.concat([train_adv, ext_adv], ignore_index=True)
    
    model = lgb.LGBMClassifier(n_estimators=200, num_leaves=31)
    scores = cross_val_score(model, combined[feature_cols], combined['is_external'],
                             cv=5, scoring='roc_auc')
    return scores.mean()
    # < 0.55: compatible; > 0.65: different distribution, use with caution
```

### ID-Based Leakage Detection

**Single-feature AUC scan:**
```python
def scan_for_id_leakage(X, y, threshold=0.52):
    leaky_features = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for col in X.columns:
        try:
            col_vals = X[[col]].fillna(-999)
            aucs = [roc_auc_score(y.iloc[val_idx], col_vals.iloc[val_idx])
                    for train_idx, val_idx in skf.split(col_vals, y)]
            if np.mean(aucs) > threshold:
                leaky_features.append((col, np.mean(aucs)))
        except: pass
    
    return sorted(leaky_features, key=lambda x: -x[1])
```

**Red flags:**
- Any single feature with AUC > 0.52 when used alone → likely leakage
- Test data highly concentrated in one bin (>80%) → possible leak
- Feature appears to be ID, timestamp, or ordering information

### Extreme Bin Pattern (Featexp)

```python
# pip install featexp
from featexp import get_trend_stats

stats = get_trend_stats(data=train, target_col='target', data_test=test)
leaky_features = stats[stats['Trend_changes'] > 3]['Feature'].tolist()
```

**What to look for:**
- Features with extreme bin patterns (bin with 100% positive class)
- Sharp discontinuity in bin vs. target pattern (threshold artifact)
- Feature values in test not in training → out-of-distribution

### File Metadata Leakage (Image Competitions)

```python
from PIL import Image
from PIL.ExifTags import TAGS

def check_image_metadata_leakage(image_paths):
    for path in image_paths:
        img = Image.open(path)
        exif_data = img._getexif()
        if exif_data:
            for tag_id, value in exif_data.items():
                tag_name = TAGS.get(tag_id, tag_id)
                if isinstance(value, str) and any(
                    kw in value.lower() 
                    for kw in ['malignant', 'benign', 'class', 'label', 'positive']
                ):
                    print(f"LEAK: {path} — {tag_name}: {value}")
```

**File size check:**
```python
sizes = [os.path.getsize(p) for p in image_paths]
auc = roc_auc_score(labels, sizes)
if abs(auc - 0.5) > 0.02:
    print(f"WARNING: File size predicts label! AUC={auc:.4f}")
```

Recurring issue: PetFinder 2019 image metadata encoded adoption likelihood.

### Oracle Probing (Whitehill 2017)

If competition uses log-loss, test labels can be inferred by observing score changes from carefully crafted submissions. This is why top GMs trust CV over public LB — public LB can be gamed via oracle probing with 5 submissions/day.

Defense: Kaggle limits submissions. Private LB (full test set) is the true evaluation.

### Data Leakage Prevention Checklist

```
Before splitting:
[ ] Drop duplicate rows (train/test overlap = guaranteed leak)
[ ] Remove ID columns unless verified as noise
[ ] Remove timestamp/ordering columns (unless modeling temporal effects)
[ ] Check for row-level overlap between train/test

Feature engineering:
[ ] GroupBy on target: use nested CV (CatBoost Ordered TS)
[ ] Target encoding: nested CV or LOO
[ ] Rolling features: use only past data
[ ] External data: run adversarial validation first

Model training:
[ ] SMOTE only INSIDE CV pipeline (never before split)
[ ] Normalization: fit scaler on train only, transform both
[ ] For time-series: purged CV with embargo gap

Submission sanity:
[ ] Test predictions have reasonable distribution vs CV
[ ] Retrain on FULL train for final submission
```

### Approved External Data by Domain

| Domain | Source |
|---|---|
| Bird audio | xeno-canto.org |
| NLP | Wikipedia, Common Crawl |
| Images | ImageNet, OpenImages (pretrained weights) |
| Medical | NIH Open-i, TCIA |
| Drug discovery | ChEMBL, PubChem, ZINC |
| Time-series | FRED, Yahoo Finance (check rules) |

## When To Use It

- At the START of every competition — run the leakage checklist before any modeling
- Before adding external data — always run adversarial validation first
- When CV is suspiciously high — run ID leakage scan
- When working with image data — always check EXIF and file size

## Gotchas

- SMOTE before train/test split inflates CV by ~0.16 AUC (ArXiv 2412.07437)
- GroupBy aggregations using the target are the #1 leakage source in tabular
- Even pretrained model weights can be "model leakage" if training data overlaps competition test set
- Oracle probing is against Kaggle Terms of Service — understand it defensively, not offensively

## Sources

- [[../raw/kaggle/external-data-leakage-strategies.md]] — full reference with code
- [Featexp package](https://github.com/abhayspawar/featexp)
- [Whitehill & Movellan 2017 oracle probing](https://arxiv.org/abs/1712.01487)
- [SMOTE leakage arXiv 2412.07437](https://arxiv.org/html/2412.07437v1)
- [Adversarial validation tutorial](https://www.kaggle.com/code/carlmcbrideellis/what-is-adversarial-validation)

## Related

- [[concepts/validation-strategy]] — adversarial validation, CV-LB breakdown
- [[strategies/kaggle-meta-strategy]] — leakage detection in grandmaster playbook
- [[concepts/imbalanced-data]] — SMOTE placement (pipeline only)
- [[concepts/feature-selection-advanced]] — adversarial validation for feature selection
