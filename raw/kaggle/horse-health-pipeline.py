"""
Horse Health Prediction — end-to-end pipeline.
Metric: micro-averaged F1 (== accuracy for full multiclass coverage).
3 classes: lived / died / euthanized.

Follows Jason's Kaggle playbook:
- Stratified 10-fold CV (small dataset, imbalanced).
- Categorical missing -> own class.
- Numeric median impute + _was_missing flags.
- Lesion code decomposition (horse-colic convention: site/type/subtype/specific).
- Target encoding for hospital_number (OOF).
- LightGBM + XGBoost + CatBoost, each tuned for early stopping.
- Fourth-root weighted ensemble of OOF probabilities.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

SEED = 42
N_FOLDS = 10
HERE = Path(__file__).parent

# ------------------------- Load -------------------------
train = pd.read_csv(HERE / "train.csv", index_col=0)
test = pd.read_csv(HERE / "test.csv", index_col=0)

CLASSES = ["died", "euthanized", "lived"]  # alphabetical, canonical order
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

y = train["outcome"].map(CLASS_TO_IDX).values
test_ids = test["id"].values


# ------------------------- Feature engineering -------------------------
def decompose_lesion(code: int) -> dict:
    """Horse-colic lesion codes encode site/type/subtype/specific.
    Codes are typically 4 digits; 5-digit codes (e.g. 11300) use the first 2
    digits for site (11 = all sites). Zero means no lesion."""
    s = str(int(code))
    if code == 0:
        return {"site": 0, "type": 0, "subtype": 0, "specific": 0}
    if len(s) >= 5:  # 11xxx -> site=11
        site = int(s[:2])
        rest = s[2:]
    else:
        site = int(s[0])
        rest = s[1:]
    rest = rest.ljust(3, "0")
    return {
        "site": site,
        "type": int(rest[0]) if rest[0].isdigit() else 0,
        "subtype": int(rest[1]) if rest[1].isdigit() else 0,
        "specific": int(rest[2:]) if rest[2:].isdigit() else 0,
    }


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Decompose lesion_1 (primary lesion has the most information)
    decomposed = df["lesion_1"].apply(decompose_lesion).apply(pd.Series)
    decomposed.columns = [f"lesion_1_{c}" for c in decomposed.columns]
    df = pd.concat([df, decomposed], axis=1)

    # Lesion count
    df["has_lesion_2"] = (df["lesion_2"] != 0).astype(int)
    df["has_lesion_3"] = (df["lesion_3"] != 0).astype(int)
    df["n_lesions"] = (df["lesion_1"] != 0).astype(int) + df["has_lesion_2"] + df["has_lesion_3"]

    # Clinical ratios / interactions (domain-inspired)
    df["pulse_resp_ratio"] = df["pulse"] / (df["respiratory_rate"] + 1)
    df["pcv_tp_ratio"] = df["packed_cell_volume"] / (df["total_protein"] + 1)
    # Deviation from normal equine rectal temp (~37.8C)
    df["temp_dev"] = (df["rectal_temp"] - 37.8).abs()
    # Heart rate severity buckets (adult resting HR ~28-40)
    df["tachy_severity"] = (df["pulse"] - 40).clip(lower=0)

    return df


train_fe = build_features(train)
test_fe = build_features(test)

DROP_COLS = ["id", "outcome", "lesion_1", "lesion_2", "lesion_3"]
FEATURES = [c for c in train_fe.columns if c not in DROP_COLS]

# Identify cat vs numeric columns (handle pandas str/object dtypes)
from pandas.api.types import is_numeric_dtype
CAT_COLS = [c for c in FEATURES if not is_numeric_dtype(train_fe[c])]
NUM_COLS = [c for c in FEATURES if c not in CAT_COLS]

print(f"Train shape: {train_fe.shape}  Test shape: {test_fe.shape}")
print(f"Categorical cols ({len(CAT_COLS)}): {CAT_COLS}")
print(f"Numeric cols ({len(NUM_COLS)}): {NUM_COLS}")


# ------------------------- Impute -------------------------
def impute_inplace(tr: pd.DataFrame, te: pd.DataFrame, num_cols, cat_cols):
    """Categorical: missing -> '__NA__'. Numeric: median impute + _was_missing flag."""
    tr, te = tr.copy(), te.copy()
    for c in cat_cols:
        tr[c] = tr[c].fillna("__NA__").astype(str)
        te[c] = te[c].fillna("__NA__").astype(str)
    for c in num_cols:
        was_missing_tr = tr[c].isna().astype(int)
        was_missing_te = te[c].isna().astype(int)
        med = tr[c].median()
        tr[c] = tr[c].fillna(med)
        te[c] = te[c].fillna(med)
        if was_missing_tr.sum() + was_missing_te.sum() > 0:
            tr[f"{c}_was_missing"] = was_missing_tr
            te[f"{c}_was_missing"] = was_missing_te
    return tr, te


train_imp, test_imp = impute_inplace(train_fe[FEATURES], test_fe[FEATURES], NUM_COLS, CAT_COLS)
# Refresh column lists (some _was_missing cols added)
ALL_FEATS = [c for c in train_imp.columns]
NUM_COLS_FINAL = [c for c in ALL_FEATS if c not in CAT_COLS]
CAT_COLS_FINAL = CAT_COLS[:]

# Label-encode categoricals for LGB/XGB (CatBoost uses string directly)
train_le = train_imp.copy()
test_le = test_imp.copy()
cat_category_maps = {}
for c in CAT_COLS_FINAL:
    combined = pd.concat([train_imp[c], test_imp[c]], axis=0).astype(str)
    cat = combined.astype("category")
    codes = dict(zip(cat.cat.categories, range(len(cat.cat.categories))))
    cat_category_maps[c] = codes
    train_le[c] = train_imp[c].map(codes).astype("int32")
    test_le[c] = test_imp[c].map(codes).astype("int32")


# ------------------------- OOF target-encoding for hospital_number -------------------------
# hospital_number has ~239 unique in train, 124 overlap with test. Frequency + OOF mean target.
def add_oof_target_encoding(train_df, test_df, y, col, folds, n_classes=3, smoothing=3.0):
    """Adds per-class OOF target mean for a given column. Returns new train/test cols."""
    train_new = np.zeros((len(train_df), n_classes))
    test_new = np.zeros((len(test_df), n_classes))
    for tr_idx, va_idx in folds:
        sub = pd.DataFrame({col: train_df[col].iloc[tr_idx].values, "y": y[tr_idx]})
        global_probs = np.bincount(y[tr_idx], minlength=n_classes) / len(tr_idx)
        for k in range(n_classes):
            grp = sub.groupby(col)["y"].apply(lambda s: ((s == k).sum() + smoothing * global_probs[k]) / (len(s) + smoothing))
            train_new[va_idx, k] = train_df[col].iloc[va_idx].map(grp).fillna(global_probs[k]).values
    # Full-train encoding for test
    sub_all = pd.DataFrame({col: train_df[col].values, "y": y})
    global_probs = np.bincount(y, minlength=n_classes) / len(y)
    for k in range(n_classes):
        grp = sub_all.groupby(col)["y"].apply(lambda s: ((s == k).sum() + smoothing * global_probs[k]) / (len(s) + smoothing))
        test_new[:, k] = test_df[col].map(grp).fillna(global_probs[k]).values
    return train_new, test_new


skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
folds = list(skf.split(train_le, y))

# Target-encode hospital_number (numeric column, treat as categorical for encoding)
train_he, test_he = add_oof_target_encoding(train_le, test_le, y, "hospital_number", folds, smoothing=5.0)
for k, cname in enumerate(CLASSES):
    train_le[f"hosp_te_{cname}"] = train_he[:, k]
    test_le[f"hosp_te_{cname}"] = test_he[:, k]

# Frequency encoding for hospital_number
combined = pd.concat([train_le["hospital_number"], test_le["hospital_number"]])
freq = combined.value_counts().to_dict()
train_le["hosp_freq"] = train_le["hospital_number"].map(freq).astype(int)
test_le["hosp_freq"] = test_le["hospital_number"].map(freq).astype(int)

# Re-identify categorical indices for CatBoost (before label-encoding)
# We'll pass string-typed versions to CatBoost
train_cb = train_imp.copy()
test_cb = test_imp.copy()
for k, cname in enumerate(CLASSES):
    train_cb[f"hosp_te_{cname}"] = train_he[:, k]
    test_cb[f"hosp_te_{cname}"] = test_he[:, k]
train_cb["hosp_freq"] = train_le["hosp_freq"]
test_cb["hosp_freq"] = test_le["hosp_freq"]

FINAL_FEATS = list(train_le.columns)
CAT_IDX_FOR_CB = [FINAL_FEATS.index(c) for c in CAT_COLS_FINAL]

print(f"\nFinal feature count: {len(FINAL_FEATS)}")


# ------------------------- Train models -------------------------
def micro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="micro")


SEEDS = [42, 1337, 2024]  # seed-average for robustness on small dataset


def train_lgb(X, y, X_test, folds, cat_cols):
    oof = np.zeros((len(X), 3))
    test_pred = np.zeros((len(X_test), 3))
    for seed in SEEDS:
        params = dict(
            objective="multiclass", num_class=3, metric="multi_logloss",
            learning_rate=0.03, num_leaves=31, max_depth=-1,
            feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=3,
            min_child_samples=10, lambda_l1=0.1, lambda_l2=0.1,
            verbose=-1, seed=seed,
        )
        for tr_idx, va_idx in folds:
            Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
            ytr, yva = y[tr_idx], y[va_idx]
            dtr = lgb.Dataset(Xtr, ytr, categorical_feature=cat_cols)
            dva = lgb.Dataset(Xva, yva, categorical_feature=cat_cols, reference=dtr)
            model = lgb.train(
                params, dtr, num_boost_round=3000,
                valid_sets=[dva], callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
            )
            oof[va_idx] += model.predict(Xva, num_iteration=model.best_iteration) / len(SEEDS)
            test_pred += model.predict(X_test, num_iteration=model.best_iteration) / (len(folds) * len(SEEDS))
    score = micro_f1(y, oof.argmax(axis=1))
    print(f"LightGBM OOF micro-F1: {score:.5f}")
    return oof, test_pred, score


def train_xgb(X, y, X_test, folds):
    oof = np.zeros((len(X), 3))
    test_pred = np.zeros((len(X_test), 3))
    for seed in SEEDS:
        params = dict(
            objective="multi:softprob", num_class=3, eval_metric="mlogloss",
            learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8,
            min_child_weight=3, reg_alpha=0.1, reg_lambda=0.5, tree_method="hist",
            seed=seed, verbosity=0,
        )
        for tr_idx, va_idx in folds:
            Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
            ytr, yva = y[tr_idx], y[va_idx]
            dtr = xgb.DMatrix(Xtr, ytr)
            dva = xgb.DMatrix(Xva, yva)
            dte = xgb.DMatrix(X_test)
            model = xgb.train(
                params, dtr, num_boost_round=3000, evals=[(dva, "v")],
                early_stopping_rounds=100, verbose_eval=0,
            )
            oof[va_idx] += model.predict(dva) / len(SEEDS)
            test_pred += model.predict(dte) / (len(folds) * len(SEEDS))
    score = micro_f1(y, oof.argmax(axis=1))
    print(f"XGBoost OOF micro-F1: {score:.5f}")
    return oof, test_pred, score


def train_cat(X, y, X_test, folds, cat_idx):
    oof = np.zeros((len(X), 3))
    test_pred = np.zeros((len(X_test), 3))
    for seed in SEEDS:
        for tr_idx, va_idx in folds:
            Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
            ytr, yva = y[tr_idx], y[va_idx]
            model = CatBoostClassifier(
                iterations=3000, learning_rate=0.03, depth=6,
                l2_leaf_reg=5.0, bagging_temperature=0.5, random_strength=1.0,
                loss_function="MultiClass", eval_metric="TotalF1:average=Micro",
                random_seed=seed, verbose=0, early_stopping_rounds=100,
            )
            model.fit(Xtr, ytr, cat_features=cat_idx, eval_set=(Xva, yva), use_best_model=True)
            oof[va_idx] += model.predict_proba(Xva) / len(SEEDS)
            test_pred += model.predict_proba(X_test) / (len(folds) * len(SEEDS))
    score = micro_f1(y, oof.argmax(axis=1))
    print(f"CatBoost OOF micro-F1: {score:.5f}")
    return oof, test_pred, score


print("\n=== Training LightGBM ===")
oof_lgb, test_lgb, score_lgb = train_lgb(train_le, y, test_le, folds, CAT_COLS_FINAL)

print("\n=== Training XGBoost ===")
oof_xgb, test_xgb, score_xgb = train_xgb(train_le, y, test_le, folds)

print("\n=== Training CatBoost ===")
oof_cat, test_cat, score_cat = train_cat(train_cb, y, test_cb, folds, CAT_IDX_FOR_CB)


# ------------------------- Ensemble -------------------------
# Fourth-root weighting on OOF micro-F1 (scores near 0.72 — baseline at 0.5 to amplify).
# Skip grid-search optimization: with only 988 rows it overfits the OOF.
def weight_from_score(s, baseline=0.5):
    return max(s - baseline, 1e-6) ** 4


w = np.array([
    weight_from_score(score_lgb),
    weight_from_score(score_xgb),
    weight_from_score(score_cat),
], dtype=float)
w /= w.sum()
print(f"\nBlend weights -> LGB: {w[0]:.3f}  XGB: {w[1]:.3f}  CAT: {w[2]:.3f}")

# Arithmetic blend
oof_arith = w[0] * oof_lgb + w[1] * oof_xgb + w[2] * oof_cat
test_arith = w[0] * test_lgb + w[1] * test_xgb + w[2] * test_cat
score_arith = micro_f1(y, oof_arith.argmax(axis=1))

# Geometric blend (weighted log-probability mean) — often beats arithmetic for calibrated probas
EPS = 1e-9
oof_geo = np.exp(
    w[0] * np.log(oof_lgb + EPS) + w[1] * np.log(oof_xgb + EPS) + w[2] * np.log(oof_cat + EPS)
)
oof_geo /= oof_geo.sum(axis=1, keepdims=True)
test_geo = np.exp(
    w[0] * np.log(test_lgb + EPS) + w[1] * np.log(test_xgb + EPS) + w[2] * np.log(test_cat + EPS)
)
test_geo /= test_geo.sum(axis=1, keepdims=True)
score_geo = micro_f1(y, oof_geo.argmax(axis=1))

print(f"Arithmetic-blend OOF micro-F1: {score_arith:.5f}")
print(f"Geometric-blend  OOF micro-F1: {score_geo:.5f}")

if score_geo >= score_arith:
    oof_blend, test_blend, blend_score = oof_geo, test_geo, score_geo
    blend_kind = "geometric"
else:
    oof_blend, test_blend, blend_score = oof_arith, test_arith, score_arith
    blend_kind = "arithmetic"
print(f"Using {blend_kind} blend — OOF micro-F1: {blend_score:.5f}")

# ------------------------- Submission -------------------------
pred_idx = test_blend.argmax(axis=1)
pred_labels = np.array([IDX_TO_CLASS[i] for i in pred_idx])

sub = pd.DataFrame({"id": test_ids, "outcome": pred_labels})
score_tag = f"{blend_score:.5f}".replace("0.", "")
filename = f"submission_oof_microF1_0.{score_tag}.csv"
sub.to_csv(HERE / filename, index=False)
print(f"\nWrote {filename}")
print(sub["outcome"].value_counts())

# Per-model submissions too (for diagnostic diversity)
for name, pred in [("lgb", test_lgb), ("xgb", test_xgb), ("cat", test_cat)]:
    df = pd.DataFrame({"id": test_ids, "outcome": [IDX_TO_CLASS[i] for i in pred.argmax(axis=1)]})
    s = {"lgb": score_lgb, "xgb": score_xgb, "cat": score_cat}[name]
    df.to_csv(HERE / f"submission_{name}_oof_microF1_{s:.5f}.csv", index=False)

print(f"\nSUMMARY")
print(f"  LightGBM  OOF: {score_lgb:.5f}")
print(f"  XGBoost   OOF: {score_xgb:.5f}")
print(f"  CatBoost  OOF: {score_cat:.5f}")
print(f"  Blend     OOF: {blend_score:.5f}")
