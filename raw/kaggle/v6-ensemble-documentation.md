# March Mania 2026: v6 Ensemble Model (0.02210 LB)

**Score:** 0.02210 (Stage 1 Leaderboard)
**Date:** Feb 21, 2026
**Type:** Weighted Ensemble

## 🏗 Architecture

The v6 model is a weighted average of three distinct pipelines. This diversity is key to its performance.

```python
Final Prediction = (0.35 * v2.9) + (0.35 * v2.8) + (0.30 * v5)
```

### 1. v2.9 "Elite" (35% weight)
*   **Model:** XGBoost Classifier
*   **Hyperparameters:** `n_estimators=600`, `max_depth=9`, `learning_rate=0.01`
*   **Key Features:**
    *   Four Factors (Offense + Defense)
    *   Elo Ratings (Day-by-day)
    *   Momentum (Last 30 days win %)
    *   Strength of Schedule (SOS)
    *   Tournament Experience (Cumulative wins)
    *   Massey Ordinals (Median ranking)
    *   Pace / Pythagorean Wins

### 2. v2.8 "Advanced" (35% weight)
*   **Model:** XGBoost Classifier
*   **Hyperparameters:** `n_estimators=500`, `max_depth=8`, `learning_rate=0.015`
*   **Key Difference:** Slightly more aggressive learning rate and shallower trees than v2.9. Focuses more on raw efficiency diffs.

### 3. v5 "Hybrid" (30% weight)
*   **Model:** A meta-ensemble itself.
*   **Composition:**
    *   XGBoost (Depth 5, 6, 7, 8 variability)
    *   LightGBM
    *   Logistic Regression (Calibrated)
    *   Elo-only baseline
*   **Role:** Provides stability. The Logistic Regression component is especially strong for Women's tournament predictions (where raw seed/efficiency often dominates).

## 📁 Files Included

*   `v6_optimized_stage1.csv`: The exact file submitted to Kaggle.
*   `v2_8_advanced.py`: Source code for component 1.
*   `v2_9_elite.py`: Source code for component 2.
*   `v5_hybrid_pipeline.py`: Source code for component 3.
*   `v6_optimization.py`: The script that calculated the optimal weights (0.35/0.35/0.30).

## 🚀 How to Reproduce

1.  Ensure data is in `data/` folder.
2.  Run `python3 v2_8_advanced.py` → Generates `v2_8_advanced_stage1.csv`
3.  Run `python3 v2_9_elite.py` → Generates `v2_9_elite_stage1.csv`
4.  Run `python3 v5_hybrid_pipeline.py` → Generates `v5_hybrid_stage1.csv`
5.  Run `python3 v6_optimization.py` → Generates final `v6_optimized_stage1.csv`

## 🧠 Why It Works
This ensemble balances **high-variance, high-accuracy** tree models (v2.8/v2.9) with **low-variance, stable** linear/hybrid models (v5). This prevents overfitting to specific years while capturing complex non-linear matchups.
