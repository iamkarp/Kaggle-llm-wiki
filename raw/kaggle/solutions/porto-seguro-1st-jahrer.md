# Porto Seguro Safe Driver Prediction — 1st Place Solution
**Author**: Michael Jahrer | **Votes**: 1099 | **Prize**: ~$25K

---

## Competition
Predict whether a Porto Seguro auto insurance policyholder will file a claim in the next year. Binary classification, Normalized Gini coefficient metric. ~600K train rows, 57 features.

## Core Approach: Denoising Autoencoders on Tabular Data

The winning solution used **denoising autoencoders (DAE)** to learn dense representations from tabular features, then trained supervised models on top. This was novel for structured/tabular data at the time — DAEs had been used primarily in computer vision and NLP.

### Why Autoencoders Work on Tabular Data
- Insurance features are noisy and have many missing values
- DAE forces the network to learn robust feature representations by reconstructing inputs from corrupted versions
- The learned embeddings capture non-linear relationships that tree models miss
- Particularly useful when signal-to-noise ratio is low

## Swap Noise > Gaussian Noise

**Key insight**: For tabular DAE, swapping feature values between training samples is a better corruption strategy than adding Gaussian noise.

**Swap noise**: With probability `p_swap`, replace a feature value in sample `i` with the value of that feature from a random other sample `j`. This preserves the marginal distribution of each feature while destroying the correlations between features.

**Why it works better**:
- Gaussian noise can push values out of the natural data distribution (e.g., negative values for counts)
- Swap noise always produces values that are "realistic" — they came from the actual data
- Swap noise is distribution-agnostic — works for categorical, ordinal, and continuous without modification
- Gaussian noise requires calibrating noise variance per feature; swap noise just needs one `p_swap` parameter (~0.1)

```python
def apply_swap_noise(X, p_swap=0.1):
    """Replace each value with probability p_swap with a value from a random row."""
    X_noisy = X.copy()
    n, m = X.shape
    for j in range(m):
        mask = np.random.rand(n) < p_swap
        random_rows = np.random.randint(0, n, size=mask.sum())
        X_noisy[mask, j] = X[random_rows, j]
    return X_noisy
```

## RankGauss Normalization

Before feeding features into the autoencoder/NN, apply **RankGauss** normalization:

1. Rank all values in each feature column (handle ties with average rank)
2. Map ranks to the [0, 1] interval
3. Apply the Gaussian inverse CDF (probit function) to map to a standard normal distribution

```python
from scipy.special import ndtri  # inverse normal CDF

def rank_gauss(x):
    n = len(x)
    ranks = np.argsort(np.argsort(x)) + 1  # 1-indexed ranks
    # Map to (0, 1) avoiding endpoints
    uniform = ranks / (n + 1)
    return ndtri(uniform)  # probit transform
```

**Why RankGauss**:
- Standard min-max or z-score normalization is sensitive to outliers
- RankGauss produces a perfectly Gaussian distribution regardless of input distribution
- Neural networks train better when inputs are approximately normal
- Robust: outliers get pushed to ±3σ range, not infinity

## Model Architecture

### DAE Architecture
```
Input (corrupt with swap noise p=0.1)
→ Dense(512, ReLU) → BatchNorm → Dropout(0.3)
→ Dense(256, ReLU) → BatchNorm → Dropout(0.3)
→ Dense(128, ReLU) → BatchNorm  [bottleneck = learned representation]
→ Dense(256, ReLU) → BatchNorm
→ Dense(512, ReLU) → BatchNorm
→ Dense(n_features) [reconstruction head]
```

Trained to minimize MSE reconstruction loss on clean inputs from corrupted inputs. After training, use the bottleneck layer (128-dim) as learned features.

### Final Supervised Models (6-model blend)
1. **LightGBM** trained on raw + RankGauss features
2. **NN #1–5**: 5 neural networks with different random seeds, trained on bottleneck embeddings + raw features, with direct classification head

**Ensemble**: Simple average of all 6 predictions.

**Key finding**: Simple averaging beat nonlinear stacking. Stacking introduced overfitting; the models were already well-calibrated and diverse enough.

## Validation
- Stratified 5-fold CV, stratified on target
- Normalized Gini = 2 × AUC − 1 (monotone transform of AUC)
- ~0.291 Gini LB (private)

## Missing Value Handling
- Categorical: NaN as own category (-1 or "__MISSING__")
- Numeric: median imputation + `_was_missing` indicator
- Then apply RankGauss to all (including indicator columns)

## Key Takeaways
1. DAE pre-training adds meaningful signal on noisy tabular data
2. Swap noise is the right corruption strategy for tabular (not Gaussian)
3. RankGauss normalization > standard scaling for NN inputs
4. Simple model averaging often beats learned stacking
5. 5-seed NN averaging reduces variance significantly
