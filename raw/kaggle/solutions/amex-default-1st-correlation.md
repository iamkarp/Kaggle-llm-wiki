# American Express Default Prediction — 1st Place Solution
**Author**: Correlation (solo) | **Year**: 2022 | **Votes**: 303
**GitHub**: Code released publicly

---

## Competition
Predict whether a credit card customer will default in the next 30 days. Binary classification, proprietary Amex metric (weighted combination of Gini + recall at 4% FPR). Each customer has multiple monthly credit card statements (time series of up to 13 statements per customer).

## Core Challenge: Sequential Credit Card Statement Data

Each customer has up to 13 monthly statements. Naive aggregation (mean/std/last across statements) loses temporal dynamics — the trajectory matters (e.g., rapidly increasing delinquency balance is more predictive than the current level alone).

## Model Architecture: Heavy Ensemble of LGB + NN

Heavy ensemble combining:
- **Multiple LightGBM models** — with extensive tabular feature engineering on aggregated statement data
- **RNN-based neural networks** — for sequential modeling of statement histories

Blend: LGB dominated with ~60-70% weight; RNN provided complementary sequential signal.

## RNN Architecture: pack_padded_sequence

Key technique: use PyTorch's `pack_padded_sequence` to efficiently handle variable-length customer histories (1–13 statements).

```python
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class CreditRNN(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=2):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
            bidirectional=False
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, lengths):
        # Pack variable-length sequences for efficient RNN computation
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        output, hidden = self.rnn(packed)
        # Use final hidden state
        out = self.head(hidden[-1])
        return out.squeeze(1)

# Data preparation
def collate_fn(batch):
    # Sort by sequence length (required for pack_padded_sequence)
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    sequences, targets = zip(*batch)
    lengths = [len(s) for s in sequences]
    padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    return padded, torch.tensor(lengths), torch.tensor(targets)
```

**Why pack_padded_sequence**:
- Handles variable-length histories without wasting computation on padding tokens
- PyTorch RNN processes only the valid steps per sequence
- Equivalent to masking but more memory-efficient
- Required for batched training when sequence lengths differ significantly

## LightGBM Feature Engineering

For the tabular component, extensive aggregation features across statement history:

### Statement-Level Aggregations
For each feature across all 13 statements:
- Last value (most recent statement)
- Last - first (change over history)
- Last - mean (deviation from average)
- Mean, std, min, max across all statements
- Mean/std of last 3 statements (recent trend)
- Count of non-null statements

### Temporal Difference Features
```python
# First-order differences between consecutive statements
for col in statement_features:
    df[f'{col}_diff1'] = df.groupby('customer_id')[col].diff(1)
    df[f'{col}_diff_last'] = df.groupby('customer_id')[col].diff(-1)  # last - prev
```

### Cross-Feature Ratios
Balance-to-limit ratios, payment-to-due ratios, etc. Domain knowledge: credit utilization rates are more predictive than raw balances.

## Validation
- 5-fold stratified CV by customer (not statement)
- **Critical**: group by customer_id — all statements of one customer go to same fold
- Using statement-level split would leak: model sees some months of a customer in train and predicts other months

## Key Takeaways
1. Sequential models (RNN with pack_padded_sequence) capture trajectory dynamics that aggregation features miss
2. Variable-length time series in batches: use pack_padded_sequence (not masking + padding alone)
3. Always group by entity (customer_id) for CV splits, not by row
4. Last - mean and last - first features capture trend direction without explicit lag features
5. Heavy LGB + RNN ensemble — trees dominate on tabular aggregations, RNN adds sequential signal
