---
title: "LLM & Transformer Fine-Tuning for Kaggle NLP"
tags: [nlp, llm, deberta, transformer, fine-tuning, lora, awp, llrd, distillation, kaggle, 2024, 2025]
date: 2026-04-15
source_count: 2
status: active
---

## What It Is

Techniques for fine-tuning large language models and transformers in Kaggle competitions. In 2022–2023, DeBERTa-v3-large was the dominant backbone; in 2024, decoder-only LLMs (Llama 3, Gemma 2, Qwen 2) began winning generation/preference tasks. This page covers the tricks that separate top-10 from top-100 in NLP competitions.

## Architecture Selection (2022–2025)

| Year | Dominant Backbone | Task Type |
|------|---|---|
| 2022–2023 | DeBERTa-v3-large | Classification, regression, NLI |
| 2024 | DeBERTa-v3-large + decoder LLMs | Classification + preference/generation |
| 2024+ | Llama 3, Gemma 2, Qwen 2, Mistral | Preference modeling, generation-adjacent |

**DeBERTa-v3-large still competitive** for classification tasks in 2024. Use for standard text classification when GPU budget is tight.

**Decoder LLMs:** Win when task requires generation reasoning, preference modeling, or instruction following.

## Fine-Tuning Tricks (The Core Toolkit)

### 1. Layer-Wise Learning Rate Decay (LLRD)
Apply different learning rates to each layer: highest LR at the classifier head, lowest at early embedding layers.

```python
def get_param_groups_with_llrd(model, base_lr=3.5e-6, decay=0.9):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = []
    
    # Get all named parameters by layer depth
    layers = [model.embeddings] + list(model.encoder.layer)
    layers.reverse()  # Start from top (classifier side)
    
    for i, layer in enumerate(layers):
        layer_lr = base_lr * (decay ** i)
        layer_params = [p for n, p in layer.named_parameters()]
        
        optimizer_grouped_parameters.extend([
            {"params": [p for n, p in layer.named_parameters() 
                       if not any(nd in n for nd in no_decay)],
             "lr": layer_lr, "weight_decay": 0.01},
            {"params": [p for n, p in layer.named_parameters() 
                       if any(nd in n for nd in no_decay)],
             "lr": layer_lr, "weight_decay": 0.0},
        ])
    return optimizer_grouped_parameters

optimizer = AdamW(get_param_groups_with_llrd(model), lr=3.5e-6)
```

**Why:** Prevents catastrophic forgetting. Early layers encode general language representations; they shouldn't change much. Only the top layers need aggressive adaptation to the new task.

**Typical values:** top_lr=2e-5 to 5e-6, decay=0.85–0.95 per layer.

### 2. Adversarial Weight Perturbation (AWP)
Add small perturbations to model weights during training to find flatter loss minima. Double perturbation: perturb both embeddings AND weights.

```python
class AWP:
    def __init__(self, model, adv_param="weight", adv_lr=0.1, adv_eps=1e-4):
        self.model = model
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}
        self.backup_eps = {}
    
    def attack(self, emb_name="word_embeddings"):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.adv_param in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.adv_lr * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self._project(name, param.data)
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
    
    def _project(self, param_name, param_data):
        r = param_data - self.backup[param_name]
        r = torch.clamp(r, -self.adv_eps, self.adv_eps)
        return self.backup[param_name] + r
```

**Usage pattern:**
```python
awp = AWP(model)
for batch in train_loader:
    loss = model(batch)
    loss.backward()
    awp.attack()           # add perturbation
    loss_adv = model(batch)
    loss_adv.backward()
    awp.restore()          # restore original weights
    optimizer.step()
    optimizer.zero_grad()
```

**Impact:** Near-universal in winning NLP solutions from 2022 onward. Adds ~2× training time. Worth it for competitions with tight margins.

**Reference:** https://www.kaggle.com/code/itsuki9180/introducing-adversarial-weight-perturbation-awp

### 3. Multisample Dropout
Create multiple dropout masks per forward pass, average the losses.

```python
class MultiSampleDropout(nn.Module):
    def __init__(self, base_model, num_samples=5, dropout_rate=0.1):
        super().__init__()
        self.base_model = base_model
        self.dropouts = nn.ModuleList([
            nn.Dropout(dropout_rate) for _ in range(num_samples)
        ])
    
    def forward(self, x):
        return torch.mean(torch.stack([
            self.classifier(dropout(x))
            for dropout in self.dropouts
        ], dim=0), dim=0)
```

**Benefits:** Accelerates convergence (fewer epochs needed), improves generalization. Low overhead (most compute in attention layers, not dropout).

**Paper:** "Multi-Sample Dropout for Accelerated Training and Better Generalization" (Inoue, 2019)

### 4. Two-Stage Training (Domain Adaptation → Fine-tune)
1. **Stage 1:** Further MLM pre-training on competition's unlabeled text (15–20% masking, 1–3 epochs).
2. **Stage 2:** Fine-tune on labeled competition data.

```python
# Stage 1: Domain adaptation
mlm_trainer = Trainer(
    model=AutoModelForMaskedLM.from_pretrained("microsoft/deberta-v3-large"),
    args=TrainingArguments(num_train_epochs=2, ...),
    train_dataset=unlabeled_competition_text_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm_probability=0.15)
)
mlm_trainer.train()
mlm_trainer.save_model("domain_adapted_deberta")

# Stage 2: Task fine-tuning
clf_trainer = Trainer(
    model=AutoModelForSequenceClassification.from_pretrained("domain_adapted_deberta"),
    ...
)
```

**AES 2.0 2024:** Jumped from rank 619 (public LB) to **1st (private LB)** using two-stage training + distribution shift analysis.

### 5. Custom Pooling Strategies
Don't rely only on CLS token. Combine multiple methods and ensemble models that differ only in pooling:

```python
class CustomPooling(nn.Module):
    def forward(self, outputs, attention_mask):
        hidden = outputs.last_hidden_state
        
        # Mean pooling (most reliable)
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
        mean_pool = (hidden * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)
        
        # Max pooling
        max_pool = (hidden * mask_expanded - (1 - mask_expanded) * 1e9).max(1).values
        
        # CLS token
        cls_pool = hidden[:, 0, :]
        
        # Concatenate all
        return torch.cat([mean_pool, max_pool, cls_pool], dim=-1)
```

**Diversity:** Models differing only in pooling strategy add genuine, cheap ensemble diversity.

### 6. Gradient Checkpointing
Saves 10× memory, adds ~10–20% compute.

```python
model.gradient_checkpointing_enable()
# Note: incompatible with naive dropout — use deterministic dropout or careful implementation
```

## LoRA / QLoRA for Large Decoder LLMs

When fine-tuning models >7B parameters:

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# QLoRA: 4-bit quantization + LoRA adapters
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-70B", 
                                              quantization_config=bnb_config)

lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, 
                         target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, lora_config)
```

## Knowledge Distillation Pattern (2024 Winning Pattern)

1. Fine-tune large models (70B) on 8× A100-80GB during training
2. Generate soft logit distributions from the large model
3. Train smaller model (7B/9B) to match those logit distributions
4. Only the small model is used at inference (fits in competition GPU budget)

```python
# Distillation loss = KL divergence between teacher and student logits
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, labels, temperature=4.0, alpha=0.5):
    soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
    soft_pred = F.log_softmax(student_logits / temperature, dim=-1)
    kl_loss = F.kl_div(soft_pred, soft_targets, reduction='batchmean') * (temperature ** 2)
    hard_loss = F.cross_entropy(student_logits, labels)
    return alpha * kl_loss + (1 - alpha) * hard_loss
```

**LMSYS 1st place:** Distilled Llama3-70B + Qwen2-72B into Gemma2-9B. Final submission used only 8-bit quantized Gemma2-9B.

## Inference Efficiency for Kaggle Submission

Kaggle GPU: single T4/P100 (16GB), 9-hour time limit.

| Technique | Memory Saving | Speed Gain |
|---|---|---|
| 4-bit quantization (NF4) | 75% vs FP16 | Minimal loss |
| Flash Attention 2 | 40% | 2–4× faster attention |
| torch.compile | 0 | 20–30% faster inference |
| FP16/BF16 inference | 50% vs FP32 | 2× faster |
| Dynamic padding | Varies | 20–40% batch speedup |

## NLP Ensemble Strategies

**Diversity sources (most to least impactful):**
1. Different backbone architectures (DeBERTa-large vs DeBERTa-base vs RoBERTa-large)
2. Different context lengths (512 vs 1024 vs 1536 tokens)
3. Different pooling methods
4. Different training data subsets / augmentations
5. Different random seeds

**Weight optimization:** Nelder-Mead optimization against OOF score, or hill-climbing greedy search.

**Rank averaging:** More robust than raw probability averaging when models are poorly calibrated.

## In Jason's Work
Not applied (Jason's competitions are tabular/sports/vision). Directly applicable if Jason enters text classification, essay scoring, or NLP-adjacent competitions.

## Sources
- [[../../raw/kaggle/timeseries-nlp-techniques.md]] — comprehensive NLP techniques reference
- [AWP Kaggle notebook](https://www.kaggle.com/code/itsuki9180/introducing-adversarial-weight-perturbation-awp)
- [LMSYS 1st place GitHub](https://github.com/tascj/kaggle-lmsys-chatbot-arena)
- [HuggingFace QLoRA blog](https://huggingface.co/blog/4bit-transformers-bitsandbytes)
- [[../../raw/kaggle/solutions/missing-batch-nlp-reasoning.md]] — ARC Prize 2024 (262 votes), ARC Prize 2025 (179 votes), Deep Past Translation (226 votes, 6th: 15-model ensemble + diverse data), Text Normalization English/Russian, MSK Cancer Treatment, ASAP Essay Scoring, Detecting Insults

## Related
- [[../concepts/knowledge-distillation]] — distillation patterns (tabular context)
- [[../concepts/pseudo-labeling]] — semi-supervised learning that applies to NLP too
- [[../concepts/ensembling-strategies]] — NLP ensembling follows same OOF stacking principles
