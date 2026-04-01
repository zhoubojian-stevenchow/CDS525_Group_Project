# FinBERT Financial Sentiment Analysis

**CDS 525 — Deep Learning Group Project**

A systematic study of FinBERT fine-tuning for three-class financial sentiment classification, comparing Weighted Cross-Entropy and Enhanced Focal Loss across 32 hyperparameter configurations.

## Key Results

| Loss Function | Batch Size | Learning Rate | Macro F1 (%) | Test Acc (%) | Balanced Acc (%) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Weighted CE | 32 | 2e-5 | **89.92** | 88.79 | 90.26 |
| Weighted CE | 128 | 2e-5 | 89.92 | 88.90 | 90.17 |
| Focal Loss | 64 | 2e-5 | 89.87 | 88.89 | 90.27 |
| Focal Loss | 128 | 2e-5 | 89.86 | **89.06** | 90.02 |
| Focal Loss | 32 | 2e-5 | 89.72 | 88.68 | 89.99 |

**Headline findings:**
- Best macro F1: **89.92%** (Weighted CE, BS=32, LR=2e-5)
- Paired t-test shows **no significant difference** between loss functions (p=0.789)
- All misclassification errors are Negative↔Positive swaps; **Neutral is never confused**
- 6 out of 100 test samples are persistently misclassified across all configurations — these are annotation-ambiguous headlines

## Project Overview

### Problem
Classify financial text into three sentiment categories (Negative, Neutral, Positive) using a domain-specific pre-trained Transformer model.

### Approach
- **Model:** [ProsusAI/FinBERT](https://huggingface.co/ProsusAI/finbert) — BERT-base further pre-trained on financial text
- **Data:** Three-source hybrid dataset with source tagging:
  - Twitter Financial News `[SOCIAL]`
  - NIFTY Indian Market Headlines `[NEWS_IN]`
  - FinancialPhraseBank 75% agreement `[FORMAL]`
- **Grid Search:** 2 loss functions × 4 batch sizes × 4 learning rates = **32 configurations**
- **Metrics:** Training loss, training accuracy, test accuracy, test loss, macro F1, balanced accuracy — tracked per epoch across all 32 configs

### Grid Search Space

| Hyperparameter | Values |
|---|---|
| Loss Function | Weighted Cross-Entropy, Enhanced Focal Loss (γ=2.0, α=[0.4,0.2,0.4]) |
| Batch Size | 16, 32, 64, 128 |
| Learning Rate | 1e-6, 2e-5, 5e-5, 1e-4 |
| Optimizer | AdamW (weight_decay=0.01, warmup=10%) |
| Epochs | 15 max, early stopping on F1 (patience=15) |

## Repository Structure

```
├── README.md
├── CDS525_Unified_GridSearch_FINAL.ipynb   # Single notebook: all 32 configs + analysis
├── CDS525_FinBERT_Report_v3.docx           # Written report (2600+ words, 18 figures)
├── notebooks/                               # Original per-batch-size notebooks
│   ├── CDS525_Exp1_WeightedCE_GridSearch_batchsize16_v2.ipynb
│   ├── CDS525_Exp1_WeightedCE_GridSearch_batchsize32_v2.ipynb
│   ├── CDS525_Exp1_WeightedCE_GridSearch_batchsize64_v2.ipynb
│   ├── CDS525_Exp1_WeightedCE_GridSearch_batchsize128_v2.ipynb
│   ├── CDS525_Exp2_FocalLoss_GridSearch_batchsize16_v2.ipynb
│   ├── CDS525_Exp2_FocalLoss_GridSearch_batchsize32_v2.ipynb
│   ├── CDS525_Exp2_FocalLoss_GridSearch_batchsize64_v2.ipynb
│   └── CDS525_Exp2_FocalLoss_GridSearch_batchsize128_v2.ipynb
├── results/
│   ├── exp1/                                # Weighted CE results
│   │   ├── exp1_history_BS{16,32,64,128}_LR{...}.csv
│   │   └── exp1_qualitative_BS{16,32,64,128}_LR{...}.csv
│   └── exp2/                                # Focal Loss results
│       ├── exp2_history_BS{16,32,64,128}_LR{...}.csv
│       └── exp2_qualitative_BS{16,32,64,128}_LR{...}.csv
└── figures/                                 # Generated figures for report
```

## Analysis Highlights

### Learning Rate is the Dominant Factor

LR=2e-5 outperforms all other rates regardless of loss function or batch size. LR=1e-4 causes **catastrophic forgetting** — at BS=16, training accuracy crashes from 80.7% to 3.9% between epochs 1–2.

| Learning Rate | Mean F1 (%) | Std |
|:---:|:---:|:---:|
| 1e-6 | 88.59 | ±0.83 |
| **2e-5** | **89.71** | **±0.22** |
| 5e-5 | 88.83 | ±0.77 |
| 1e-4 | 86.57 | ±2.47 |

### Loss Functions Are Statistically Equivalent

Paired t-test across all 16 matched hyperparameter pairs:
- Mean difference (WCE − FL): **0.060%**
- t = 0.272, **p = 0.789** (not significant)
- WCE wins 9/16 pairs, FL wins 7/16

### Test Loss Diverges from F1

Test loss is minimized at **epoch 2**, but F1 continues improving until epochs 11–15. This validates using F1 rather than loss for early stopping in sentiment classification.

### Confusion Matrix Pattern

All errors are **Negative↔Positive swaps**. Neutral class achieves 100% accuracy on the first 100 test predictions — the model develops strong neutral-detection capability.

### Persistent Errors Are Annotation Ambiguity

6 samples are always wrong across all 8 best-LR configurations:
- *"Trade war between US and China helping India boost exports"* — labeled Negative, but positive framing for India
- *"Xiaomi IPO hopeful set to blow past 2017 revenue target"* — labeled Positive, but "blow past" triggers negative detection

## How to Run

### Requirements
- Python 3.10+
- Google Colab with A100 GPU (recommended) or any CUDA GPU
- ~8GB VRAM for BS=128, ~4GB for BS=16

### Quick Start

1. Upload `CDS525_Unified_GridSearch_FINAL.ipynb` to Google Colab
2. Place dataset files in Google Drive under `CDS525_Project/`:
   - `sent_train.csv` (Twitter financial sentiment)
   - `News_sentiment_Jan2017_to_Apr2021.csv` (NIFTY news)
   - `Sentences_75Agree.txt` (FinancialPhraseBank)
3. Run all cells — the notebook executes both experiments sequentially (16 configs each)
4. Results are saved to `outputs_compliant_exp1_test/` on Google Drive

### Runtime Estimate
- ~15 min per config on A100 → ~8 hours for full 32-config grid search
- Each experiment (16 configs) can be run independently

## Technical Details

### Model
- **ProsusAI/FinBERT**: BERT-base (110M params) pre-trained on financial corpus
- Classification head: linear layer on [CLS] token → 3 classes
- Full fine-tuning (all layers trainable)

### Enhanced Focal Loss

```python
FL(p_t) = -α_t × (1 - p_t)^γ × CE(p_t, y_smoothed)
# γ = 2.0, α = [0.4, 0.2, 0.4], label_smoothing = 0.1
```

### Training Pipeline
- Tokenization: WordPiece, max_length=128
- Split: 80/20 stratified (random_state=42)
- Optimizer: AdamW (lr schedule: linear warmup 10% + linear decay)
- Gradient clipping: max_norm=1.0
- Early stopping: patience=15 on macro F1

## References

1. Arora & Kansal (2020). *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models.* arXiv:2006.08097
2. Devlin et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers.* NAACL-HLT
3. Lin et al. (2017). *Focal Loss for Dense Object Detection.* IEEE ICCV
4. Loshchilov & Hutter (2019). *Decoupled Weight Decay Regularization.* ICLR
5. Malo et al. (2014). *Good Debt or Bad Debt: Detecting Semantic Orientations in Economic Texts.* JASIST
6. Vaswani et al. (2017). *Attention Is All You Need.* NeurIPS

## License

This project was developed for CDS 525 (Deep Learning) at Lingnan University, Hong Kong.
