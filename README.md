# Football Sentiment Analysis — Euro 2024

**Predictive Power of Social Media: A Comparative Study of Classical ML, Lexicon-Based, and LLM Approaches for Football Sentiment Analysis**
https://zenodo.org/records/19675434 | https://doi.org/10.5281/zenodo.19675434

Mohammad Taiyab Khan | MSc Data Science & Analytics | Royal Holloway, University of London

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

This repository contains the full code, data, and results for a systematic comparison of four sentiment analysis methods applied to public Twitter discourse about the England national football team during UEFA Euro 2024.

**The core finding:** Benchmark classification accuracy and domain-level sentiment sensitivity are dissociable properties of sentiment classifiers. The model with the highest benchmark accuracy (TF-IDF + Logistic Regression, AUC = 0.927) fails to detect statistically significant sentiment differences between match outcomes, while a zero-shot LLM classifier (Claude Haiku) succeeds (p = .012).

---

## Key Results

### Benchmark Performance (TweetEval Sentiment Test Set)

| Model | Accuracy | F1 (Weighted) | AUC-ROC |
|---|---|---|---|
| TF-IDF + LR (binary) | **0.862** | **0.860** | **0.927** |
| TF-IDF + LR (3-class) | 0.646 | 0.643 | 0.800 |
| VADER (lexicon) | 0.778 | 0.769 | 0.820 |
| Claude Haiku (LLM) | — | — | — |

### Domain Analysis — Win Day vs Loss Day Sentiment

| Model | Win Day (Jul 10) | Loss Day (Jul 14) | p-value | Significant? |
|---|---|---|---|---|
| TF-IDF + LR | 0.721 | 0.668 | 0.104 | No |
| TF-IDF + LR (3-class) | 0.434 | 0.405 | 0.467 | No |
| VADER | 0.633 | 0.588 | 0.113 | No |
| **Claude Haiku** | **0.593** | **0.528** | **0.012** | **Yes** |

- Win day: England 2–1 Netherlands (Semi-Final, 14 July 2024)
- Loss day: England 1–2 Spain (Final, 14 July 2024)
- 81.3% cross-model agreement across all classifiers
- 79.2% of football tweets classified as **neutral** by LLM — suggesting football Twitter is predominantly informational, not emotional

### Figures

| | |
|---|---|
| ![Sentiment Timeline](outputs/figures/fig1_sentiment_timeline.png) | ![Model Comparison](outputs/figures/fig2_model_comparison.png) |
| *Fig 1. Sentiment trend across three models (Jul 7–21, 2024)* | *Fig 2. Benchmark performance comparison* |
| ![Win vs Loss](outputs/figures/fig5_match_sentiment.png) | ![Word Clouds](outputs/figures/fig6_wordclouds.png) |
| *Fig 3. Sentiment distribution on win vs loss days* | *Fig 4. Most frequent terms by sentiment class* |

---

## Repository Structure

```
euro2024-football-sentiment/
│
├── README.md                          ← you are here
├── LICENSE                            ← MIT licence
├── requirements.txt                   ← Python dependencies
│
├── data/
│   ├── README.md                      ← data sources and download instructions
│   ├── euro2024_domain.csv            ← 871 Euro 2024 tweets (domain set)
│   └── euro_all_models_final.csv      ← domain set with all 4 model scores
│
├── notebooks/
│   ├── 01_preprocessing.ipynb         ← text cleaning pipeline
│   ├── 02_baseline_models.ipynb       ← TF-IDF + LR (binary & 3-class) + VADER
│   ├── 03_claude_classifier.ipynb     ← Claude API zero-shot classification
│   └── 04_analysis_figures.ipynb      ← all publication figures
│
├── outputs/
│   ├── figures/
│   │   ├── fig1_sentiment_timeline.png
│   │   ├── fig2_model_comparison.png
│   │   ├── fig3_roc_curves.png
│   │   ├── fig4_confusion_matrices.png
│   │   ├── fig5_match_sentiment.png
│   │   └── fig6_wordclouds.png
│   └── results/
│       └── model_comparison.csv       ← Table 1 & 2 from paper
│
└── paper/
    └── Euro2024_Sentiment_Paper.docx  ← full paper draft
```

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/data-professional-taiyabkhan/euro2024-football-sentiment.git
cd euro2024-football-sentiment
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Get the training data

The domain dataset (871 Euro 2024 tweets) is included in `data/`. The training dataset (TweetEval) must be downloaded separately due to licensing:

```bash
# Option A — HuggingFace datasets (recommended)
pip install datasets
python -c "from datasets import load_dataset; ds = load_dataset('tweet_eval', 'sentiment'); ds.save_to_disk('data/tweeteval')"

# Option B — clone from Cardiff NLP GitHub
git clone https://github.com/cardiffnlp/tweeteval.git data/tweeteval_raw
```

See `data/README.md` for full instructions.

### 4. Run the notebooks in order

```bash
jupyter notebook notebooks/
```

Run `01` → `02` → `03` → `04` in sequence. Each notebook saves its outputs for the next step.

> **Note:** Notebook `03` requires an Anthropic API key.  
> Get a free key at [console.anthropic.com](https://console.anthropic.com).  
> Set it as an environment variable: `export ANTHROPIC_API_KEY=your_key_here`

---

## Data

### Domain dataset (`data/euro2024_domain.csv`)

871 English-language tweets collected July 7–21, 2024 via APIFY Twitter Scraper.

| Column | Description |
|---|---|
| `date` | Tweet date (YYYY-MM-DD) |
| `time` | Tweet time (HH:MM:SS) |
| `text` | Original tweet text |
| `nearest_match` | Closest England match label |
| `match_result` | 1 = win, -1 = loss, NaN = non-match day |
| `lr_pred` | Binary LR prediction (0/1) |
| `lr_proba_pos` | LR positive probability |
| `vader_proba_pos` | VADER positive score (transformed) |
| `claude_label` | Claude classification (positive/neutral/negative) |
| `claude_proba_pos` | Claude pseudo-probability of positive |
| `agreement` | Cross-model agreement category |

### Training data

TweetEval (Barbieri et al., 2020) — 57,763 labelled tweets (negative/neutral/positive). Not included in this repo; see Quickstart section for download instructions.

---

## Models

| Model | Type | Training | Key Parameter |
|---|---|---|---|
| 2-class LR | Classical ML | TweetEval (binary) | C=5, L2, TF-IDF 80K features |
| 3-class LR | Classical ML | TweetEval (full) | Same architecture, 3-class output |
| VADER | Rule-based | None (lexicon) | Compound score threshold = 0 |
| Claude Haiku | LLM (zero-shot) | None | Domain-adapted system prompt |

### Claude prompt design

The LLM classifier uses a domain-adapted system prompt that explicitly encodes football Twitter conventions:
- British understatement ("not bad" = positive)
- Football-specific negativity ("typical England", "bottle it")
- Sarcasm indicators ("brilliant 🙄" = negative)
- Neutral factual updates (score lines, lineup news)

The full prompt is available in `notebooks/03_claude_classifier.ipynb`.

---

## Reproducing the Paper

All results in the paper can be reproduced by running the four notebooks in order. The final figures and results table are saved to `outputs/`. Expected runtimes on a standard laptop (CPU only):

| Step | Runtime |
|---|---|
| Notebook 01 — Preprocessing | ~2 min |
| Notebook 02 — Baseline models | ~8 min |
| Notebook 03 — Claude API classifier | ~6 min (rate-limited) |
| Notebook 04 — Figures | ~1 min |

---

## Citation

If you use this code or dataset, please cite:

```bibtex
@article{khan2024football,
  title   = {Predictive Power of Social Media: A Comparative Study of Classical ML,
             Lexicon-Based, and LLM Approaches for Football Sentiment Analysis},
  author  = {Khan, Mohammad Taiyab},
  year    = {2025},
  journal = {arXiv preprint},
  note    = {arXiv:XXXX.XXXXX}
}
```

Update the arXiv ID once uploaded.

---

## References

- Barbieri, F., Camacho-Collados, J., Espinosa Anke, L., & Neves, L. (2020). TweetEval: Unified benchmark and comparative evaluation for tweet classification. *EMNLP Findings 2020*.
- Hutto, C. J., & Gilbert, E. (2014). VADER: A parsimonious rule-based model for sentiment analysis of social media text. *ICWSM 2014*.
- Brown, T. B. et al. (2020). Language models are few-shot learners. *NeurIPS 2020*.
- Anthropic. (2024). Claude Haiku [Large language model]. https://www.anthropic.com

---

## Licence

MIT — see [LICENSE](LICENSE) for details.

---

## Contact

Mohammad Taiyab Khan  
MSc Data Science & Analytics, Royal Holloway, University of London  
mohammadtaiyabkhan21@gmail.com  
[LinkedIn](https://www.linkedin.com/in/khanmohdtaiyab) · [GitHub](https://github.com/data-professional-taiyabkhan)
