# Adaptive Multi-Agent System for Scientific Claim Verification

Team: Akshayaa Mahesh, Dhruthi Rajesh, Meghana Raghavendra

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Project Structure
```
scientific-claim-verification/
├── preprocessing/        # Dataset construction from SciFact
├── retrieval/            # TF-IDF, BM25, Hybrid, Dense, and Hybrid Dense retrievers
├── evidence/             # Sentence-level evidence selector 
├── agents/               # Claim Analyzer, NLI Verifier, Judge Agent, Multi-Agent Verifier
├── evaluation/           # Ablation study, error analysis, full pipeline eval, multi-agent eval
├── models/               # Fine-tuned NLI model 
├── data/                 # Processed datasets and results (generated)
├── main.py               # Main entry point
├── train.py              # Fine-tuning script
└── config.py             # All hyperparameters and paths
```

## Step 1 — Build Dataset
```bash
python preprocessing/build_scifact_dataset.py
```

## Step 2 — Verify a Single Claim
```bash
# Single-agent 
python main.py --claim "Vitamin C reduces the duration of cold symptoms."

# Choose retriever
python main.py --claim "Smoking causes lung cancer." --retriever dense
python main.py --claim "Smoking causes lung cancer." --retriever tfidf
python main.py --claim "Smoking causes lung cancer." --retriever hybrid

# Multi-agent mode 
python main.py --claim "Smoking causes lung cancer." --retriever dense --multi
```

## Step 3 — Run Full Evaluation
```bash
# Best configuration 
python main.py --evaluate --retriever dense --n 100

# Compare all retrievers
python main.py --evaluate --retriever tfidf --n 100
python main.py --evaluate --retriever bm25 --n 100
python main.py --evaluate --retriever hybrid --n 100
python main.py --evaluate --retriever hybrid_dense --n 100
```

## Step 4 — Run Retrieval Comparison (all 5 methods)
```bash
python retrieval/evaluate_retrieval.py
```

## Step 5 — Run Ablation Study
```bash
python evaluation/ablation_study.py
```

## Step 6 — Run Error Analysis
```bash
python evaluation/error_analysis.py
```

## Step 7 — Run Single-Agent vs Multi-Agent Comparison
```bash
python evaluation/evaluate_multi_agent.py
```

## Fine-tuning (Optional — pre-trained model included)
A fine-tuned model is already saved in models/nli-finetuned/ so you do not need to run this.
To retrain from scratch (takes ~2 hours on CPU):
```bash
python train.py --epochs 5 --batch_size 4 --lr 1e-5
```
After training, update config.py:
```python
NLI_MODEL = "models/nli-finetuned"
```
Note: The fine-tuned model performs best with gold evidence (71.3% dev accuracy, 0.682 macro F1).
For pipeline evaluation, the zero-shot model is recommended (config default).
