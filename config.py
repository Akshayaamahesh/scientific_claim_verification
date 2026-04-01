"""
Configuration for the Scientific Claim Verification system.

"""

DATASET_PATH       = "data/processed/scifact_processed.pkl"
TRAIN_PATH         = "data/processed/scifact_train.pkl"
DEV_PATH           = "data/processed/scifact_dev.pkl"
RESULTS_DIR        = "data/"

TFIDF_MAX_FEATURES = 50000
RETRIEVAL_TOP_K    = 5
RETRIEVER          = "dense"  # options: tfidf|bm25|hybrid|dense|hybrid_dense

EVIDENCE_TOP_K     = 3

NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
FINETUNED_MODEL    = "models/nli-finetuned"
NLI_LABELS         = ["SUPPORT", "CONTRADICT", "NEI"]

HIGH_CONFIDENCE_THRESHOLD = 0.70
LOW_CONFIDENCE_THRESHOLD  = 0.40

RANDOM_SEED = 42