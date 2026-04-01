"""
Full pipeline evaluation.

Runs the complete system on the dev set with different retrievers.
Reports accuracy, per-class F1, disagreement stats, and saves results.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.tfidf_retriever import TFIDFRetriever
from retrieval.bm25_retriever  import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from evidence.evidence_selector import EvidenceSelector
from agents.verifier import NLIVerifier
from agents.judge    import JudgeAgent
import config


def evaluate(retriever_name="tfidf", n_samples=None):

    print(f"\nLoading dev set...")
    dev_df = pd.read_pickle(config.DEV_PATH)

    if n_samples:
        dev_df = dev_df.sample(
            n=min(n_samples, len(dev_df)),
            random_state=config.RANDOM_SEED
        )

    print(f"Evaluating on {len(dev_df)} examples "
          f"using {retriever_name.upper()} retrieval...")
    print(f"Label distribution:\n{dev_df['label'].value_counts()}\n")

    if retriever_name == "bm25":
        retriever = BM25Retriever(config.DATASET_PATH)
    elif retriever_name == "hybrid":
        retriever = HybridRetriever(config.DATASET_PATH, alpha=0.5)
    elif retriever_name == "dense":
        from retrieval.dense_retriever import DenseRetriever
        retriever = DenseRetriever(config.DATASET_PATH)
    elif retriever_name == "hybrid_dense":
        from retrieval.hybrid_dense_retriever import HybridDenseRetriever
        retriever = HybridDenseRetriever(config.DATASET_PATH)
    else:
        retriever = TFIDFRetriever(config.DATASET_PATH)

    selector = EvidenceSelector()
    verifier = NLIVerifier()
    judge    = JudgeAgent()

    y_true = []
    y_pred = []
    rows   = []

    for i, (_, row) in enumerate(dev_df.iterrows()):

        claim      = row["claim"]
        true_label = row["label"]

        retrieved = retriever.retrieve(claim, top_k=config.RETRIEVAL_TOP_K)

        evidence  = selector.select_evidence_multi(
            claim, retrieved[:3], top_k=config.EVIDENCE_TOP_K
        )
        sentences = [e["sentence"] for e in evidence]

        verifier_out = verifier.verify_multi(claim, sentences)

        decision = judge.adjudicate(claim, verifier_out)

        pred_label = decision["final_label"]
        y_true.append(true_label)
        y_pred.append(pred_label)

        rows.append({
            "claim":            claim,
            "true_label":       true_label,
            "pred_label":       pred_label,
            "final_confidence": decision["final_confidence"],
            "disagreement":     decision["disagreement"],
            "uncertain":        decision["uncertain"],
            "claim_type":       decision["claim_type"],
            "best_sentence":    decision["best_sentence"],
            "retrieved_doc_id": retrieved[0]["doc_id"] if retrieved else None,
            "retriever":        retriever_name
        })

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(dev_df)}")

    print("\n" + "=" * 55)
    print(f"RESULTS — {retriever_name.upper()} + NLI + Judge")
    print("=" * 55)

    acc = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy : {acc:.3f}")

    print("\nPer-class Report:")
    print(classification_report(
        y_true, y_pred,
        labels=["SUPPORT", "CONTRADICT", "NEI"],
        zero_division=0
    ))

    results_df = pd.DataFrame(rows)

    print(f"Uncertain predictions : "
          f"{results_df['uncertain'].sum()} / {len(results_df)}")
    print(f"Mean confidence       : "
          f"{results_df['final_confidence'].mean():.3f}")
    print(f"Mean disagreement     : "
          f"{results_df['disagreement'].mean():.3f}")

    print("\nClaim type breakdown:")
    print(results_df["claim_type"].value_counts())

    out_path = config.RESULTS_DIR + f"results_{retriever_name}.pkl"
    results_df.to_pickle(out_path)
    print(f"\nResults saved to {out_path}")

    return results_df, acc


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever", type=str, default="tfidf",
                        choices=["tfidf", "bm25"],
                        help="Which retriever to use")
    parser.add_argument("--n", type=int, default=None,
                        help="Number of dev samples (default: all)")
    args = parser.parse_args()

    evaluate(retriever_name=args.retriever, n_samples=args.n)