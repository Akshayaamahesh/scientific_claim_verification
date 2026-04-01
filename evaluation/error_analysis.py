"""
Error Analysis.

Breaks down system errors by:
    1. Claim type - numeric, comparative, causal, general 
    2. True label 
    3. Confidence of wrong predictions
    4. Most common confusion pairs 

Reads from saved full pipeline results.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def run_error_analysis(results_path=None):

    if results_path is None:
        results_path = config.RESULTS_DIR + "results_tfidf.pkl"

    print(f"Loading results from {results_path}...")
    df = pd.read_pickle(results_path)

    correct   = df["true_label"] == df["pred_label"]
    incorrect = ~correct

    print("\n" + "=" * 55)
    print("ERROR ANALYSIS")
    print("=" * 55)

    print(f"\nTotal examples : {len(df)}")
    print(f"Correct        : {correct.sum()}  ({correct.mean():.1%})")
    print(f"Incorrect      : {incorrect.sum()}  ({incorrect.mean():.1%})")

    print("\n── Accuracy by Claim Type ──")
    for ctype in ["general", "numeric", "causal", "comparative"]:
        subset = df[df["claim_type"] == ctype]
        if len(subset) == 0:
            continue
        acc = (subset["true_label"] == subset["pred_label"]).mean()
        print(f"  {ctype:<14} : {acc:.3f}  (n={len(subset)})")

    print("\n── Accuracy by True Label ──")
    for label in ["SUPPORT", "CONTRADICT", "NEI"]:
        subset = df[df["true_label"] == label]
        if len(subset) == 0:
            continue
        acc = (subset["true_label"] == subset["pred_label"]).mean()
        print(f"  {label:<14} : {acc:.3f}  (n={len(subset)})")

    labels = ["SUPPORT", "CONTRADICT", "NEI"]
    cm     = confusion_matrix(df["true_label"], df["pred_label"],
                               labels=labels)

    print("\n── Confusion Matrix ──")
    print(f"{'':>14}", end="")
    for l in labels:
        print(f"{l:>12}", end="")
    print()
    for i, true_l in enumerate(labels):
        print(f"  {true_l:<12}", end="")
        for j in range(len(labels)):
            print(f"{cm[i][j]:>12}", end="")
        print()

    wrong_df = df[incorrect]
    print(f"\n── Confidence on Wrong Predictions ──")
    print(f"  Mean confidence (wrong) : "
          f"{wrong_df['final_confidence'].mean():.3f}")
    print(f"  Mean confidence (right) : "
          f"{df[correct]['final_confidence'].mean():.3f}")
    print(f"  High-confidence errors  : "
          f"{(wrong_df['final_confidence'] > 0.6).sum()}")

    print("\n── Most Common Confusion Pairs (true → predicted) ──")
    error_pairs = wrong_df.groupby(
        ["true_label", "pred_label"]
    ).size().sort_values(ascending=False)

    for (true_l, pred_l), count in error_pairs.items():
        print(f"  {true_l} → {pred_l} : {count} times")

    print("\n── Sample Wrong Predictions ──")
    sample = wrong_df.sample(
        n=min(5, len(wrong_df)),
        random_state=config.RANDOM_SEED
    )
    for _, row in sample.iterrows():
        print(f"\n  Claim      : {row['claim'][:100]}")
        print(f"  True label : {row['true_label']}")
        print(f"  Predicted  : {row['pred_label']}")
        print(f"  Confidence : {row['final_confidence']}")
        print(f"  Claim type : {row['claim_type']}")

    print("\nError analysis complete.")
    return df


if __name__ == "__main__":
    run_error_analysis()