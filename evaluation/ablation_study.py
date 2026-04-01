"""
Ablation Study — compares four pipeline configurations (gold abstract baseline,
TF-IDF only, TF-IDF + Judge, Hybrid + Judge) to isolate each component's contribution
to overall accuracy and per-class F1 on the SciFact development set.
"""

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.tfidf_retriever  import TFIDFRetriever
from retrieval.hybrid_retriever import HybridRetriever
from evidence.evidence_selector import EvidenceSelector
from agents.verifier import NLIVerifier
from agents.judge    import JudgeAgent
import config


def run_configuration(dev_df, config_name, retriever, hybrid_retriever,
                       selector, verifier, judge, use_judge,
                       use_gold_abstract, use_hybrid=False):

    print(f"\n  Running: {config_name}...")
    y_true, y_pred = [], []

    for _, row in dev_df.iterrows():

        claim      = row["claim"]
        true_label = row["label"]

        if use_gold_abstract:
            abstract = row["abstract"]
            evidence = selector.select_evidence(
                claim, abstract, top_k=config.EVIDENCE_TOP_K
            )
        elif use_hybrid:
            retrieved = hybrid_retriever.retrieve(
                claim, top_k=config.RETRIEVAL_TOP_K
            )
            evidence  = selector.select_evidence_multi(
                claim, retrieved[:3], top_k=config.EVIDENCE_TOP_K
            )
        else:
            retrieved = retriever.retrieve(claim, top_k=config.RETRIEVAL_TOP_K)
            evidence  = selector.select_evidence_multi(
                claim, retrieved[:3], top_k=config.EVIDENCE_TOP_K
            )

        sentences    = [e["sentence"] for e in evidence]
        verifier_out = verifier.verify_multi(claim, sentences)

        if use_judge:
            decision   = judge.adjudicate(claim, verifier_out)
            pred_label = decision["final_label"]
        else:
            pred_label = verifier_out["label"]

        y_true.append(true_label)
        y_pred.append(pred_label)

    acc           = accuracy_score(y_true, y_pred)
    macro_f1      = f1_score(y_true, y_pred,
                             labels=["SUPPORT", "CONTRADICT", "NEI"],
                             average="macro", zero_division=0)
    support_f1    = f1_score(y_true, y_pred, labels=["SUPPORT"],
                             average="macro", zero_division=0)
    contradict_f1 = f1_score(y_true, y_pred, labels=["CONTRADICT"],
                             average="macro", zero_division=0)
    nei_f1        = f1_score(y_true, y_pred, labels=["NEI"],
                             average="macro", zero_division=0)

    return {
        "config":        config_name,
        "accuracy":      round(acc, 3),
        "macro_f1":      round(macro_f1, 3),
        "support_f1":    round(support_f1, 3),
        "contradict_f1": round(contradict_f1, 3),
        "nei_f1":        round(nei_f1, 3)
    }


def run_ablation(n_samples=100):

    print("Loading dev set...")
    dev_df = pd.read_pickle(config.DEV_PATH)

    if n_samples:
        dev_df = dev_df.sample(
            n=min(n_samples, len(dev_df)),
            random_state=config.RANDOM_SEED
        )

    print(f"Running ablation on {len(dev_df)} examples...\n")

    retriever        = TFIDFRetriever(config.DATASET_PATH)
    hybrid_retriever = HybridRetriever(config.DATASET_PATH, alpha=0.5)
    selector         = EvidenceSelector()
    verifier         = NLIVerifier()
    judge            = JudgeAgent()

    configurations = [
        {
            "name":               "1. Gold Abstract + NLI (no retrieval)",
            "use_gold_abstract":  True,
            "use_judge":          False
        },
        {
            "name":               "2. TF-IDF Retrieval + NLI (no Judge)",
            "use_gold_abstract":  False,
            "use_judge":          False
        },
        {
            "name":               "3. TF-IDF + NLI + Judge (full system)",
            "use_gold_abstract":  False,
            "use_judge":          True
        },
        {
            "name":               "4. Hybrid Retrieval + NLI + Judge (best system)",
            "use_gold_abstract":  False,
            "use_judge":          True,
            "use_hybrid":         True
        },
    ]

    results = []
    for cfg in configurations:
        result = run_configuration(
            dev_df           = dev_df,
            config_name      = cfg["name"],
            retriever        = retriever,
            hybrid_retriever = hybrid_retriever,
            selector         = selector,
            verifier         = verifier,
            judge            = judge,
            use_judge        = cfg["use_judge"],
            use_gold_abstract= cfg["use_gold_abstract"],
            use_hybrid       = cfg.get("use_hybrid", False)
        )
        results.append(result)

    print("\n" + "=" * 75)
    print("ABLATION STUDY RESULTS")
    print("=" * 75)
    print(f"{'Configuration':<45} {'Acc':>6} {'MacF1':>7} "
          f"{'SUP-F1':>8} {'CON-F1':>8} {'NEI-F1':>8}")
    print("-" * 75)

    for r in results:
        print(f"{r['config']:<45} {r['accuracy']:>6.3f} {r['macro_f1']:>7.3f} "
              f"{r['support_f1']:>8.3f} {r['contradict_f1']:>8.3f} "
              f"{r['nei_f1']:>8.3f}")

    print("-" * 75)

    results_df = pd.DataFrame(results)
    results_df.to_pickle(config.RESULTS_DIR + "ablation_results.pkl")
    print("\nAblation results saved to data/ablation_results.pkl")

    return results_df


if __name__ == "__main__":
    run_ablation(n_samples=100)