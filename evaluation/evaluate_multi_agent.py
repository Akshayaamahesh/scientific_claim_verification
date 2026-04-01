"""
Multi-agent evaluation.

Compares single-agent vs multi-agent verification directly. 

Configurations compared:
    1. Single agent (DeBERTa only)     ← baseline
    2. Multi-agent majority vote       ← our system
    3. Multi-agent weighted vote       ← variant
"""

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.hybrid_retriever  import HybridRetriever
from evidence.evidence_selector  import EvidenceSelector
from agents.verifier             import NLIVerifier
from agents.multi_agent_verifier import MultiAgentVerifier, AGENT_MODELS
from agents.judge                import JudgeAgent
import config


def evaluate_single_agent(dev_df, retriever, selector, verifier, judge):
    print("\n  Running single-agent baseline...")
    y_true, y_pred = [], []

    for _, row in dev_df.iterrows():
        claim      = row["claim"]
        true_label = row["label"]

        retrieved = retriever.retrieve(claim, top_k=config.RETRIEVAL_TOP_K)
        abstract  = retrieved[0]["abstract"] if retrieved else ""
        evidence  = selector.select_evidence(
            claim, abstract, top_k=config.EVIDENCE_TOP_K
        )
        sentences    = [e["sentence"] for e in evidence]
        verifier_out = verifier.verify_multi(claim, sentences)
        decision     = judge.adjudicate(claim, verifier_out)

        y_true.append(true_label)
        y_pred.append(decision["final_label"])

    return y_true, y_pred


def evaluate_multi_agent(dev_df, retriever, selector,
                          multi_verifier, judge, use_weighted=False):
    label = "weighted" if use_weighted else "majority"
    print(f"\n  Running multi-agent ({label} vote)...")
    y_true, y_pred = [], []

    for _, row in dev_df.iterrows():
        claim      = row["claim"]
        true_label = row["label"]

        retrieved = retriever.retrieve(claim, top_k=config.RETRIEVAL_TOP_K)
        evidence  = selector.select_evidence_multi(
            claim, retrieved[:3], top_k=config.EVIDENCE_TOP_K
        )
        sentences = [e["sentence"] for e in evidence]
        multi_out = multi_verifier.verify(claim, sentences)
        decision  = judge.adjudicate_multi(claim, multi_out)

        pred = decision["final_label"] if not use_weighted else \
               multi_out["weighted_label"]

        y_true.append(true_label)
        y_pred.append(pred)

    return y_true, y_pred


def print_results(config_name, y_true, y_pred):
    acc      = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred,
                        labels=["SUPPORT", "CONTRADICT", "NEI"],
                        average="macro", zero_division=0)
    sup_f1   = f1_score(y_true, y_pred, labels=["SUPPORT"],
                        average="macro", zero_division=0)
    con_f1   = f1_score(y_true, y_pred, labels=["CONTRADICT"],
                        average="macro", zero_division=0)
    nei_f1   = f1_score(y_true, y_pred, labels=["NEI"],
                        average="macro", zero_division=0)
    return {
        "config":        config_name,
        "accuracy":      round(acc, 3),
        "macro_f1":      round(macro_f1, 3),
        "support_f1":    round(sup_f1, 3),
        "contradict_f1": round(con_f1, 3),
        "nei_f1":        round(nei_f1, 3)
    }


def run_multi_agent_evaluation(n_samples=100):

    print("Loading dev set...")
    dev_df = pd.read_pickle(config.DEV_PATH).sample(
        n=min(n_samples, 173),
        random_state=config.RANDOM_SEED
    )
    print(f"Evaluating on {len(dev_df)} examples...\n")

    retriever      = HybridRetriever(config.DATASET_PATH, alpha=0.5)
    selector       = EvidenceSelector()
    single_verifier = NLIVerifier()
    multi_verifier  = MultiAgentVerifier()
    judge           = JudgeAgent()

    results = []

    y_true, y_pred = evaluate_single_agent(
        dev_df, retriever, selector, single_verifier, judge
    )
    results.append(print_results("1. Single Agent (DeBERTa)", y_true, y_pred))

    y_true, y_pred = evaluate_multi_agent(
        dev_df, retriever, selector, multi_verifier, judge,
        use_weighted=False
    )
    results.append(print_results("2. Multi-Agent Majority Vote", y_true, y_pred))

    y_true, y_pred = evaluate_multi_agent(
        dev_df, retriever, selector, multi_verifier, judge,
        use_weighted=True
    )
    results.append(print_results("3. Multi-Agent Weighted Vote", y_true, y_pred))

    print("\n" + "=" * 75)
    print("SINGLE-AGENT vs MULTI-AGENT COMPARISON")
    print("=" * 75)
    print(f"{'Configuration':<40} {'Acc':>6} {'MacF1':>7} "
          f"{'SUP-F1':>8} {'CON-F1':>8} {'NEI-F1':>8}")
    print("-" * 75)

    for r in results:
        print(f"{r['config']:<40} {r['accuracy']:>6.3f} "
              f"{r['macro_f1']:>7.3f} {r['support_f1']:>8.3f} "
              f"{r['contradict_f1']:>8.3f} {r['nei_f1']:>8.3f}")

    print("-" * 75)

    results_df = pd.DataFrame(results)
    results_df.to_pickle(config.RESULTS_DIR + "multi_agent_results.pkl")
    print("\nResults saved to data/multi_agent_results.pkl")

    return results_df

if __name__ == "__main__":
    run_multi_agent_evaluation(n_samples=100)