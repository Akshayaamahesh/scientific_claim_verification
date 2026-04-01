"""
main.py — Main file for the Scientific Claim Verification system.

Usage:
    python main.py --claim "Your scientific claim here"
    python main.py --claim "..." --retriever bm25
    python main.py --claim "..." --retriever hybrid
    python main.py --claim "..." --retriever hybrid --multi
    python main.py --evaluate --n 50
    python main.py --evaluate --retriever hybrid --n 100
"""

import argparse
import sys
import os

from retrieval.tfidf_retriever  import TFIDFRetriever
from retrieval.bm25_retriever   import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from evidence.evidence_selector import EvidenceSelector
from agents.verifier import NLIVerifier
from agents.judge    import JudgeAgent
import config


def get_retriever(retriever_name):
    """returns the correct retriever by name."""
    if retriever_name == "bm25":
        return BM25Retriever(config.DATASET_PATH)
    elif retriever_name == "hybrid":
        return HybridRetriever(config.DATASET_PATH, alpha=0.5)
    elif retriever_name == "dense":
        from retrieval.dense_retriever import DenseRetriever
        return DenseRetriever(config.DATASET_PATH)
    elif retriever_name == "hybrid_dense":
        from retrieval.hybrid_dense_retriever import HybridDenseRetriever
        return HybridDenseRetriever(config.DATASET_PATH)
    else:
        return TFIDFRetriever(config.DATASET_PATH)


def verify_claim(claim, retriever_name="hybrid", use_multi=False, verbose=True):
    """
    Run the full verification pipeline on a single claim.
    """
    retriever = get_retriever(retriever_name)
    selector  = EvidenceSelector()
    judge     = JudgeAgent()

    if verbose:
        print(f"\nClaim: {claim}")
        print(f"Retriever : {retriever_name.upper()}")
        print(f"Mode      : {'Multi-Agent' if use_multi else 'Single-Agent'}")
        print("=" * 60)

    retrieved = retriever.retrieve(claim, top_k=config.RETRIEVAL_TOP_K)

    if verbose:
        print(f"\nStep 1 — Retrieved {len(retrieved)} documents")
        print(f"  Top doc ID : {retrieved[0]['doc_id']}")
        print(f"  Top score  : {retrieved[0]['score']:.4f}")

    evidence  = selector.select_evidence_multi(
        claim, retrieved[:3], top_k=config.EVIDENCE_TOP_K
    )
    sentences = [e["sentence"] for e in evidence]

    if verbose:
        print(f"\nStep 2 — Selected {len(sentences)} evidence sentences")
        for e in evidence:
            print(f"  • [doc {e['doc_id']}, score {e['score']:.3f}] "
                  f"{e['sentence'][:100]}")

    if use_multi:
        from agents.multi_agent_verifier import MultiAgentVerifier

        multi_verifier = MultiAgentVerifier()
        multi_out      = multi_verifier.verify(claim, sentences)
        decision       = judge.adjudicate_multi(claim, multi_out)

        if verbose:
            print(f"\nStep 3 — Multi-Agent Verifier")
            print(f"  Agent votes    : {multi_out['vote_counts']}")
            print(f"  Unanimous      : {multi_out['unanimous']}")
            print(f"  Agreement      : {multi_out['agreement_ratio']:.0%}")
            print(f"  Disagreement   : {multi_out['disagreement']}")
            for pred in multi_out["agent_predictions"]:
                print(f"  {pred['agent']:<12} → {pred['label']} "
                      f"(confidence: {pred['confidence']})")

    else:
        verifier     = NLIVerifier()
        verifier_out = verifier.verify_multi(claim, sentences)
        decision     = judge.adjudicate(claim, verifier_out)

        if verbose:
            print(f"\nStep 3 — NLI Verifier")
            print(f"  Raw label      : {verifier_out['label']}")
            print(f"  Raw confidence : {verifier_out['confidence']}")

    if verbose:
        print(f"\nStep 4 — Judge Agent")
        print(f"  Final label      : {decision['final_label']}")
        print(f"  Final confidence : {decision['final_confidence']}")
        print(f"  Disagreement     : {decision['disagreement']}")
        print(f"  Uncertain        : {decision['uncertain']}")
        print(f"  Claim type       : {decision['claim_type']}")
        print(f"\n{'='*60}")
        print(f"  VERDICT: {decision['final_label']}  "
              f"(confidence: {decision['final_confidence']})")
        print(f"{'='*60}\n")

    return decision


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Scientific Claim Verification System"
    )
    parser.add_argument("--claim",     type=str,
                        help="Claim to verify")
    parser.add_argument("--retriever", type=str, default="hybrid_dense",
                       choices=["tfidf", "bm25", "hybrid", "dense", "hybrid_dense"],
                       help="Retriever to use (default: hybrid_dense)")
    parser.add_argument("--multi",     action="store_true",
                        help="Use multi-agent verification (3 NLI models)")
    parser.add_argument("--evaluate",  action="store_true",
                        help="Run evaluation on dev set")
    parser.add_argument("--n",         type=int, default=50,
                        help="Number of dev samples to evaluate (default: 50)")
    args = parser.parse_args()

    if args.evaluate:
        from evaluation.evaluate_full_pipeline import evaluate
        evaluate(retriever_name=args.retriever, n_samples=args.n)

    elif args.claim:
        verify_claim(
            args.claim,
            retriever_name=args.retriever,
            use_multi=args.multi
        )

    else:
        test_claim = "Vitamin C reduces the duration of cold symptoms."
        print("\n" + "=" * 60)
        print("DEMO MODE — Single-Agent vs Multi-Agent")
        print("=" * 60)

        print("\n--- Single Agent (Hybrid Retrieval) ---")
        verify_claim(test_claim, retriever_name="hybrid", use_multi=False)

        print("\n--- Multi-Agent (Hybrid Retrieval) ---")
        verify_claim(test_claim, retriever_name="hybrid", use_multi=True)