"""
Judge / Confidence Agent.

Takes the outputs from the NLI Verifier and computes:
    - A final verified label
    - A calibrated confidence score
    - A disagreement score 
    - An uncertainty flag

"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from agents.claim_analyzer import ClaimAnalyzer


class JudgeAgent:

    def __init__(self):
        self.analyzer = ClaimAnalyzer()

    def compute_disagreement(self, all_predictions: list) -> float:
        if len(all_predictions) <= 1:
            return 0.0

        confidences  = [p["confidence"] for p in all_predictions]
        disagreement = float(np.var(confidences))

        return round(min(disagreement, 1.0), 4)

    def adjudicate(self, claim: str, verifier_output: dict) -> dict:
        label      = verifier_output["label"]
        confidence = verifier_output["confidence"]
        best_sent  = verifier_output.get("best_sentence", "")
        all_preds  = verifier_output.get("all_predictions", [])

        disagreement     = self.compute_disagreement(all_preds)
        final_confidence = round(confidence * (1.0 - disagreement), 4)

        claim_meta = self.analyzer.analyze(claim)

        if claim_meta["claim_type"] == "numeric" and confidence < 0.6:
            final_confidence = round(final_confidence * 0.85, 4)

        uncertain   = final_confidence < config.LOW_CONFIDENCE_THRESHOLD
        final_label = label
        if uncertain and label != "NEI":
            final_label = "NEI"

        return {
            "final_label":      final_label,
            "raw_confidence":   confidence,
            "disagreement":     disagreement,
            "final_confidence": final_confidence,
            "uncertain":        uncertain,
            "claim_type":       claim_meta["claim_type"],
            "best_sentence":    best_sent
        }

    def adjudicate_multi(self, claim: str, multi_agent_output: dict) -> dict:
        label           = multi_agent_output["final_label"]
        confidence      = multi_agent_output["mean_confidence"]
        disagreement    = multi_agent_output["disagreement"]
        unanimous       = multi_agent_output["unanimous"]
        agreement_ratio = multi_agent_output["agreement_ratio"]

        claim_meta = self.analyzer.analyze(claim)

        if unanimous:
            final_confidence = round(min(confidence * 1.1, 1.0), 4)
        else:
            final_confidence = round(
                confidence * agreement_ratio * (1.0 - disagreement), 4
            )

        if claim_meta["claim_type"] == "numeric" and confidence < 0.6:
            final_confidence = round(final_confidence * 0.85, 4)

        uncertain   = final_confidence < config.LOW_CONFIDENCE_THRESHOLD
        final_label = label
        if uncertain and label != "NEI":
            final_label = "NEI"

        return {
            "final_label":      final_label,
            "raw_confidence":   confidence,
            "disagreement":     disagreement,
            "final_confidence": final_confidence,
            "uncertain":        uncertain,
            "unanimous":        unanimous,
            "agreement_ratio":  agreement_ratio,
            "claim_type":       claim_meta["claim_type"],
            "vote_counts":      multi_agent_output["vote_counts"],
            "best_sentence":    multi_agent_output["agent_predictions"][0].get(
                                    "best_sentence", ""
                                )
        }


if __name__ == "__main__":

    from agents.verifier import NLIVerifier

    verifier = NLIVerifier()
    judge    = JudgeAgent()

    claim = "Vitamin C reduces the duration of cold symptoms."
    sentences = [
        "Several trials show Vitamin C reduces cold duration.",
        "Prevention evidence remains inconclusive.",
        "Vitamin C has been widely studied."
    ]

    verifier_out = verifier.verify_multi(claim, sentences)
    decision     = judge.adjudicate(claim, verifier_out)

    print("\nClaim:", claim)
    print("\nJudge decision:")
    for k, v in decision.items():
        print(f"  {k}: {v}")