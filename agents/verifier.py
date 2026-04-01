"""
NLI Verifier Agent.

Two modes:
    Zero-shot mode  and  Fine-tuned mode. 

"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import pipeline
import config

FINETUNED_ID2LABEL = {0: "SUPPORT", 1: "CONTRADICT", 2: "NEI"}


def is_finetuned_model(model_path):
    return "nli-finetuned" in str(model_path) or \
           "models/" in str(model_path)


class NLIVerifier:

    def __init__(self):
        print(f"Loading NLI model: {config.NLI_MODEL} ...")

        if is_finetuned_model(config.NLI_MODEL):
            self.mode = "finetuned"
            self.nli  = pipeline(
                "text-classification",
                model=config.NLI_MODEL,
                top_k=None
            )
        else:
            self.mode = "zero-shot"
            self.nli  = pipeline(
                "zero-shot-classification",
                model=config.NLI_MODEL
            )

        print(f"NLI model ready (mode: {self.mode}).")

    def verify_single(self, claim, evidence):
        if not evidence or evidence.strip() == "":
            return {
                "label":      "NEI",
                "confidence": 0.0,
                "all_scores": {"SUPPORT": 0.0, "CONTRADICT": 0.0, "NEI": 1.0}
            }

        if self.mode == "finetuned":
            return self._verify_finetuned(claim, evidence)
        else:
            return self._verify_zero_shot(claim, evidence)

    def _verify_finetuned(self, claim, evidence):
        text   = f"{claim} [SEP] {evidence}"
        result = self.nli(text)[0]  

        scores = {r["label"]: round(r["score"], 4) for r in result}

        best = max(result, key=lambda x: x["score"])

        return {
            "label":      best["label"],
            "confidence": round(best["score"], 4),
            "all_scores": scores
        }

    def _verify_zero_shot(self, claim, evidence):
        result = self.nli(
            sequences=evidence,
            candidate_labels=config.NLI_LABELS,
            hypothesis_template="This text {} the following claim: " + claim
        )

        scores = dict(zip(result["labels"], result["scores"]))

        return {
            "label":      result["labels"][0],
            "confidence": round(result["scores"][0], 4),
            "all_scores": {k: round(v, 4) for k, v in scores.items()}
        }

    def verify_multi(self, claim, sentences):
        if not sentences:
            return {
                "label":           "NEI",
                "confidence":      0.0,
                "best_sentence":   "",
                "all_predictions": []
            }

        all_predictions = []
        for sent in sentences:
            pred           = self.verify_single(claim, sent)
            pred["sentence"] = sent
            all_predictions.append(pred)

        best = max(all_predictions, key=lambda x: x["confidence"])

        return {
            "label":           best["label"],
            "confidence":      best["confidence"],
            "best_sentence":   best["sentence"],
            "all_predictions": all_predictions
        }


if __name__ == "__main__":

    verifier = NLIVerifier()

    claim = "Vitamin C reduces the duration of cold symptoms."
    sentences = [
        "Several trials show Vitamin C reduces cold duration.",
        "Prevention evidence remains inconclusive.",
        "Vitamin C has been studied for the common cold."
    ]

    print("\n-- Single sentence test --")
    print(verifier.verify_single(claim, sentences[0]))

    print("\n-- Multi-sentence test --")
    result = verifier.verify_multi(claim, sentences)
    print("Label:", result["label"])
    print("Confidence:", result["confidence"])
    print("Best sentence:", result["best_sentence"])