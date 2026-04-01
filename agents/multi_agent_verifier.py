"""
Multi-Agent Verifier.

Models used:
    Agent 1: cross-encoder/nli-deberta-v3-small  
    Agent 2: facebook/bart-large-mnli            
    Agent 3: cross-encoder/nli-MiniLM2-L6-H768  
"""

import numpy as np
from collections import Counter
from transformers import pipeline
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


AGENT_MODELS = [
    {
        "name":  "DeBERTa",
        "model": "cross-encoder/nli-deberta-v3-small"
    },
    {
        "name":  "BART",
        "model": "facebook/bart-large-mnli"
    },
    {
        "name":  "MiniLM",
        "model": "cross-encoder/nli-MiniLM2-L6-H768"
    }
]


class SingleVerifierAgent:

    def __init__(self, model_name, agent_name):
        self.agent_name = agent_name
        print(f"  Loading agent '{agent_name}': {model_name} ...")
        self.nli = pipeline(
            "zero-shot-classification",
            model=model_name
        )
        print(f"  Agent '{agent_name}' ready.")

    def predict(self, claim, evidence):
        if not evidence or evidence.strip() == "":
            return {
                "agent":      self.agent_name,
                "label":      "NEI",
                "confidence": 0.0,
                "all_scores": {"SUPPORT": 0.0, "CONTRADICT": 0.0, "NEI": 1.0}
            }

        result = self.nli(
            sequences=evidence,
            candidate_labels=config.NLI_LABELS,
            hypothesis_template="This text {} the following claim: " + claim
        )

        scores = dict(zip(result["labels"], result["scores"]))

        return {
            "agent":      self.agent_name,
            "label":      result["labels"][0],
            "confidence": round(result["scores"][0], 4),
            "all_scores": {k: round(v, 4) for k, v in scores.items()}
        }

    def predict_multi(self, claim, sentences):
        if not sentences:
            return {
                "agent":        self.agent_name,
                "label":        "NEI",
                "confidence":   0.0,
                "best_sentence": "",
                "all_preds":    []
            }

        all_preds = []
        for sent in sentences:
            pred = self.predict(claim, sent)
            pred["sentence"] = sent
            all_preds.append(pred)

        best = max(all_preds, key=lambda x: x["confidence"])

        return {
            "agent":         self.agent_name,
            "label":         best["label"],
            "confidence":    best["confidence"],
            "best_sentence": best["sentence"],
            "all_preds":     all_preds
        }


class MultiAgentVerifier:
    def __init__(self, models=None):
        if models is None:
            models = AGENT_MODELS

        print(f"\nLoading {len(models)} verifier agents...")
        self.agents = [
            SingleVerifierAgent(m["model"], m["name"])
            for m in models
        ]
        print(f"All {len(models)} agents ready.\n")

    def verify(self, claim, sentences):

        agent_preds = []
        for agent in self.agents:
            pred = agent.predict_multi(claim, sentences)
            agent_preds.append(pred)

        labels     = [p["label"] for p in agent_preds]
        vote_count = Counter(labels)
        majority_label = vote_count.most_common(1)[0][0]
        majority_count = vote_count.most_common(1)[0][1]
        agreement_ratio = majority_count / len(self.agents)

        weighted_scores = {"SUPPORT": 0.0, "CONTRADICT": 0.0, "NEI": 0.0}
        for pred in agent_preds:
            for label in config.NLI_LABELS:
                weighted_scores[label] += pred["all_preds"][0]["all_scores"].get(
                    label, 0.0
                ) if pred["all_preds"] else 0.0

        weighted_label = max(weighted_scores, key=weighted_scores.get)

        confidences  = [p["confidence"] for p in agent_preds]
        disagreement = round(float(np.std(confidences)), 4)

        unanimous = len(set(labels)) == 1

        majority_agents = [p for p in agent_preds if p["label"] == majority_label]
        mean_confidence = round(
            float(np.mean([p["confidence"] for p in majority_agents])), 4
        )

        return {
            "final_label":       majority_label,
            "weighted_label":    weighted_label,
            "unanimous":         unanimous,
            "agreement_ratio":   round(agreement_ratio, 4),
            "disagreement":      disagreement,
            "mean_confidence":   mean_confidence,
            "agent_predictions": agent_preds,
            "vote_counts":       dict(vote_count)
        }


if __name__ == "__main__":

    verifier = MultiAgentVerifier()

    claim = "Vitamin C reduces the duration of cold symptoms."

    sentences = [
        "Several clinical trials show that Vitamin C reduces cold duration.",
        "Prevention evidence remains inconclusive.",
        "Vitamin C has been studied for the common cold."
    ]

    print("\nClaim:", claim)
    print("\nRunning multi-agent verification...\n")

    result = verifier.verify(claim, sentences)

    print(f"Final label    : {result['final_label']}")
    print(f"Weighted label : {result['weighted_label']}")
    print(f"Unanimous      : {result['unanimous']}")
    print(f"Agreement      : {result['agreement_ratio']:.0%}")
    print(f"Disagreement   : {result['disagreement']}")
    print(f"Vote counts    : {result['vote_counts']}")
    print(f"\nPer-agent predictions:")
    for pred in result["agent_predictions"]:
        print(f"  {pred['agent']:<12} → {pred['label']}  "
              f"(confidence: {pred['confidence']})")