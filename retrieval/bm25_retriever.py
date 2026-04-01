"""
BM25 Retrieval Agent.

"""

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class BM25Retriever:

    def __init__(self, dataset_path):
        print("Loading dataset...")
        df = pd.read_pickle(dataset_path)

        self.docs = df[["doc_id", "abstract"]].drop_duplicates(
            subset=["doc_id"]
        ).reset_index(drop=True)

        print(f"Total unique documents: {len(self.docs)}")

        self.abstracts = self.docs["abstract"].tolist()
        self.doc_ids   = self.docs["doc_id"].tolist()

        print("Building BM25 index...")
        tokenized = [doc.lower().split() for doc in self.abstracts]
        self.bm25 = BM25Okapi(tokenized)
        print("BM25 index built.")

    def retrieve(self, claim, top_k=5):
        query_tokens = claim.lower().split()
        scores       = self.bm25.get_scores(query_tokens)
        top_indices  = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "doc_id":   self.doc_ids[idx],
                "abstract": self.abstracts[idx],
                "score":    round(float(scores[idx]), 4)
            })

        return results


if __name__ == "__main__":

    retriever = BM25Retriever(config.DATASET_PATH)

    claim = "Vitamin C reduces the duration of cold symptoms."
    results = retriever.retrieve(claim, top_k=3)

    print(f"\nClaim: {claim}\n")
    for r in results:
        print(f"Doc ID : {r['doc_id']}")
        print(f"Score  : {r['score']}")
        print(f"Abstract: {r['abstract'][:200]}...")
        print("-" * 50)