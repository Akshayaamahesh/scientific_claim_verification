"""
Dense Retrieval Agent.

"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class DenseRetriever:

    def __init__(self, dataset_path, model_name="all-MiniLM-L6-v2"):
        print("Loading dataset...")
        df = pd.read_pickle(dataset_path)

        self.docs = df[["doc_id", "abstract"]].drop_duplicates(
            subset=["doc_id"]
        ).reset_index(drop=True)

        print(f"Total unique documents: {len(self.docs)}")
        self.abstracts = self.docs["abstract"].tolist()
        self.doc_ids   = self.docs["doc_id"].tolist()

        print(f"Loading sentence transformer: {model_name} ...")
        self.model = SentenceTransformer(model_name)

        print("Encoding documents (this may take a minute)...")
        self.doc_embeddings = self.model.encode(
            self.abstracts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        print("Dense index ready.")

    def retrieve(self, claim, top_k=5):
        claim_embedding = self.model.encode(
            [claim], convert_to_numpy=True
        )

        scores      = cosine_similarity(claim_embedding, self.doc_embeddings)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "doc_id":   self.doc_ids[idx],
                "abstract": self.abstracts[idx],
                "score":    round(float(scores[idx]), 4)
            })

        return results


if __name__ == "__main__":

    retriever = DenseRetriever(config.DATASET_PATH)

    claim = "Vitamin C reduces the duration of cold symptoms."
    results = retriever.retrieve(claim, top_k=3)

    print(f"\nClaim: {claim}\n")
    for r in results:
        print(f"Doc ID  : {r['doc_id']}")
        print(f"Score   : {r['score']}")
        print(f"Abstract: {r['abstract'][:200]}...")
        print("-" * 50)