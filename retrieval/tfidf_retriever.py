"""
TD-IDF Retrieval Agent.

"""

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TFIDFRetriever:

    def __init__(self, dataset_path):
      
        print("Loading dataset...")
        self.df = pd.read_pickle(dataset_path)

        self.docs = self.df[["doc_id", "abstract"]].drop_duplicates()

        print("Total unique documents:", len(self.docs))

        self.abstracts = self.docs["abstract"].tolist()
        self.doc_ids = self.docs["doc_id"].tolist()

        print("Building TF-IDF index...")

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=50000
        )

        self.doc_vectors = self.vectorizer.fit_transform(self.abstracts)

        print("TF-IDF index built.")

    def retrieve(self, claim, top_k=5):

        claim_vec = self.vectorizer.transform([claim])

        similarities = cosine_similarity(claim_vec, self.doc_vectors)[0]

        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []

        for idx in top_indices:
            results.append({
                "doc_id": self.doc_ids[idx],
                "abstract": self.abstracts[idx],
                "score": similarities[idx]
            })

        return results

if __name__ == "__main__":

    retriever = TFIDFRetriever(
        "data/processed/scifact_processed.pkl"
    )

    test_claim = "Vitamin C reduces cold duration"

    results = retriever.retrieve(test_claim, top_k=3)

    print("\nClaim:", test_claim)

    print("\nTop results:\n")

    for r in results:
        print("Doc ID:", r["doc_id"])
        print("Score:", r["score"])
        print("Abstract:", r["abstract"][:200])
        print("-" * 50)