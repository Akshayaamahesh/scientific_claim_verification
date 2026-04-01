"""
Hybrid Retrieval Agent.

"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from rank_bm25 import BM25Okapi
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class HybridRetriever:

    def __init__(self, dataset_path, alpha=0.5):
        self.alpha = alpha

        print("Loading dataset...")
        df = pd.read_pickle(dataset_path)

        self.docs = df[["doc_id", "abstract"]].drop_duplicates(
            subset=["doc_id"]
        ).reset_index(drop=True)

        print(f"Total unique documents: {len(self.docs)}")
        self.abstracts = self.docs["abstract"].tolist()
        self.doc_ids   = self.docs["doc_id"].tolist()

        print("Building TF-IDF index...")
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=config.TFIDF_MAX_FEATURES
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.abstracts)

        print("Building BM25 index...")
        tokenized = [doc.lower().split() for doc in self.abstracts]
        self.bm25 = BM25Okapi(tokenized)

        print(f"Hybrid retriever ready (alpha={self.alpha}).")

    def retrieve(self, claim, top_k=5):
        claim_vec    = self.tfidf_vectorizer.transform([claim])
        tfidf_scores = cosine_similarity(claim_vec, self.tfidf_matrix)[0]

        query_tokens = claim.lower().split()
        bm25_scores  = np.array(self.bm25.get_scores(query_tokens))

        def normalize(scores):
            min_s, max_s = scores.min(), scores.max()
            if max_s - min_s == 0:
                return np.zeros_like(scores)
            return (scores - min_s) / (max_s - min_s)

        tfidf_norm = normalize(tfidf_scores)
        bm25_norm  = normalize(bm25_scores)

        hybrid_scores = self.alpha * tfidf_norm + (1 - self.alpha) * bm25_norm

        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "doc_id":      self.doc_ids[idx],
                "abstract":    self.abstracts[idx],
                "score":       round(float(hybrid_scores[idx]), 4),
                "tfidf_score": round(float(tfidf_norm[idx]), 4),
                "bm25_score":  round(float(bm25_norm[idx]), 4)
            })

        return results


if __name__ == "__main__":

    retriever = HybridRetriever(config.DATASET_PATH, alpha=0.5)

    claim = "Vitamin C reduces the duration of cold symptoms."
    results = retriever.retrieve(claim, top_k=3)

    print(f"\nClaim: {claim}\n")
    for r in results:
        print(f"Doc ID      : {r['doc_id']}")
        print(f"Hybrid score: {r['score']}  "
              f"(TF-IDF: {r['tfidf_score']}, BM25: {r['bm25_score']})")
        print(f"Abstract    : {r['abstract'][:150]}...")
        print("-" * 50)