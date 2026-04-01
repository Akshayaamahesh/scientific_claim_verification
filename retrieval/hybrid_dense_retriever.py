"""
Hybrid Dense Retriever.

Combines TF-IDF, BM25, and Dense scores using a weighted linear combination.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class HybridDenseRetriever:

    def __init__(self, dataset_path, alpha=0.33, beta=0.33, gamma=0.34):
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma

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
        tokenized  = [doc.lower().split() for doc in self.abstracts]
        self.bm25  = BM25Okapi(tokenized)

        print("Loading sentence transformer...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        print("Encoding documents...")
        self.doc_embeddings = self.model.encode(
            self.abstracts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        print(f"Hybrid Dense retriever ready "
              f"(TF-IDF:{alpha} BM25:{beta} Dense:{gamma})")

    def retrieve(self, claim, top_k=5):
        def normalize(scores):
            min_s, max_s = scores.min(), scores.max()
            if max_s - min_s == 0:
                return np.zeros_like(scores)
            return (scores - min_s) / (max_s - min_s)

        claim_vec    = self.tfidf_vectorizer.transform([claim])
        tfidf_scores = sklearn_cosine(claim_vec, self.tfidf_matrix)[0]

        bm25_scores  = np.array(
            self.bm25.get_scores(claim.lower().split())
        )

        claim_emb    = self.model.encode([claim], convert_to_numpy=True)
        dense_scores = sklearn_cosine(claim_emb, self.doc_embeddings)[0]

        tfidf_norm  = normalize(tfidf_scores)
        bm25_norm   = normalize(bm25_scores)
        dense_norm  = normalize(dense_scores)

        combined = (self.alpha * tfidf_norm +
                    self.beta  * bm25_norm  +
                    self.gamma * dense_norm)

        top_indices = np.argsort(combined)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "doc_id":        self.doc_ids[idx],
                "abstract":      self.abstracts[idx],
                "score":         round(float(combined[idx]), 4),
                "tfidf_score":   round(float(tfidf_norm[idx]), 4),
                "bm25_score":    round(float(bm25_norm[idx]), 4),
                "dense_score":   round(float(dense_norm[idx]), 4)
            })

        return results


if __name__ == "__main__":

    retriever = HybridDenseRetriever(config.DATASET_PATH)

    claim = "Vitamin C reduces the duration of cold symptoms."
    results = retriever.retrieve(claim, top_k=3)

    print(f"\nClaim: {claim}\n")
    for r in results:
        print(f"Doc ID      : {r['doc_id']}")
        print(f"Score       : {r['score']} "
              f"(TF-IDF:{r['tfidf_score']} "
              f"BM25:{r['bm25_score']} "
              f"Dense:{r['dense_score']})")
        print(f"Abstract    : {r['abstract'][:150]}...")
        print("-" * 50)