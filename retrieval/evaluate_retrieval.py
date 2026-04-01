"""
Retrieval evaluation — compares TF-IDF, BM25, Hybrid, Dense, HybridDense.
Reports Recall@K for all methods.

"""

import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.tfidf_retriever       import TFIDFRetriever
from retrieval.bm25_retriever        import BM25Retriever
from retrieval.hybrid_retriever      import HybridRetriever
from retrieval.dense_retriever       import DenseRetriever
from retrieval.hybrid_dense_retriever import HybridDenseRetriever
import config


def recall_at_k(df, retriever, k):
    hits = 0
    for _, row in df.iterrows():
        results       = retriever.retrieve(row["claim"], top_k=k)
        retrieved_ids = [r["doc_id"] for r in results]
        if row["doc_id"] in retrieved_ids:
            hits += 1
    return hits / len(df)


if __name__ == "__main__":

    df = pd.read_pickle(config.DATASET_PATH)

    print("Loading all retrievers...\n")
    tfidf        = TFIDFRetriever(config.DATASET_PATH)
    bm25         = BM25Retriever(config.DATASET_PATH)
    hybrid       = HybridRetriever(config.DATASET_PATH, alpha=0.5)
    dense        = DenseRetriever(config.DATASET_PATH)
    hybrid_dense = HybridDenseRetriever(config.DATASET_PATH)

    print("\n" + "=" * 65)
    print("RETRIEVAL COMPARISON")
    print("=" * 65)
    print(f"{'K':<6} {'TF-IDF':>8} {'BM25':>8} {'Hybrid':>8} "
          f"{'Dense':>8} {'H+Dense':>8}")
    print("-" * 50)

    for k in [1, 5, 10]:
        t  = recall_at_k(df, tfidf,        k)
        b  = recall_at_k(df, bm25,         k)
        h  = recall_at_k(df, hybrid,       k)
        d  = recall_at_k(df, dense,        k)
        hd = recall_at_k(df, hybrid_dense, k)
        print(f"@{k:<5} {t:>8.3f} {b:>8.3f} {h:>8.3f} "
              f"{d:>8.3f} {hd:>8.3f}")

    print("-" * 50)
    print("\nRetrieval evaluation complete.")