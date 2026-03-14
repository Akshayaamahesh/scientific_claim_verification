import pandas as pd

from tfidf_retriever import TFIDFRetriever


def compute_recall_at_k(df, retriever, k):

    success = 0

    for _, row in df.iterrows():

        claim = row["claim"]
        gold_doc = row["doc_id"]

        results = retriever.retrieve(claim, top_k=k)

        retrieved_docs = [r["doc_id"] for r in results]

        if gold_doc in retrieved_docs:
            success += 1

    recall = success / len(df)

    return recall


if __name__ == "__main__":

    dataset_path = "data/processed/scifact_processed.pkl"

    df = pd.read_pickle(dataset_path)

    retriever = TFIDFRetriever(dataset_path)

    print("\nEvaluating retrieval...\n")

    for k in [1, 5, 10]:

        recall = compute_recall_at_k(df, retriever, k)

        print(f"Recall@{k}: {recall:.3f}")

    all_results = []

    for _, row in df.iterrows():

        claim = row["claim"]
        gold_doc = row["doc_id"]

        results = retriever.retrieve(claim, top_k=5)

        all_results.append({
            "claim": claim,
            "gold_doc": gold_doc,
            "retrieved_docs": [r["doc_id"] for r in results]
        })

    results_df = pd.DataFrame(all_results)

    results_df.to_pickle("data/retrieval_results.pkl")

    print("\nRetrieval results saved to data/retrieval_results.pkl")