import pandas as pd

from retrieval.tfidf_retriever import TFIDFRetriever
from evidence.evidence_selector import EvidenceSelector


dataset_path = "data/processed/scifact_processed.pkl"

df = pd.read_pickle(dataset_path)

retriever = TFIDFRetriever(dataset_path)

selector = EvidenceSelector()

results = []

for _, row in df.iterrows():

    claim = row["claim"]

    retrieved_docs = retriever.retrieve(claim, top_k=3)

    for doc in retrieved_docs:

        evidence = selector.select_evidence(
            claim,
            doc["abstract"],
            top_k=2
        )

        results.append({
            "claim": claim,
            "doc_id": doc["doc_id"],
            "evidence": evidence
        })


results_df = pd.DataFrame(results)

results_df.to_pickle("data/evidence_results.pkl")

print("Evidence results saved.")

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 200)
print(results_df.shape)
print(results_df.head(10))