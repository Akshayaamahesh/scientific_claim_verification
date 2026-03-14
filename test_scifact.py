import pandas as pd

df = pd.read_pickle("data/processed/scifact_processed.pkl")

print(df.sample(3)[["label", "gold_evidence_text"]])

    