from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split

print("Loading SciFact datasets...")

corpus = load_dataset("allenai/scifact", "corpus")["train"]
claims = load_dataset("allenai/scifact", "claims")["train"]

corpus_dict = {
    item["doc_id"]: item["abstract"]
    for item in corpus
}

rows = []

for claim_item in claims:
    claim_text = claim_item["claim"]

    evidence_doc_id = claim_item["evidence_doc_id"]
    evidence_label = claim_item["evidence_label"]
    evidence_sentences = claim_item["evidence_sentences"]
    cited_docs = claim_item["cited_doc_ids"]

    if evidence_label != "" and evidence_doc_id != "":
        doc_id = int(evidence_doc_id)

        if doc_id in corpus_dict:
            rows.append({
                "claim": claim_text,
                "doc_id": doc_id,
                "abstract": corpus_dict[doc_id],  
                "label": evidence_label,
                "evidence_sentence_ids": evidence_sentences
            })

    else:
        if len(cited_docs) > 0:
            doc_id = cited_docs[0]

            if doc_id in corpus_dict:
                rows.append({
                    "claim": claim_text,
                    "doc_id": doc_id,
                    "abstract": corpus_dict[doc_id], 
                    "label": "NEI",
                    "evidence_sentence_ids": []
                })

df = pd.DataFrame(rows)

df = df.drop_duplicates(subset=["claim", "doc_id", "label"])

def extract_evidence_text(row):
    sentences = row["abstract"] 
    evidence_ids = row["evidence_sentence_ids"]

    if not evidence_ids:
        return ""

    selected_sentences = []

    for idx in evidence_ids:
        if idx < len(sentences):  
            selected_sentences.append(sentences[idx])

    return " ".join(selected_sentences)


df["gold_evidence_text"] = df.apply(extract_evidence_text, axis=1)

df["abstract"] = df["abstract"].apply(
    lambda x: " ".join(x).replace("  ", " ").strip()
)

print("\nFinal dataset size:", len(df))

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())

print("\nExact duplicate rows after cleaning:",df.duplicated(subset=["claim", "doc_id", "label"]).sum())

print("\nNon-empty gold evidence rows:",
      (df["gold_evidence_text"] != "").sum())

print("Non-empty gold evidence:", (df["gold_evidence_text"] != "").sum())
print("SUPPORT + CONTRADICT:", (df["label"] != "NEI").sum())

mismatch_count = 0
for _, row in df.iterrows():
    if row["label"] != "NEI" and row["gold_evidence_text"] == "":
        mismatch_count += 1
print("Evidence mismatch count:", mismatch_count)

df.to_pickle("data/processed/scifact_processed.pkl")


train_df, dev_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

train_df.to_pickle("data/processed/scifact_train.pkl")
dev_df.to_pickle("data/processed/scifact_dev.pkl")

print("Train size:", len(train_df))
print("Dev size:", len(dev_df))

print("Saved successfully to data/processed/scifact_processed.pkl")