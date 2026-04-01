"""
Evidence Selector.

Selects the most relevant sentences from retrieved documents. 
"""

import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

from nltk.tokenize import sent_tokenize


class EvidenceSelector:

    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english")

    def split_sentences(self, abstract):
        return sent_tokenize(abstract)

    def select_evidence(self, claim, abstract, top_k=2):
        sentences = self.split_sentences(abstract)

        if len(sentences) == 0:
            return []

        corpus     = [claim] + sentences
        tfidf      = self.vectorizer.fit_transform(corpus)
        claim_vec  = tfidf[0]
        sent_vecs  = tfidf[1:]
        similarities = cosine_similarity(claim_vec, sent_vecs)[0]
        top_indices  = np.argsort(similarities)[::-1][:top_k]

        return [
            {"sentence": sentences[idx], "score": similarities[idx]}
            for idx in top_indices
        ]

    def select_evidence_multi(self, claim, retrieved_docs, top_k=3):
        all_sentences = []

        for rank, doc in enumerate(retrieved_docs):
            sentences = self.split_sentences(doc["abstract"])
            for sent in sentences:
                all_sentences.append({
                    "sentence": sent,
                    "doc_id":   doc["doc_id"],
                    "doc_rank": rank
                })

        if len(all_sentences) == 0:
            return []

        corpus    = [claim] + [s["sentence"] for s in all_sentences]
        tfidf     = self.vectorizer.fit_transform(corpus)
        claim_vec = tfidf[0]
        sent_vecs = tfidf[1:]
        scores    = cosine_similarity(claim_vec, sent_vecs)[0]

        for i, score in enumerate(scores):
            all_sentences[i]["score"] = round(float(score), 4)

        all_sentences.sort(key=lambda x: x["score"], reverse=True)

        return all_sentences[:top_k]


if __name__ == "__main__":

    selector = EvidenceSelector()

    claim = "Vitamin C reduces cold duration"
    abstract = """
    Vitamin C has been widely studied for treating common cold.
    Several clinical trials show that Vitamin C reduces the duration
    of cold symptoms. However, evidence about prevention remains
    inconclusive.
    """

    print("Single doc evidence:")
    for e in selector.select_evidence(claim, abstract, top_k=2):
        print(f"  [{e['score']:.3f}] {e['sentence']}")