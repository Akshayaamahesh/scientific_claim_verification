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

        sentences = sent_tokenize(abstract)

        return sentences


    def select_evidence(self, claim, abstract, top_k=2):

        sentences = self.split_sentences(abstract)

        if len(sentences) == 0:
            return []

        corpus = [claim] + sentences

        tfidf = self.vectorizer.fit_transform(corpus)

        claim_vec = tfidf[0]
        sent_vecs = tfidf[1:]

        similarities = cosine_similarity(claim_vec, sent_vecs)[0]

        top_indices = np.argsort(similarities)[::-1][:top_k]

        evidence = []

        for idx in top_indices:

            evidence.append({
                "sentence": sentences[idx],
                "score": similarities[idx]
            })

        return evidence
    
if __name__ == "__main__":

    claim = "Vitamin C reduces cold duration"

    abstract = """
    Vitamin C has been widely studied for treating common cold.
    Several clinical trials show that Vitamin C reduces the duration of cold symptoms.
    However, evidence about prevention remains inconclusive.
    """

    selector = EvidenceSelector()

    evidence = selector.select_evidence(claim, abstract, top_k=2)

    print("\nClaim:", claim)

    print("\nSelected Evidence:\n")

    for e in evidence:
        print("Sentence:", e["sentence"])
        print("Score:", e["score"])
        print("-" * 40)