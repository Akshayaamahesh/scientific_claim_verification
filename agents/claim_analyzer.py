"""
Claim Analyzer Agent.

Classifies the TYPE of a scientific claim (numeric, comparative,
causal, general) and extracts key entities. 

"""

import re


class ClaimAnalyzer:

    NUMERIC_PATTERNS = [
        r'\d+\.?\d*\s*%',         
        r'\d+\.?\d*-fold',         
        r'p\s*[<=>]\s*0\.\d+',  
        r'\d+\s*(mg|kg|ml|mm|cm)', 
    ]

    COMPARATIVE_KEYWORDS = [
        "greater than", "less than", "higher", "lower",
        "more than", "fewer than", "outperforms", "compared to",
        "superior", "inferior", "versus", "vs"
    ]

    CAUSAL_KEYWORDS = [
        "causes", "leads to", "results in", "induces",
        "prevents", "reduces", "increases", "decreases",
        "associated with", "linked to"
    ]

    def analyze(self, claim: str) -> dict:
        claim_lower = claim.lower()

        numbers = re.findall(r'\d+\.?\d*', claim)
        has_numbers = len(numbers) > 0

        numeric_match = any(
            re.search(p, claim_lower) for p in self.NUMERIC_PATTERNS
        )

        if numeric_match or (has_numbers and len(numbers) >= 1):
            claim_type = "numeric"
        elif any(kw in claim_lower for kw in self.COMPARATIVE_KEYWORDS):
            claim_type = "comparative"
        elif any(kw in claim_lower for kw in self.CAUSAL_KEYWORDS):
            claim_type = "causal"
        else:
            claim_type = "general"

        entities = re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*', claim)

        return {
            "claim_type": claim_type,
            "has_numbers": has_numbers,
            "numbers": numbers,
            "entities": entities
        }


if __name__ == "__main__":

    analyzer = ClaimAnalyzer()

    test_claims = [
        "Vitamin C reduces cold duration by 42%.",
        "Model A outperforms Model B on all benchmarks.",
        "Smoking causes lung cancer.",
        "0-dimensional biomaterials lack inductive properties."
    ]

    for claim in test_claims:
        result = analyzer.analyze(claim)
        print(f"\nClaim : {claim}")
        print(f"Result: {result}")