"""
Phase 4: Claim Verification using NLI
------------------------------------
This module verifies a claim against evidence sentences
using a pretrained transformer-based NLI model.
"""

from transformers import pipeline


# Load NLI model once (important for performance)
nli_pipeline = pipeline(
    task="text-classification",
    model="roberta-large-mnli"
)


def verify_claim(claim, evidence_sentences, confidence_threshold=0.7):
    """
    Verify a claim using ranked evidence sentences.

    Parameters:
    ----------
    claim : str
        The input claim.

    evidence_sentences : list[str]
        Ranked evidence sentences.

    confidence_threshold : float
        Minimum confidence score to accept a prediction.

    Returns:
    -------
    dict
        {
            "verdict": "SUPPORTS / REFUTES / NOT ENOUGH INFO",
            "evidence": str or None,
            "score": float
        }
    """

    best_result = {
        "verdict": "NOT ENOUGH INFO",
        "evidence": None,
        "score": 0.0
    }

    for sentence in evidence_sentences:
        # Format input as (premise, hypothesis)
        input_text = f"{sentence} </s></s> {claim}"

        result = nli_pipeline(input_text)[0]

        label = result["label"]
        score = result["score"]

        # Map NLI labels to fact-checking labels
        if label == "ENTAILMENT" and score > best_result["score"]:
            best_result = {
                "verdict": "SUPPORTS",
                "evidence": sentence,
                "score": score
            }

        elif label == "CONTRADICTION" and score > best_result["score"]:
            best_result = {
                "verdict": "REFUTES",
                "evidence": sentence,
                "score": score
            }

    # If confidence is too low, fallback to NEI
    if best_result["score"] < confidence_threshold:
        best_result["verdict"] = "NOT ENOUGH INFO"
        best_result["evidence"] = None

    return best_result
