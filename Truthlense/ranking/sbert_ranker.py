"""
Phase 3 (Upgraded): SBERT-based Evidence Ranking
------------------------------------------------
Ranks evidence sentences based on semantic similarity
with the claim using Sentence-BERT.
"""

from sentence_transformers import SentenceTransformer, util

# Load model once (important for performance)
model = SentenceTransformer("all-MiniLM-L6-v2")


def rank_evidence(claim, evidence_sentences, top_k=3):
    """
    Rank evidence sentences using SBERT semantic similarity.

    Parameters:
    ----------
    claim : str
        Input claim.

    evidence_sentences : list[str]
        Retrieved evidence sentences.

    top_k : int
        Number of top sentences to return.

    Returns:
    -------
    list[str]
        Top-K semantically relevant evidence sentences.
    """

    if not evidence_sentences:
        return []

    # Encode claim and evidence
    claim_embedding = model.encode(claim, convert_to_tensor=True)
    evidence_embeddings = model.encode(
        evidence_sentences, convert_to_tensor=True
    )

    # Compute cosine similarity
    similarities = util.cos_sim(
        claim_embedding, evidence_embeddings
    )[0]

    # Get indices of top-k similarities
    top_indices = similarities.argsort(descending=True)[:top_k]

    # Select ranked evidence
    ranked_evidence = [
        evidence_sentences[idx] for idx in top_indices
    ]

    return ranked_evidence
