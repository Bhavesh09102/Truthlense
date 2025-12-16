"""
Phase 3: Evidence Ranking
-------------------------
This module ranks retrieved evidence sentences based on their relevance
to the input claim using TF-IDF and cosine similarity.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def rank_evidence(claim, evidence_sentences, top_k=3):
    """
    Rank evidence sentences based on similarity to the claim.

    Parameters:
    ----------
    claim : str
        The input claim provided by the user.

    evidence_sentences : list[str]
        List of evidence sentences retrieved from Wikipedia.

    top_k : int
        Number of top-ranked sentences to return.

    Returns:
    -------
    list[str]
        Top-K most relevant evidence sentences.
    """

    # Safety check: if no evidence, return empty list
    if not evidence_sentences:
        return []

    # Combine claim and evidence into a single list
    # TF-IDF needs all text together to build vocabulary
    texts = [claim] + evidence_sentences

    # Initialize TF-IDF vectorizer
    # stop_words="english" removes common words like "is", "the", etc.
    vectorizer = TfidfVectorizer(stop_words="english")

    # Convert text into TF-IDF vectors
    tfidf_matrix = vectorizer.fit_transform(texts)

    # First vector corresponds to the claim
    claim_vector = tfidf_matrix[0]

    # Remaining vectors correspond to evidence sentences
    evidence_vectors = tfidf_matrix[1:]

    # Compute cosine similarity between claim and each evidence sentence
    similarities = cosine_similarity(claim_vector, evidence_vectors)[0]

    # Get indices of sentences sorted by similarity (highest first)
    ranked_indices = similarities.argsort()[::-1]

    # Select top-K ranked sentences
    ranked_evidence = [
        evidence_sentences[i] for i in ranked_indices[:top_k]
    ]

    return ranked_evidence
