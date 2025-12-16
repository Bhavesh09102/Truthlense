from truthlense.claim.preprocess import preprocess_claim
from truthlense.retrieval.ner import extract_entities
from truthlense.retrieval.wikipedia import retrieve_evidence
from truthlense.ranking.sbert_ranker import rank_evidence
from truthlense.verification.verifier import verify_claim


if __name__ == "__main__":
    # -------- Phase 1: Claim preprocessing --------
    claim = input("Enter a claim: ")
    claim = preprocess_claim(claim)

    # -------- Phase 2: Evidence retrieval --------
    entities = extract_entities(claim)
    print("\nEntities:", entities)

    evidence = retrieve_evidence(entities)
    print("\nRetrieved Evidence:")
    for e in evidence:
        print("-", e)

    # -------- Phase 3: Evidence ranking --------
    ranked_evidence = rank_evidence(claim, evidence)

    print("\nTop Ranked Evidence:")
    for sent in ranked_evidence:
        print("-", sent)

    # -------- Phase 4: Claim verification --------
    result = verify_claim(claim, ranked_evidence)

    print("\nFINAL VERDICT")
    print("Verdict   :", result["verdict"])
    print("Confidence:", round(result["score"], 3))

    if result["evidence"]:
        print("Evidence  :", result["evidence"])
