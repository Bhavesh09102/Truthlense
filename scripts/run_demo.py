from truthlense.claim.preprocess import preprocess_claim
from truthlense.retrieval.ner import extract_entities
from truthlense.retrieval.wikipedia import retrieve_evidence
from truthlense.ranking.tfidf_ranker import rank_evidence

if __name__ == "__main__":
    claim = input("Enter a claim: ")
    claim = preprocess_claim(claim)

    entities = extract_entities(claim)
    print("\nEntities:", entities)

    evidence = retrieve_evidence(entities)
    print("\nRetrieved Evidence:")
    for e in evidence:
        print("-", e)
    ranked_evidence = rank_evidence(claim, evidence)

    print("\nTop Ranked Evidence:")
    for sent in ranked_evidence:
        print("-", sent)
