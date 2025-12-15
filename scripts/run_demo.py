from truthlense.claim.preprocess import preprocess_claim
from truthlense.retrieval.ner import extract_entities
from truthlense.retrieval.wikipedia import retrieve_evidence

if __name__ == "__main__":
    claim = input("Enter a claim: ")
    claim = preprocess_claim(claim)

    entities = extract_entities(claim)
    print("\nEntities:", entities)

    evidence = retrieve_evidence(entities)
    print("\nRetrieved Evidence:")
    for e in evidence:
        print("-", e["sentence"])
