# import spacy
# from truthlense.logger import get_logger

# logger = get_logger("NER")

# nlp = spacy.load("en_core_web_sm")

# def extract_entities(text: str):
#     logger.info("Extracting entities from claim")

#     doc = nlp(text)

#     # 1️⃣ Try named entities first
#     entities = [ent.text for ent in doc.ents]

#     # 2️⃣ Fallback: if no named entities, use nouns
#     if not entities:
#         logger.info("No named entities found, using noun fallback")
#         entities = [
#             token.text
#             for token in doc
#             if token.pos_ in ["NOUN", "PROPN"]
#         ]

#     return list(set(entities))


import spacy

nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    doc = nlp(text)

    entities = set()

    # Named entities (India, Einstein)
    for ent in doc.ents:
        entities.add(ent.text)

    # Noun phrases (water, 100 degrees Celsius)
    for chunk in doc.noun_chunks:
        entities.add(chunk.text)

    # Important standalone tokens
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN", "NUM"]:
            entities.add(token.text)

    return list(entities)
