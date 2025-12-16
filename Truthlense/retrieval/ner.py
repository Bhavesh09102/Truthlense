import spacy
from truthlense.logger import get_logger

logger = get_logger("NER")

nlp = spacy.load("en_core_web_sm")

def extract_entities(text: str):
    logger.info("Extracting entities from claim")

    doc = nlp(text)

    # 1️⃣ Try named entities first
    entities = [ent.text for ent in doc.ents]

    # 2️⃣ Fallback: if no named entities, use nouns
    if not entities:
        logger.info("No named entities found, using noun fallback")
        entities = [
            token.text
            for token in doc
            if token.pos_ in ["NOUN", "PROPN"]
        ]

    return list(set(entities))
