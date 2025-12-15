import spacy
from truthlense.logger import get_logger

logger = get_logger("NER")

nlp = spacy.load("en_core_web_sm")

def extract_entities(text: str):
    logger.info("Extracting entities from claim")
    doc = nlp(text)
    return list(set(ent.text for ent in doc.ents))
