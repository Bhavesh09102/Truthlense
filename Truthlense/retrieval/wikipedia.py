import wikipediaapi
from truthlense.logger import get_logger
from truthlense.config import CACHE_DIR

logger = get_logger("WikipediaRetriever")


wiki = wikipediaapi.Wikipedia(
    language="en",
    user_agent="TruthLense/1.0 (educational project)"
)


def get_page(title: str):
    page = wiki.page(title)
    if not page.exists():
        logger.warning(f"Wikipedia page not found: {title}")
        return None
    return page

def extract_sentences(text: str, max_sentences=5):
    sentences = text.split(". ")
    return sentences[:max_sentences]

def retrieve_evidence(entities, max_sentences=5):
    logger.info("Retrieving evidence from Wikipedia")
    evidence = []

    for entity in entities:
        page = get_page(entity)
        if page:
            sentences = extract_sentences(page.summary, max_sentences)
            for sent in sentences:
                evidence.append({
                    "entity": entity,
                    "sentence": sent,
                    "source": page.fullurl
                })

    return evidence
