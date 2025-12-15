import re
from truthlense.config import LOWERCASE, MIN_CLAIM_LENGTH
from truthlense.logger import get_logger

logger = get_logger("ClaimPreprocessor")

def normalize_text(text: str) -> str:
    logger.info("Normalizing claim text")

    text = text.strip()

    if LOWERCASE:
        text = text.lower()

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    return text


def validate_claim(text: str) -> bool:
    logger.info("Validating claim length")

    if len(text) < MIN_CLAIM_LENGTH:
        logger.warning("Claim too short")
        return False

    return True


def preprocess_claim(raw_claim: str) -> str:
    logger.info("Preprocessing claim")

    normalized = normalize_text(raw_claim)

    if not validate_claim(normalized):
        raise ValueError("Invalid claim: too short or empty")

    return normalized
