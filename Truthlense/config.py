from pathlib import Path

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"

# Text processing
MIN_CLAIM_LENGTH = 5
LOWERCASE = True
