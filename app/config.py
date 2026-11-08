import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Load .env file
load_dotenv(BASE_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
DATABASE_PATH = BASE_DIR / "contextlens.db"
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

# Latest High-Throughput Gemini Models
VISION_MODEL_NAME = os.getenv("VISION_MODEL_NAME", "gemini-3.1-flash-lite")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "gemini-embedding-001")

# Decision & Safety Thresholds
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.70"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.65"))

# Image storage
IMAGES_DIR = BASE_DIR / "data" / "images"
