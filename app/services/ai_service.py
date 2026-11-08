import json
import time
from pathlib import Path
from typing import Optional, Tuple, List
from PIL import Image
from google import genai
from google.genai import types

from app.config import (
    GEMINI_API_KEY,
    VISION_MODEL_NAME,
    EMBEDDING_MODEL_NAME,
)
from app.database import get_db_connection
from app.schemas import ImageVisionMetadata

# Pricing constants for cost attribution (USD per 1,000,000 tokens)
FLASH_INPUT_COST_PER_M = 0.075
FLASH_OUTPUT_COST_PER_M = 0.300
EMBEDDING_COST_PER_M = 0.020

# Active fallback vision models prioritized for high speed and low latency
VISION_MODEL_CANDIDATES = [
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
    "gemini-3.7-flash",
    "gemini-flash-latest",
]


def get_genai_client() -> genai.Client:
    """Instantiate a Google GenAI client with explicit timeout."""
    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY is not set. Please add your Gemini API key to .env file."
        )
    return genai.Client(
        api_key=GEMINI_API_KEY,
        http_options=types.HttpOptions(timeout=20000)
    )


def record_cost(operation: str, model: str, tokens_in: int, tokens_out: int, cost_usd: float):
    """Log AI usage and attributed cost to the database."""
    conn = get_db_connection()
    conn.execute(
        """
        INSERT INTO ai_cost_logs (operation, model, tokens_in, tokens_out, estimated_cost_usd)
        VALUES (?, ?, ?, ?, ?)
        """,
        (operation, model, tokens_in, tokens_out, cost_usd),
    )
    conn.commit()
    conn.close()


def analyze_image_with_vision(image_path: Path) -> Tuple[ImageVisionMetadata, dict]:
    """
    Run Gemini Vision on an image with structured output validation against ImageVisionMetadata.
    Iterates through candidate models if rate-limited, retries, and logs per-call token costs.
    """
    client = get_genai_client()
    pil_image = Image.open(image_path).convert("RGB")

    prompt = (
        "You are an expert visual asset intelligence assistant. Analyze this image thoroughly and "
        "extract structured factual metadata. Identify the primary subject (e.g. 'red fox', 'gray wolf', 'domestic dog', 'brown bear', 'deer'), "
        "the broad category (e.g. 'animal', 'abstract'), key visual attributes (e.g. ['orange fur', 'pointed ears', 'snow']), "
        "and a concise descriptive caption. "
        "For the confidence field (0.0 to 1.0): report your certainty in identifying a clear, specific real-world subject. "
        "If the image is an abstract texture, blurry gradient, ambiguous silhouette, or lacks a distinct identifiable subject, "
        "assign a low confidence score strictly between 0.20 and 0.50."
    )

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=ImageVisionMetadata,
        temperature=0.1,
    )

    last_error = None
    models_to_try = list(dict.fromkeys(VISION_MODEL_CANDIDATES))

    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[pil_image, prompt],
                config=config,
            )

            # Extract token usage
            tokens_in = getattr(response.usage_metadata, "prompt_token_count", 258)
            tokens_out = getattr(response.usage_metadata, "candidates_token_count", 50)

            cost_usd = (tokens_in * FLASH_INPUT_COST_PER_M / 1_000_000) + (
                tokens_out * FLASH_OUTPUT_COST_PER_M / 1_000_000
            )

            # Validate against Pydantic schema
            parsed_metadata = ImageVisionMetadata.model_validate_json(response.text)

            # Record cost
            record_cost("VISION_CLASSIFICATION", model_name, tokens_in, tokens_out, cost_usd)

            raw_dict = json.loads(response.text)
            return parsed_metadata, raw_dict

        except Exception as exc:
            last_error = exc
            err_str = str(exc)
            time.sleep(1.0)
            continue

    raise RuntimeError(f"Vision analysis failed across all models for {image_path.name}: {last_error}")


def generate_text_embedding(text: str) -> List[float]:
    """
    Generate a semantic embedding vector using Gemini Embedding (gemini-embedding-001).
    Logs token usage and cost.
    """
    client = get_genai_client()

    max_attempts = 2
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.models.embed_content(
                model=EMBEDDING_MODEL_NAME,
                contents=text,
            )

            if hasattr(response, "embeddings") and response.embeddings:
                vector = response.embeddings[0].values
            elif hasattr(response, "embedding") and response.embedding:
                vector = response.embedding.values
            else:
                raise ValueError("Could not extract embedding values from response")

            # Embedding token estimation (approx 1 token per 4 chars)
            tokens_in = max(1, len(text) // 4)
            cost_usd = tokens_in * EMBEDDING_COST_PER_M / 1_000_000

            record_cost("TEXT_EMBEDDING", EMBEDDING_MODEL_NAME, tokens_in, 0, cost_usd)
            return list(vector)

        except Exception as exc:
            if attempt < max_attempts:
                time.sleep(1.0)
                continue
            raise RuntimeError(f"Embedding generation failed: {exc}") from exc
