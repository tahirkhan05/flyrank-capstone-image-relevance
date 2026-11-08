from typing import Optional, Dict, Any
from pydantic import BaseModel
from app.config import CONFIDENCE_THRESHOLD, SIMILARITY_THRESHOLD


class GuardDecision(BaseModel):
    status: str  # ACCEPTED, REJECTED, NO_CONFIDENT_MATCH
    reason: str
    similarity_score: float
    confidence_score: float


# Canonical taxonomies for animal category/subject alignment
CANONICAL_SUBJECTS = {
    "fox": ["fox", "vulpes", "red fox", "cape fox", "kit", "vixen"],
    "wolf": ["wolf", "gray wolf", "canis lupus", "timber wolf"],
    "dog": ["dog", "golden retriever", "retriever", "puppy", "domestic dog", "hound", "canine"],
    "bear": ["bear", "brown bear", "ursus", "grizzly", "bear cub"],
    "deer": ["deer", "red deer", "stag", "doe", "fawn", "cervus", "elk"],
}


def normalize_subject(text: str) -> Optional[str]:
    """Identify the canonical category key from free text."""
    lower = text.lower()
    for canonical, synonyms in CANONICAL_SUBJECTS.items():
        for syn in synonyms:
            if syn in lower:
                return canonical
    return None


def evaluate_pairing_guard(
    post_subject: Optional[str],
    image_metadata: Dict[str, Any],
    similarity_score: float,
) -> GuardDecision:
    """
    Evaluate candidate image pairing against the 4 Mismatch Guard safety rules:
    1. Low Confidence Gate: Reject if vision certainty < threshold.
    2. Category/Subject Conflict: Reject if expected and detected subjects contradict.
    3. Similarity Threshold Gate: Refuse with 'NO_CONFIDENT_MATCH' if similarity < threshold.
    4. Acceptance Rule: Accept pairing when confident and compatible.
    """
    confidence = float(image_metadata.get("confidence", 0.0))
    img_subject = str(image_metadata.get("subject", "")).lower()

    # Rule 1: Confidence Gate
    if confidence < CONFIDENCE_THRESHOLD:
        return GuardDecision(
            status="REJECTED",
            reason=f"Low confidence classification ({confidence:.2f}): image subject is unverified or ambiguous",
            similarity_score=similarity_score,
            confidence_score=confidence,
        )

    # Rule 2: Category / Subject Mismatch Gate
    if post_subject:
        norm_post = normalize_subject(post_subject)
        norm_img = normalize_subject(img_subject)

        if norm_post and norm_img and norm_post != norm_img:
            return GuardDecision(
                status="REJECTED",
                reason=f"Animal category mismatch: expected {norm_post}, detected {norm_img}",
                similarity_score=similarity_score,
                confidence_score=confidence,
            )

    # Rule 3: Semantic Similarity Threshold Gate
    if similarity_score < SIMILARITY_THRESHOLD:
        return GuardDecision(
            status="NO_CONFIDENT_MATCH",
            reason=f"Semantic similarity ({similarity_score:.2f}) below threshold ({SIMILARITY_THRESHOLD:.2f})",
            similarity_score=similarity_score,
            confidence_score=confidence,
        )

    # Rule 4: Match Accepted
    return GuardDecision(
        status="ACCEPTED",
        reason=f"Strong semantic match ({similarity_score:.2f}) and verified subject match ('{img_subject}')",
        similarity_score=similarity_score,
        confidence_score=confidence,
    )
