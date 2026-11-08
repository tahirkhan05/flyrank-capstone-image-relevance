from typing import Optional, List
from pydantic import BaseModel, Field


class ImageVisionMetadata(BaseModel):
    """Structured output schema enforced on Gemini Vision responses."""
    subject: str = Field(description="Primary identified subject of the image (e.g. red fox, gray wolf, dog)")
    category: str = Field(description="Broad category classification (e.g. animal, vehicle, landscape)")
    attributes: List[str] = Field(description="List of distinctive visual attributes (e.g. ['orange fur', 'pointed ears', 'snow'])")
    caption: str = Field(description="Factual, accurate description of the visual scene")
    confidence: float = Field(ge=0.0, le=1.0, description="Self-reported certainty score between 0.0 and 1.0")


class ImageRecordResponse(BaseModel):
    id: int
    filename: str
    filepath: str
    status: str
    created_at: str
    metadata: Optional[ImageVisionMetadata] = None


class PostCreate(BaseModel):
    title: str = Field(..., min_length=2, description="Article title")
    content: str = Field(..., min_length=10, description="Full article or paragraph content")
    expected_subject: Optional[str] = Field(None, description="Expected subject topic (e.g. 'red fox')")
    expected_category: Optional[str] = Field(None, description="Expected broad category (e.g. 'animal')")


class PostResponse(BaseModel):
    id: int
    title: str
    content: str
    expected_subject: Optional[str]
    expected_category: Optional[str]
    created_at: str


class SuggestionResponse(BaseModel):
    id: int
    post_id: int
    image_id: int
    image_filename: str
    image_caption: str
    image_subject: str
    similarity_score: float
    guard_status: str  # ACCEPTED, REJECTED, NO_CONFIDENT_MATCH
    guard_reason: str


class ReviewCreate(BaseModel):
    suggestion_id: int
    reviewer_action: str = Field(..., pattern="^(APPROVED|REJECTED)$")
    notes: Optional[str] = None


class ReviewResponse(BaseModel):
    id: int
    suggestion_id: int
    reviewer_action: str
    notes: Optional[str]
    reviewed_at: str


class CostSummary(BaseModel):
    total_calls: int
    total_tokens_in: int
    total_tokens_out: int
    total_estimated_cost_usd: float
    breakdown_by_operation: dict


class AltTextResponse(BaseModel):
    image_id: int
    filename: str
    alt_text: str
    subject: str
    category: str
    attributes: List[str]
    wcag_compliant: bool
    suggested_html: str


class DuplicateItem(BaseModel):
    image_id: int
    filename: str
    similarity: float
    is_duplicate_candidate: bool


class DuplicateCheckResponse(BaseModel):
    image_id: int
    filename: str
    subject: str
    threshold: float
    potential_duplicates_count: int
    duplicates: List[DuplicateItem]


