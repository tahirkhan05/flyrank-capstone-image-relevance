# ContextLens — System Design Document
**Explainable Multimodal Visual Asset Intelligence & Mismatch Prevention System**

---

## 1. Problem Statement

Digital publishers, newsrooms, and editorial teams manage expanding visual asset libraries. Standard asset management tools associate images with written articles using keyword searches or raw visual resemblance. However, superficial similarity frequently causes contextual errors:

* An article discussing the ecology of **Red Foxes** is illustrated with a **Gray Wolf** simply because both are wild canines in woodland environments.
* An article describing a recent regional flood is paired with a decades-old image from an unrelated continent because both depict water submersion.
* Conventional similarity engines always return a "best" candidate even when every candidate is contextually inappropriate, forcing false recommendations.

**ContextLens** solves this problem by functioning not as a search engine, but as an **explainable multimodal decision and verification system**:
1. **Multimodal Visual Understanding**: Extracts verified structured metadata (subject, category, visual attributes, descriptive caption, and confidence score) from images.
2. **Dual-Stream Semantic Matching**: Embeds post narratives and image captions into a unified semantic space for conceptual matching.
3. **Evidence-Based Mismatch Guard**: Validates candidate pairings against explicit domain constraints, similarity thresholds, and confidence limits. It accepts strong matches, rejects contradictions with plain-language explanations, and abstains when evidence is insufficient.
4. **Human-in-the-Loop Review**: Exposes verified recommendations and transparent explanations to editorial staff for final sign-off.

---

## 2. Explicit Non-Goal

**ContextLens is not:**
* A web-scale web scraping search engine or general asset repository.
* A generative AI image synthesis or inpainting platform.
* A tool that attempts to infer unprovable metadata (such as exact camera capture timestamps or GPS coordinates from raw pixels alone).

ContextLens is strictly focused on **contextual compatibility verification, explainable mismatch rejection, and trustworthy asset-article pairing**.

---

## 3. System Architecture & Layer Sketch

The system adheres to a clean layered architecture separating data storage, domain logic, HTTP controllers, and external AI services:

```text
+---------------------------------------------------------------------------------+
|                                 HTTP Layer                                      |
|            FastAPI Endpoints: Ingestion, Retrieval, Review & Auditing           |
+---------------------------------------------------------------------------------+
                                       |
+---------------------------------------------------------------------------------+
|                             Domain / Logic Layer                                |
|  - Batch Ingestion & Retry Worker                                               |
|  - Semantic Similarity Ranker (Cosine Distance)                                |
|  - Explainable Mismatch Guard (Subject, Category & Confidence Gate)             |
|  - Cost & Token Attribution Tracker                                             |
+---------------------------------------------------------------------------------+
                 /                                           \
+----------------------------------+       +--------------------------------------+
|          Data Layer              |       |           External AI Layer          |
|  - Relational Models & Storage   |       |  - Gemini 1.5/2.0 Flash Vision       |
|  - Embeddings & Vector Index     |       |  - Semantic Text Embeddings          |
|  - Audit & Review Trails         |       |  - Schema Validation (Pydantic)      |
+----------------------------------+       +--------------------------------------+
```

### Data Flow Diagram

```text
Images ──[Batch Job]──> Vision Model ──> Schema Validation ──> image_metadata
                              │
                              └─> embed(caption) ─────────────> image_vectors
                                                                     │
Posts  ─────────────────────────> embed(post_text) ───────────> post_vectors
                                                                     │
GET /posts/{id}/images ──────────────────────────────────────────────┤
                                                                     ▼
                                                          Cosine Similarity Ranking
                                                                     │
                                                                     ▼
                                                             Mismatch Guard:
                                                       - Subject / Category match
                                                       - Confidence score >= 0.70
                                                       - Similarity >= threshold
                                                                     │
                                                ┌────────────────────┴────────────────────┐
                                                ▼                                         ▼
                                        ACCEPTED MATCH                             REJECTED / ABSTAIN
                                 (Ranked & Explained Pairing)             ("Animal category mismatch: expected
                                                │                          fox, detected wolf")
                                                ▼                                         │
                                       Review Workflow API <──────────────────────────────┘
                                  (Approve / Reject / Audit)
```

---

## 4. Data Models & Database Design

ContextLens utilizes structured relational storage with targeted indexes for fast lookup and relational integrity.

### 1. `images`
* `id` (INTEGER, Primary Key, Auto-increment)
* `filename` (TEXT, Unique, Indexed)
* `filepath` (TEXT)
* `status` (TEXT: `PENDING`, `PROCESSED`, `FLAGGED_LOW_CONFIDENCE`, `FAILED`, Indexed)
* `created_at` (TIMESTAMP)

### 2. `image_metadata`
* `image_id` (INTEGER, Foreign Key -> `images.id`, Primary Key)
* `subject` (TEXT, Indexed)
* `category` (TEXT, Indexed)
* `attributes` (JSON — array of extracted descriptive traits)
* `caption` (TEXT)
* `confidence` (FLOAT: 0.0 to 1.0)
* `raw_response` (JSON)

### 3. `embeddings`
* `id` (INTEGER, Primary Key, Auto-increment)
* `entity_type` (TEXT: `IMAGE_CAPTION`, `POST_CONTENT`, Indexed)
* `entity_id` (INTEGER, Indexed)
* `vector` (JSON / Float Array — normalized 768-dim embedding)
* `model_name` (TEXT)
* `created_at` (TIMESTAMP)

### 4. `posts`
* `id` (INTEGER, Primary Key, Auto-increment)
* `title` (TEXT)
* `content` (TEXT)
* `expected_subject` (TEXT)
* `expected_category` (TEXT)
* `created_at` (TIMESTAMP)

### 5. `suggestions`
* `id` (INTEGER, Primary Key, Auto-increment)
* `post_id` (INTEGER, Foreign Key -> `posts.id`, Indexed)
* `image_id` (INTEGER, Foreign Key -> `images.id`, Indexed)
* `similarity_score` (FLOAT)
* `guard_status` (TEXT: `ACCEPTED`, `REJECTED`, `NO_CONFIDENT_MATCH`)
* `guard_reason` (TEXT — plain-language decision rationale)
* `created_at` (TIMESTAMP)

### 6. `reviews`
* `id` (INTEGER, Primary Key, Auto-increment)
* `suggestion_id` (INTEGER, Foreign Key -> `suggestions.id`, Unique)
* `reviewer_action` (TEXT: `APPROVED`, `REJECTED`)
* `notes` (TEXT, Optional)
* `reviewed_at` (TIMESTAMP)

### 7. `ai_cost_logs`
* `id` (INTEGER, Primary Key, Auto-increment)
* `operation` (TEXT: `VISION_CLASSIFICATION`, `TEXT_EMBEDDING`)
* `model` (TEXT)
* `tokens_in` (INTEGER)
* `tokens_out` (INTEGER)
* `estimated_cost_usd` (FLOAT)
* `created_at` (TIMESTAMP)

---

## 5. Image Metadata Schema (Structured Vision Output)

All vision model responses are strictly coerced and validated against this Pydantic schema:

```json
{
  "subject": "red fox",
  "category": "animal",
  "attributes": ["orange fur", "wild", "forest"],
  "caption": "A red fox standing in a forest",
  "confidence": 0.94
}
```

* **Pydantic Model**:
  ```python
  from pydantic import BaseModel, Field

  class ImageVisionMetadata(BaseModel):
      subject: str = Field(description="Primary identified subject of the image (e.g. red fox)")
      category: str = Field(description="Broad category (e.g. animal, vehicle, landscape)")
      attributes: list[str] = Field(description="List of distinctive visual attributes")
      caption: str = Field(description="Accurate factual caption of the visual scene")
      confidence: float = Field(ge=0.0, le=1.0, description="Model self-reported certainty between 0.0 and 1.0")
  ```

---

## 6. Matching Strategy & Mismatch Guard Engine

Candidate retrieval and verification operate in three sequential stages:

### Stage A: Vector Similarity Ranking
* Query post text $T_{\text{post}}$ is converted into vector $v_{\text{post}}$.
* Candidate image captions $C_{\text{img}}$ have precomputed vectors $v_{\text{img}}$.
* Top candidates are ranked by Cosine Similarity:
  $$\text{Cosine}(u, v) = \frac{u \cdot v}{\|u\| \|v\|}$$

### Stage B: The Mismatch Guard Decision Logic
For each top candidate:
1. **Low-Confidence Filter**: If image metadata `confidence < 0.70`, the image is flagged as `UNVERIFIED_ASSET` and withheld from automatic pairing.
2. **Category & Subject Mismatch Rule**:
   * Evaluates semantic compatibility between post subject and image subject.
   * If post topic is "Red Fox" and candidate subject is "Wolf", rule triggers:
     * **Status**: `REJECTED`
     * **Reason**: `"Category/subject mismatch: expected fox, detected wolf"`
3. **Similarity Threshold Gate**:
   * If best candidate similarity $< \tau_{\text{sim}}$ (default: `0.65`):
     * **Status**: `NO_CONFIDENT_MATCH`
     * **Reason**: `"Semantic similarity (0.48) below acceptable threshold (0.65)"`
4. **Acceptance Rule**:
   * Candidate passes subject compatibility, confidence check, and clears threshold:
     * **Status**: `ACCEPTED`
     * **Reason**: `"High semantic similarity (0.87) and compatible subject (red fox)"`

---

## 7. API Surface Contract

All endpoints validate boundaries and return standard JSON responses:

* `POST /api/images/batch` — Queue batch ingestion and structured vision analysis.
* `GET /api/images` — List processed image assets with extracted tags and confidence status.
* `POST /api/posts` — Register an article/post for asset matching.
* `GET /api/posts/{id}/images` — Retrieve ranked and guard-evaluated image suggestions with explanations.
* `POST /api/reviews` — Submit editorial approval or rejection for an image suggestion.
* `GET /api/costs` — Retrieve total token consumption and attributed AI costs.
* `GET /api/eval` — Execute evaluation benchmark and report Top-1 Precision on the labeled dataset.

---

## 8. Corpus & Evaluation Plan

* **Corpus Size**: 50 curated high-resolution, CC0/free-license images across 5 distinct animal classes:
  1. Red Fox (10 images)
  2. Gray Wolf (10 images)
  3. Domestic Dog (10 images)
  4. Brown Bear (10 images)
  5. White-tailed Deer (10 images)
  * Includes 1 intentionally low-contrast / ambiguous image to rigorously test low-confidence flagging.
* **Evaluation Dataset**: 10 labeled editorial posts with ground-truth target images, measuring **Top-1 Precision**:
  $$\text{Top-1 Precision} = \frac{\text{Correct First Suggestions}}{\text{Total Test Posts}}$$
