# ContextLens
**Explainable Multimodal Visual Asset Intelligence & Mismatch Prevention System**

ContextLens is a production-grade AI decision system that understands visual asset libraries, extracts structured metadata, and intelligently pairs the right image with the right article — **avoiding wrong recommendations when confident, and safely abstaining when uncertain**.

A blog post about red foxes surfaces a red fox photo, a similar-looking wolf image is provably rejected, and unrelated articles receive a safe "no confident match" response.

---

## Architecture Overview

ContextLens separates data storage, domain logic, HTTP controllers, and external AI services into clean layers:

```text
Images ──[Batch Job]──> Vision Model ──> Schema Validation ──> image_metadata
                              │
                              └─> embed(caption) ─────────────> image_vectors
                                                                     │
Posts  ─────────────────────────> embed(post_text) ───────────> post_vectors
                                                                     │
GET /api/posts/{id}/images ──────────────────────────────────────────┤
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

## Key Capabilities

1. **Structured Vision Ingestion**: Runs images through Gemini Flash (`gemini-3.1-flash-lite`), enforcing strict Pydantic schema validation (`subject`, `category`, `attributes`, `caption`, `confidence`).
2. **Dual-Stream Semantic Retrieval**: Embeds captions and articles into a shared 3072-dimensional vector space (`gemini-embedding-001`) with normalized cosine similarity ranking.
3. **The Mismatch Guard (Safety Layer)**:
   * **Confidence Gate**: Flags unverified or ambiguous assets (e.g. `ambiguous_shape_01.jpg` scored `0.35` $\rightarrow$ `FLAGGED_LOW_CONFIDENCE`).
   * **Category & Subject Mismatch**: Rejects conflicting candidates (e.g. forcing a gray wolf on a red fox article $\rightarrow$ `REJECTED: Animal category mismatch: expected fox, detected wolf`).
   * **Similarity Threshold Gate**: Refuses poor matches below threshold (e.g. quantum computing article $\rightarrow$ `NO_CONFIDENT_MATCH`).
4. **Human-in-the-Loop Review**: Provides validated endpoints for editorial staff to inspect explanations and submit approvals or rejections.
5. **Per-Call AI Cost Tracking**: Tracks token consumption and costs attributed per operation in `ai_cost_logs`.
6. **Benchmark-Proven Quality**: Achieves **91.67% Top-1 Precision** on a 12-item labeled evaluation dataset.

---

## Quickstart (Under 5 Minutes)

### 1. Installation

```bash
git clone https://github.com/tahirkhan05/ContextLens.git
cd ContextLens
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and insert your Gemini API key (free from [Google AI Studio](https://aistudio.google.com/)):

```bash
cp .env.example .env
```

Edit `.env`:
```ini
GEMINI_API_KEY=your_actual_gemini_api_key
```

### 3. Seed Demo Data

Populate the database with sample editorial posts, generated suggestions, and review logs:

```bash
python scripts/seed_demo.py
```

### 4. Run the API Server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive Swagger documentation is available at:  
👉 **[http://localhost:8000/docs](http://localhost:8000/docs)**

---

## Verification & Acceptance Probes

ContextLens includes automated test and verification suites matching all acceptance probes:

### 1. Run Automated Test Suite
```bash
python -m pytest tests/
```
Output:
```text
tests/test_api.py ......... [100%]
9 passed in 3.04s
```

### 2. Verify Phase 3 Probes (Probe 2, 3, 4)
```bash
python scripts/verify_phase3.py
```
* **Probe 2**: Fox post ranks red fox first (`0.7314`), while wolves and dogs rank significantly lower.
* **Probe 3**: Forcing wolf on fox post triggers `REJECTED: Animal category mismatch: expected fox, detected wolf`.
* **Probe 4**: Out-of-domain post safely triggers `NO_CONFIDENT_MATCH`.

### 3. Run Evaluation Benchmark (Probe 5)
```bash
python scripts/run_eval.py
```
Output:
```text
Total Evaluated Posts: 12
Correct Top-1 Decisions: 11
Top-1 Precision: 91.67%
```

### 4. Check AI Cost Tracking (Probe 6)
Visit `GET /api/costs` or query SQLite:
```bash
python -c "import sqlite3; conn = sqlite3.connect('contextlens.db'); print(conn.cursor().execute('SELECT operation, model, COUNT(*), SUM(estimated_cost_usd) FROM ai_cost_logs GROUP BY operation, model').fetchall())"
```

---

## Implemented Stretch Goals (Section 9)

ContextLens implements multiple production-grade stretch goals beyond the core requirements:
1. **Automatic Alt-Text Generation (`GET /api/images/{id}/alt-text`)**: Formulates accessible, WCAG-compliant descriptive alt text and standard `<img>` markup directly from extracted vision attributes and scene captions.
2. **Near-Duplicate Detection (`GET /api/images/{id}/duplicates`)**: Computes pairwise cosine distance against all corpus embeddings to identify redundant, visually similar, or duplicate assets (e.g. brown bear variants at similarity > 0.85).
3. **Comprehensive Pytest Suite (`tests/test_api.py`)**: 9 integration tests validating API routing, schema constraints, category mismatch guard, and stretch endpoints.

---

## Honest Limitations

1. **Dependence on Model Taxonomy**: The Mismatch Guard relies on accurate subject identification from the vision model. If an image is so rare or unrepresented that the vision model misidentifies the primary subject as a generic object with high false confidence, the guard may evaluate against the flawed label.
2. **Static Corpus Pre-indexing**: Embeddings are pre-generated during batch ingestion. Real-time dynamic image ingestion on large libraries requires a distributed background task queue (e.g. Celery / Redis) rather than synchronous processing.
3. **No Pixel-Level Temporal Provenance**: The system cannot infer capture dates or geolocation from raw pixel values alone without independent EXIF or cryptographic provenance headers.

---

## Submission Pack

* `capstone.yaml` — Machine-readable evaluation manifest.
* `EVIDENCE.md` — Verified console transcripts for all 10 requirements and 6 probes.
* `BUILDLOG.md` — Honest log of AI assistance, failure cases, and engineering fixes.
* `DESIGN.md` — System architecture and data models.
* `LICENSE` — MIT License.
