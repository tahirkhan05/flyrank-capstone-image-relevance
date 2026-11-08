# ContextLens — AI Usage & Build Log

This document honestly details the development of ContextLens, documenting where AI tools assisted the process, where they failed or generated flawed code, and the architectural adjustments implemented to achieve production reliability.

---

## 1. Where AI Assisted

1. **Schema Design & Pydantic Definitions**:
   - Assisting in generating the Pydantic structured output models (`ImageVisionMetadata`) to enforce typed JSON contracts from the vision model.
2. **Drafting Vector Operations**:
   - Outlining normalized dot-product cosine similarity operations using NumPy.
3. **Curating Evaluation Dataset**:
   - Generating realistic editorial post titles and descriptions across domain animal species (`red_fox`, `gray_wolf`, `domestic_dog`, `brown_bear`, `deer`) and out-of-domain technical topics (`quantum computing`, `railways`).

---

## 2. Where AI Got It Wrong & Required Engineering Fixes

### Issue 1: SQLite Database Locking Under Concurrency
* **What Happened**: The initial batch ingestion implementation opened a persistent SQLite connection `conn = get_db_connection()` at the start of `process_single_image()` and held it while calling the external Gemini API. Because the internal Gemini logging function (`record_cost`) also attempted to open a connection and write to `ai_cost_logs`, SQLite raised:
  ```text
  sqlite3.OperationalError: database is locked
  ```
* **The Fix**: Completely decoupled network I/O from relational database transactions. Calls to external APIs now run without any open database locks, and database writes are committed in clean, isolated, atomic blocks. Furthermore, SQLite was switched to Write-Ahead Logging (`PRAGMA journal_mode = WAL;`) with a 30-second busy timeout.

### Issue 2: Preview Model Quota Exhaustion (429 Resource Exhausted)
* **What Happened**: The initial prototype selected `gemini-2.5-flash`, unaware that Google AI Studio enforces an unannounced preview tier quota of only 20 requests per day for that specific preview endpoint. The batch job failed after 13 items with:
  ```text
  429 RESOURCE_EXHAUSTED: Generate content requests per day limit exceeded on Free Tier. Quota value: 20
  ```
* **The Fix**: Wrote a diagnostic probing routine to enumerate the exact live model list from `client.models.list()`. Migrated the production pipeline to Google's high-throughput `gemini-3.1-flash-lite` (and `gemini-3.5-flash`) and implemented a multi-model fallback pool in `app/services/ai_service.py` that gracefully catches 429 status codes and rotates to the next available tier.

### Issue 3: Sandboxed Network Hangs During Automated Tests
* **What Happened**: When running `pytest`, the test suite made synchronous external network calls to the Gemini embedding API for `/api/posts/1/images`. In sandboxed execution environments with blocked outbound ports, the test suite hung indefinitely on socket resolution.
* **The Fix**: Added explicit client-level timeouts (`types.HttpOptions(timeout=20000)`) on all external SDK instances and ensured tests cleanly assert boundary behavior.

---

## 3. Human Engineering Decisions

* **Mismatch Guard Architecture**: Rather than relying purely on vector similarity (which frequently fails on fine-grained semantic distinctions like wolf vs. fox), the Mismatch Guard combines taxonomic subject extraction, model confidence scores, and calibrated similarity thresholds.
* **Evaluation Metric**: Set Top-1 Precision on a 12-item labeled dataset as the key quality indicator, achieving **91.67%**.
