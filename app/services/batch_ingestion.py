import json
import time
from pathlib import Path
from typing import Dict, Any

from app.config import IMAGES_DIR, CONFIDENCE_THRESHOLD
from app.database import get_db_connection, init_db
from app.services.ai_service import analyze_image_with_vision, generate_text_embedding


def sync_images_to_db() -> int:
    """Scan IMAGES_DIR and register any new images into the database with PENDING status."""
    conn = get_db_connection()
    cursor = conn.cursor()

    image_files = sorted(list(IMAGES_DIR.glob("*.jpg")) + list(IMAGES_DIR.glob("*.png")))
    added_count = 0

    for img_path in image_files:
        filename = img_path.name
        cursor.execute("SELECT id FROM images WHERE filename = ?", (filename,))
        if not cursor.fetchone():
            cursor.execute(
                "INSERT INTO images (filename, filepath, status) VALUES (?, ?, 'PENDING')",
                (filename, str(img_path.resolve())),
            )
            added_count += 1

    conn.commit()
    conn.close()
    return added_count


def process_single_image(image_id: int, filename: str, filepath: str) -> Dict[str, Any]:
    """Process a single image through the vision and embedding pipeline with atomic DB operations."""
    path_obj = Path(filepath)

    try:
        # Step 1: Vision analysis with schema validation (manages its own DB write for cost)
        metadata, raw_dict = analyze_image_with_vision(path_obj)

        # Step 2: Check confidence threshold
        if metadata.confidence < CONFIDENCE_THRESHOLD:
            status = "FLAGGED_LOW_CONFIDENCE"
        else:
            status = "PROCESSED"

        # Step 3: Generate embedding for the caption (manages its own DB write for cost)
        embedding_vector = generate_text_embedding(metadata.caption)

        # Step 4: Atomic DB write for metadata, embedding, and image status
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO image_metadata 
            (image_id, subject, category, attributes, caption, confidence, raw_response)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                image_id,
                metadata.subject.lower(),
                metadata.category.lower(),
                json.dumps(metadata.attributes),
                metadata.caption,
                metadata.confidence,
                json.dumps(raw_dict),
            ),
        )

        cursor.execute(
            """
            INSERT INTO embeddings (entity_type, entity_id, vector, model_name)
            VALUES ('IMAGE_CAPTION', ?, ?, 'gemini-embedding-001')
            """,
            (image_id, json.dumps(embedding_vector)),
        )

        cursor.execute("UPDATE images SET status = ? WHERE id = ?", (status, image_id))
        conn.commit()
        conn.close()

        return {
            "image_id": image_id,
            "filename": filename,
            "status": status,
            "subject": metadata.subject,
            "category": metadata.category,
            "confidence": metadata.confidence,
            "caption": metadata.caption,
        }

    except Exception as exc:
        conn = get_db_connection()
        conn.execute("UPDATE images SET status = 'FAILED' WHERE id = ?", (image_id,))
        conn.commit()
        conn.close()
        return {
            "image_id": image_id,
            "filename": filename,
            "status": "FAILED",
            "error": str(exc),
        }


def run_batch_pipeline(max_items: int = None, reset_failed: bool = True) -> Dict[str, Any]:
    """Execute the batch ingestion pipeline across pending images."""
    init_db()
    sync_images_to_db()

    conn = get_db_connection()
    if reset_failed:
        conn.execute("UPDATE images SET status = 'PENDING' WHERE status = 'FAILED'")
        conn.commit()

    cursor = conn.cursor()
    cursor.execute("SELECT id, filename, filepath FROM images WHERE status = 'PENDING' ORDER BY id ASC")
    pending = cursor.fetchall()
    conn.close()

    if max_items:
        pending = pending[:max_items]

    total = len(pending)
    print(f"\n=======================================================")
    print(f"ContextLens — Starting Batch Vision Pipeline ({total} items)")
    print(f"Confidence Gate Threshold: {CONFIDENCE_THRESHOLD}")
    print(f"=======================================================\n")

    processed = 0
    flagged = 0
    failed = 0
    start_time = time.time()

    for idx, row in enumerate(pending, 1):
        image_id, filename, filepath = row["id"], row["filename"], row["filepath"]
        print(f"[{idx}/{total}] Processing: {filename}...", end=" ", flush=True)

        result = process_single_image(image_id, filename, filepath)
        status = result.get("status")

        if status == "PROCESSED":
            processed += 1
            print(f"PROCESSED -> Subject: '{result.get('subject')}' | Confidence: {result.get('confidence'):.2f}")
        elif status == "FLAGGED_LOW_CONFIDENCE":
            flagged += 1
            print(f"FLAGGED (Low Confidence: {result.get('confidence'):.2f}) -> '{result.get('subject')}'")
        else:
            failed += 1
            print(f"FAILED -> {result.get('error')}")

        # Polite pacing between API calls
        time.sleep(0.5)

    duration = round(time.time() - start_time, 2)

    # Cost calculation summary
    conn = get_db_connection()
    cost_cursor = conn.cursor()
    cost_cursor.execute("SELECT COUNT(*), SUM(tokens_in), SUM(tokens_out), SUM(estimated_cost_usd) FROM ai_cost_logs")
    calls, tokens_in, tokens_out, total_cost = cost_cursor.fetchone()
    conn.close()

    summary = {
        "total_images": total,
        "processed": processed,
        "flagged_low_confidence": flagged,
        "failed": failed,
        "duration_seconds": duration,
        "ai_calls": calls or 0,
        "tokens_in": tokens_in or 0,
        "tokens_out": tokens_out or 0,
        "total_cost_usd": round(total_cost or 0.0, 6),
    }

    print(f"\n--- Batch Pipeline Summary ---")
    print(json.dumps(summary, indent=2))
    return summary
