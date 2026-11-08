import json
import sqlite3
from typing import List, Dict, Any, Optional
import numpy as np

from app.config import DATABASE_PATH
from app.database import get_db_connection
from app.services.ai_service import generate_text_embedding
from app.services.guard import evaluate_pairing_guard, GuardDecision


def get_all_image_embeddings() -> List[Dict[str, Any]]:
    """Retrieve all image caption embeddings along with their metadata."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            i.id,
            i.filename,
            i.filepath,
            m.subject,
            m.category,
            m.attributes,
            m.caption,
            m.confidence,
            e.vector
        FROM images i
        JOIN image_metadata m ON i.id = m.image_id
        JOIN embeddings e ON i.id = e.entity_id
        WHERE e.entity_type = 'IMAGE_CAPTION'
    """)
    rows = cursor.fetchall()
    conn.close()

    candidates = []
    for row in rows:
        candidates.append({
            "image_id": row["id"],
            "filename": row["filename"],
            "filepath": row["filepath"],
            "subject": row["subject"],
            "category": row["category"],
            "attributes": json.loads(row["attributes"]) if row["attributes"] else [],
            "caption": row["caption"],
            "confidence": float(row["confidence"]),
            "vector": np.array(json.loads(row["vector"])),
        })
    return candidates


def match_post_images(
    post_title: str,
    post_content: str,
    expected_subject: Optional[str] = None,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Given a post narrative, embed the text, rank all candidate images by cosine similarity,
    and evaluate every candidate through the Mismatch Guard.
    """
    full_text = f"{post_title}. {post_content}".strip()
    post_vector = np.array(generate_text_embedding(full_text))

    candidates = get_all_image_embeddings()
    scored_candidates = []

    for cand in candidates:
        img_vector = cand["vector"]
        # Embeddings are normalized, cosine similarity is the dot product
        similarity = float(np.dot(post_vector, img_vector))

        guard_decision: GuardDecision = evaluate_pairing_guard(
            post_subject=expected_subject or post_title,
            image_metadata=cand,
            similarity_score=similarity,
        )

        scored_candidates.append({
            "image_id": cand["image_id"],
            "filename": cand["filename"],
            "subject": cand["subject"],
            "category": cand["category"],
            "caption": cand["caption"],
            "confidence": cand["confidence"],
            "similarity_score": round(similarity, 4),
            "guard_status": guard_decision.status,
            "guard_reason": guard_decision.reason,
        })

    # Sort primarily by similarity score descending
    scored_candidates.sort(key=lambda x: x["similarity_score"], reverse=True)
    return scored_candidates[:top_k]


def force_pairing_evaluation(post_subject: str, post_content: str, image_filename: str) -> Dict[str, Any]:
    """Force an explicit image candidate for a post to test Mismatch Guard rejection."""
    candidates = get_all_image_embeddings()
    target_img = next((c for c in candidates if c["filename"] == image_filename), None)

    if not target_img:
        raise ValueError(f"Image '{image_filename}' not found in database.")

    full_text = f"{post_subject}. {post_content}".strip()
    post_vector = np.array(generate_text_embedding(full_text))
    similarity = float(np.dot(post_vector, target_img["vector"]))

    guard_decision = evaluate_pairing_guard(
        post_subject=post_subject,
        image_metadata=target_img,
        similarity_score=similarity,
    )

    return {
        "post": post_subject,
        "candidate": f"{target_img['filename']} ({target_img['subject']})",
        "similarity_score": round(similarity, 4),
        "result": guard_decision.status,
        "reason": guard_decision.reason,
    }
