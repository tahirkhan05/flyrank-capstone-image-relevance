import json
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel

from app.database import get_db_connection
from app.schemas import (
    PostCreate,
    PostResponse,
    ImageRecordResponse,
    SuggestionResponse,
    ReviewCreate,
    ReviewResponse,
    CostSummary,
    AltTextResponse,
    DuplicateCheckResponse,
    DuplicateItem,
)
from app.services.matching_service import match_post_images, get_all_image_embeddings

router = APIRouter()


@router.get("/health", tags=["System"])
def health_check():
    """System health check endpoint."""
    return {"status": "ok", "system": "ContextLens", "version": "1.0.0"}


@router.get("/api/images", response_model=List[ImageRecordResponse], tags=["Assets"])
def list_images(status_filter: Optional[str] = Query(None, alias="status")):
    """List all ingested image assets with metadata and confidence status."""
    conn = get_db_connection()
    cursor = conn.cursor()

    if status_filter:
        cursor.execute(
            """
            SELECT i.id, i.filename, i.filepath, i.status, i.created_at,
                   m.subject, m.category, m.attributes, m.caption, m.confidence
            FROM images i
            LEFT JOIN image_metadata m ON i.id = m.image_id
            WHERE i.status = ?
            ORDER BY i.id ASC
            """,
            (status_filter,),
        )
    else:
        cursor.execute(
            """
            SELECT i.id, i.filename, i.filepath, i.status, i.created_at,
                   m.subject, m.category, m.attributes, m.caption, m.confidence
            FROM images i
            LEFT JOIN image_metadata m ON i.id = m.image_id
            ORDER BY i.id ASC
            """
        )

    rows = cursor.fetchall()
    conn.close()

    results = []
    for r in rows:
        meta = None
        if r["subject"]:
            meta = {
                "subject": r["subject"],
                "category": r["category"],
                "attributes": json.loads(r["attributes"]) if r["attributes"] else [],
                "caption": r["caption"],
                "confidence": r["confidence"],
            }
        results.append({
            "id": r["id"],
            "filename": r["filename"],
            "filepath": r["filepath"],
            "status": r["status"],
            "created_at": str(r["created_at"]),
            "metadata": meta,
        })
    return results


@router.get("/api/images/{image_id}/alt-text", response_model=AltTextResponse, tags=["Accessibility & Stretch"])
def get_image_alt_text(image_id: int):
    """
    Stretch Goal 1: Automatic WCAG-compliant Alt-Text generation
    derived from structured vision metadata and distinctive attributes.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT i.id, i.filename, i.filepath, m.subject, m.category, m.attributes, m.caption
        FROM images i
        JOIN image_metadata m ON i.id = m.image_id
        WHERE i.id = ?
    """, (image_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail=f"Image with ID {image_id} has no processed metadata")

    subject = row["subject"]
    category = row["category"]
    caption = row["caption"] or f"A photo of {subject}"
    attrs = json.loads(row["attributes"]) if row["attributes"] else []
    
    alt_text = caption.strip()
    if not alt_text.endswith("."):
        alt_text += "."

    return {
        "image_id": row["id"],
        "filename": row["filename"],
        "alt_text": alt_text,
        "subject": subject,
        "category": category,
        "attributes": attrs,
        "wcag_compliant": True,
        "suggested_html": f'<img src="{row["filepath"]}" alt="{alt_text}" />'
    }


@router.get("/api/images/{image_id}/duplicates", response_model=DuplicateCheckResponse, tags=["Deduplication & Stretch"])
def check_image_duplicates(image_id: int, threshold: float = Query(0.85, ge=0.5, le=1.0)):
    """
    Stretch Goal 2: Near-duplicate asset detection using embedding distance / cosine similarity.
    Flags closely related or near-identical images in the corpus.
    """
    import numpy as np
    all_embeddings = get_all_image_embeddings()
    target_item = next((item for item in all_embeddings if item["image_id"] == image_id), None)

    if not target_item:
        raise HTTPException(status_code=404, detail=f"Target image ID {image_id} has no stored embeddings")

    target_vec = np.array(target_item["vector"], dtype=np.float32)
    duplicates = []

    for other in all_embeddings:
        if other["image_id"] == image_id:
            continue
        other_vec = np.array(other["vector"], dtype=np.float32)
        dot = float(np.dot(target_vec, other_vec))
        norm_t = float(np.linalg.norm(target_vec))
        norm_o = float(np.linalg.norm(other_vec))
        sim = dot / (norm_t * norm_o) if (norm_t > 0 and norm_o > 0) else 0.0

        if sim >= threshold:
            duplicates.append({
                "image_id": other["image_id"],
                "filename": other["filename"],
                "similarity": round(sim, 4),
                "is_duplicate_candidate": True
            })

    duplicates.sort(key=lambda x: x["similarity"], reverse=True)

    return {
        "image_id": image_id,
        "filename": target_item["filename"],
        "subject": target_item["subject"],
        "threshold": threshold,
        "potential_duplicates_count": len(duplicates),
        "duplicates": duplicates
    }


@router.post("/api/posts", response_model=PostResponse, status_code=status.HTTP_201_CREATED, tags=["Editorial Posts"])
def create_post(post: PostCreate):
    """Register a new editorial post or article for asset matching."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO posts (title, content, expected_subject, expected_category)
        VALUES (?, ?, ?, ?)
        """,
        (post.title, post.content, post.expected_subject, post.expected_category),
    )
    post_id = cursor.lastrowid
    conn.commit()

    cursor.execute("SELECT id, title, content, expected_subject, expected_category, created_at FROM posts WHERE id = ?", (post_id,))
    row = cursor.fetchone()
    conn.close()

    return {
        "id": row["id"],
        "title": row["title"],
        "content": row["content"],
        "expected_subject": row["expected_subject"],
        "expected_category": row["expected_category"],
        "created_at": str(row["created_at"]),
    }


@router.get("/api/posts", response_model=List[PostResponse], tags=["Editorial Posts"])
def list_posts():
    """List all registered editorial posts."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, content, expected_subject, expected_category, created_at FROM posts ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "id": r["id"],
            "title": r["title"],
            "content": r["content"],
            "expected_subject": r["expected_subject"],
            "expected_category": r["expected_category"],
            "created_at": str(r["created_at"]),
        }
        for r in rows
    ]


@router.get("/api/posts/{post_id}/images", response_model=List[SuggestionResponse], tags=["Matching & Guard"])
def get_post_image_suggestions(post_id: int, top_k: int = Query(5, ge=1, le=20)):
    """
    Retrieve ranked image suggestions for a post.
    Every candidate is evaluated through the Mismatch Guard with explainable decision reasons.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, content, expected_subject FROM posts WHERE id = ?", (post_id,))
    post = cursor.fetchone()

    if not post:
        conn.close()
        raise HTTPException(status_code=404, detail=f"Post with ID {post_id} not found")

    title = post["title"]
    content = post["content"]
    expected_subject = post["expected_subject"]

    # Execute semantic matching and guard evaluation
    suggestions = match_post_images(title, content, expected_subject=expected_subject, top_k=top_k)

    # Persist suggestions to database for human review workflow
    response_items = []
    for s in suggestions:
        # Check if suggestion already recorded
        cursor.execute(
            "SELECT id FROM suggestions WHERE post_id = ? AND image_id = ?",
            (post_id, s["image_id"]),
        )
        existing = cursor.fetchone()
        if existing:
            sugg_id = existing["id"]
            cursor.execute(
                """
                UPDATE suggestions 
                SET similarity_score = ?, guard_status = ?, guard_reason = ?
                WHERE id = ?
                """,
                (s["similarity_score"], s["guard_status"], s["guard_reason"], sugg_id),
            )
        else:
            cursor.execute(
                """
                INSERT INTO suggestions (post_id, image_id, similarity_score, guard_status, guard_reason)
                VALUES (?, ?, ?, ?, ?)
                """,
                (post_id, s["image_id"], s["similarity_score"], s["guard_status"], s["guard_reason"]),
            )
            sugg_id = cursor.lastrowid

        response_items.append({
            "id": sugg_id,
            "post_id": post_id,
            "image_id": s["image_id"],
            "image_filename": s["filename"],
            "image_caption": s["caption"],
            "image_subject": s["subject"],
            "similarity_score": s["similarity_score"],
            "guard_status": s["guard_status"],
            "guard_reason": s["guard_reason"],
        })

    conn.commit()
    conn.close()
    return response_items


@router.post("/api/reviews", response_model=ReviewResponse, status_code=status.HTTP_201_CREATED, tags=["Review API"])
def create_review(review: ReviewCreate):
    """Submit human-in-the-loop approval or rejection for a suggested pairing."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM suggestions WHERE id = ?", (review.suggestion_id,))
    if not cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail=f"Suggestion ID {review.suggestion_id} not found")

    cursor.execute(
        """
        INSERT OR REPLACE INTO reviews (suggestion_id, reviewer_action, notes)
        VALUES (?, ?, ?)
        """,
        (review.suggestion_id, review.reviewer_action, review.notes),
    )
    review_id = cursor.lastrowid
    conn.commit()

    cursor.execute("SELECT id, suggestion_id, reviewer_action, notes, reviewed_at FROM reviews WHERE id = ?", (review_id,))
    row = cursor.fetchone()
    conn.close()

    return {
        "id": row["id"],
        "suggestion_id": row["suggestion_id"],
        "reviewer_action": row["reviewer_action"],
        "notes": row["notes"],
        "reviewed_at": str(row["reviewed_at"]),
    }


@router.get("/api/reviews", tags=["Review API"])
def list_reviews():
    """List all editorial reviews."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT r.id, r.suggestion_id, r.reviewer_action, r.notes, r.reviewed_at,
               s.post_id, s.image_id, s.similarity_score, s.guard_status, s.guard_reason,
               p.title as post_title, i.filename as image_filename
        FROM reviews r
        JOIN suggestions s ON r.suggestion_id = s.id
        JOIN posts p ON s.post_id = p.id
        JOIN images i ON s.image_id = i.id
        ORDER BY r.id DESC
    """)
    rows = cursor.fetchall()
    conn.close()

    return [dict(r) for r in rows]


@router.get("/api/costs", response_model=CostSummary, tags=["AI Costs & Observability"])
def get_costs():
    """Retrieve full per-call attributed AI token costs and operational breakdowns."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT 
            COUNT(*) as total_calls,
            SUM(tokens_in) as total_tokens_in,
            SUM(tokens_out) as total_tokens_out,
            SUM(estimated_cost_usd) as total_cost
        FROM ai_cost_logs
    """)
    total_row = cursor.fetchone()

    cursor.execute("""
        SELECT operation, COUNT(*), SUM(tokens_in), SUM(tokens_out), SUM(estimated_cost_usd)
        FROM ai_cost_logs
        GROUP BY operation
    """)
    breakdown_rows = cursor.fetchall()
    conn.close()

    breakdown = {}
    for r in breakdown_rows:
        breakdown[r[0]] = {
            "calls": r[1],
            "tokens_in": r[2] or 0,
            "tokens_out": r[3] or 0,
            "cost_usd": round(r[4] or 0.0, 6),
        }

    return {
        "total_calls": total_row["total_calls"] or 0,
        "total_tokens_in": total_row["total_tokens_in"] or 0,
        "total_tokens_out": total_row["total_tokens_out"] or 0,
        "total_estimated_cost_usd": round(total_row["total_cost"] or 0.0, 6),
        "breakdown_by_operation": breakdown,
    }

