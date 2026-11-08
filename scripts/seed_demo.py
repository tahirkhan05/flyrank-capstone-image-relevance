"""
ContextLens - Demo Seeding Script
Seeds sample editorial posts, generates image suggestions via the matching engine and guard,
and records an approved review to populate the human-in-the-loop review workflow.
"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import get_db_connection, init_db
from app.services.matching_service import match_post_images


def seed_demo_data():
    init_db()

    print("Seeding sample editorial posts...")
    sample_posts = [
        (
            "The Behaviour of Red Foxes",
            "Red foxes are solitary omnivores that hunt rodents and display remarkable adaptability across varied woodlands.",
            "red fox",
            "animal",
        ),
        (
            "Apex Predators: Gray Wolf Pack Dynamics",
            "Gray wolves coordinate in hierarchical packs to track and bring down large prey across northern taiga habitats.",
            "gray wolf",
            "animal",
        ),
        (
            "Quantum Superconducting Circuits and Qubit Coherence",
            "Engineering Josephson junctions and cryogenic dilution systems to maintain quantum coherence for digital computation.",
            "quantum computing",
            "technology",
        ),
    ]

    post_records = []
    conn = get_db_connection()
    cursor = conn.cursor()

    for title, content, subj, cat in sample_posts:
        cursor.execute("SELECT id FROM posts WHERE title = ?", (title,))
        row = cursor.fetchone()
        if not row:
            cursor.execute(
                """
                INSERT INTO posts (title, content, expected_subject, expected_category)
                VALUES (?, ?, ?, ?)
                """,
                (title, content, subj, cat),
            )
            post_id = cursor.lastrowid
            print(f"  Created post {post_id}: '{title}'")
        else:
            post_id = row["id"]
        post_records.append((post_id, title, content, subj))

    conn.commit()
    conn.close()

    # Generate suggestions without holding open an active DB transaction
    for post_id, title, content, subj in post_records:
        suggestions = match_post_images(title, content, expected_subject=subj, top_k=3)

        conn = get_db_connection()
        cursor = conn.cursor()
        for s in suggestions:
            cursor.execute(
                """
                INSERT OR REPLACE INTO suggestions (post_id, image_id, similarity_score, guard_status, guard_reason)
                VALUES (?, ?, ?, ?, ?)
                """,
                (post_id, s["image_id"], s["similarity_score"], s["guard_status"], s["guard_reason"]),
            )
            sugg_id = cursor.lastrowid

            # If it's an accepted match for the first post, simulate human reviewer approval
            if post_id == 1 and s["guard_status"] == "ACCEPTED":
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO reviews (suggestion_id, reviewer_action, notes)
                    VALUES (?, 'APPROVED', 'Editorial team verified: excellent pairing of red fox subject with article.')
                    """,
                    (sugg_id,),
                )
        conn.commit()
        conn.close()

    print("Demo seeding complete! Ready for live inspection and review API.")


if __name__ == "__main__":
    seed_demo_data()
