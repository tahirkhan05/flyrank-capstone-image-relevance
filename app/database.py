import json
import sqlite3
from pathlib import Path
from typing import Any, Optional
from app.config import DATABASE_PATH


def get_db_connection() -> sqlite3.Connection:
    """Get a thread-safe connection to the SQLite database with row factory and WAL mode."""
    conn = sqlite3.connect(DATABASE_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db():
    """Initialize SQLite database with required tables, constraints, and indexes."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Images table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT UNIQUE NOT NULL,
        filepath TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'PENDING',  -- PENDING, PROCESSED, FLAGGED_LOW_CONFIDENCE, FAILED
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_status ON images(status);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_filename ON images(filename);")

    # 2. Image Metadata table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS image_metadata (
        image_id INTEGER PRIMARY KEY,
        subject TEXT NOT NULL,
        category TEXT NOT NULL,
        attributes TEXT NOT NULL,  -- JSON string of list[str]
        caption TEXT NOT NULL,
        confidence REAL NOT NULL,
        raw_response TEXT NOT NULL,
        FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE
    );
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_metadata_subject ON image_metadata(subject);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_metadata_category ON image_metadata(category);")

    # 3. Embeddings table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        entity_type TEXT NOT NULL,  -- IMAGE_CAPTION, POST_CONTENT
        entity_id INTEGER NOT NULL,
        vector TEXT NOT NULL,       -- JSON string of float array
        model_name TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_entity ON embeddings(entity_type, entity_id);")

    # 4. Posts table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS posts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        content TEXT NOT NULL,
        expected_subject TEXT,
        expected_category TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # 5. Suggestions table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS suggestions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        post_id INTEGER NOT NULL,
        image_id INTEGER NOT NULL,
        similarity_score REAL NOT NULL,
        guard_status TEXT NOT NULL,  -- ACCEPTED, REJECTED, NO_CONFIDENT_MATCH
        guard_reason TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (post_id) REFERENCES posts(id) ON DELETE CASCADE,
        FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE
    );
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_suggestions_post ON suggestions(post_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_suggestions_image ON suggestions(image_id);")

    # 6. Reviews table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS reviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        suggestion_id INTEGER UNIQUE NOT NULL,
        reviewer_action TEXT NOT NULL,  -- APPROVED, REJECTED
        notes TEXT,
        reviewed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (suggestion_id) REFERENCES suggestions(id) ON DELETE CASCADE
    );
    """)

    # 7. AI Cost Logs table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ai_cost_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        operation TEXT NOT NULL,  -- VISION_CLASSIFICATION, TEXT_EMBEDDING
        model TEXT NOT NULL,
        tokens_in INTEGER NOT NULL,
        tokens_out INTEGER NOT NULL,
        estimated_cost_usd REAL NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_cost_operation ON ai_cost_logs(operation);")

    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    print("ContextLens database initialized successfully with WAL mode!")
