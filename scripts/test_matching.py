import sqlite3
import json
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).parent.parent / ".env")
from google import genai

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Query for Probe 2
query = "The behaviour of red foxes in the wild"
q_vec = np.array(client.models.embed_content(model="gemini-embedding-001", contents=query).embeddings[0].values)

conn = sqlite3.connect(Path(__file__).parent.parent / "contextlens.db")
c = conn.cursor()
rows = c.execute("""
    SELECT i.id, i.filename, m.subject, m.caption, e.vector 
    FROM images i 
    JOIN image_metadata m ON i.id = m.image_id 
    JOIN embeddings e ON i.id = e.entity_id 
    WHERE e.entity_type = 'IMAGE_CAPTION'
""").fetchall()

scores = []
for img_id, filename, subj, caption, vec_str in rows:
    v = np.array(json.loads(vec_str))
    sim = float(np.dot(q_vec, v))
    scores.append((sim, filename, subj, caption))

scores.sort(key=lambda x: x[0], reverse=True)

print("=" * 60)
print(f"PROBE 2: Query images for '{query}'")
print("=" * 60)
print("\nTop 5 Ranked Candidates:")
for rank, (sim, fn, subj, cap) in enumerate(scores[:5], 1):
    print(f"Rank {rank}: {sim:.4f} | {fn:<18} | {subj:<20}")

print("\nWolf & Dog Candidates (Must rank significantly lower):")
for rank, (sim, fn, subj, cap) in enumerate(scores, 1):
    if "wolf" in fn or "dog" in fn:
        print(f"Rank {rank:2d}: {sim:.4f} | {fn:<18} | {subj:<20}")
        if rank > 20:
            break
