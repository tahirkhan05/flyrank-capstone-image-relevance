"""
ContextLens - Evaluation Benchmark Runner
Measures Top-1 Matching Precision across the labeled evaluation dataset.
Fulfills Probe 5 and Section 6 Quality Requirements.
"""

import json
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.matching_service import match_post_images

EVAL_FILE = Path(__file__).parent.parent / "data" / "eval_set.json"


def evaluate_dataset() -> dict:
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        posts = json.load(f)

    print("\n" + "=" * 80)
    print("ContextLens — Evaluation Benchmark: Top-1 Precision on Labeled Set")
    print("=" * 80)

    correct_count = 0
    total_count = len(posts)
    results = []

    print(f"{'ID':<3} | {'Post Title':<38} | {'Target':<12} | {'Top-1 Result':<18} | {'Status':<10} | {'Decision'}")
    print("-" * 105)

    for p in posts:
        post_id = p["id"]
        title = p["title"]
        content = p["content"]
        target = p["target_class"]
        expected_subject = p["expected_subject"]

        # Run matcher for top-1 suggestion
        suggestions = match_post_images(title, content, expected_subject=expected_subject, top_k=1)
        top = suggestions[0] if suggestions else None

        if target == "NONE":
            # For out-of-domain posts, correct means the Mismatch Guard refused the pairing
            is_correct = top is None or top["guard_status"] in ("NO_CONFIDENT_MATCH", "REJECTED")
            decision_str = "SAFE_ABSTAIN" if is_correct else "FALSE_ACCEPT"
        else:
            # For in-domain posts, correct means top match is ACCEPTED and belongs to the correct class
            is_correct = (
                top is not None
                and top["guard_status"] == "ACCEPTED"
                and target in top["filename"]
            )
            decision_str = "CORRECT_MATCH" if is_correct else "MISMATCH_OR_REJECT"

        if is_correct:
            correct_count += 1

        top_file = top["filename"] if top else "None"
        guard_st = top["guard_status"] if top else "None"
        print(f"{post_id:<3} | {title[:36]:<38} | {target:<12} | {top_file:<18} | {guard_st:<10} | {decision_str}")

        results.append({
            "post_id": post_id,
            "title": title,
            "target": target,
            "top_suggestion": top_file,
            "guard_status": guard_st,
            "is_correct": is_correct,
        })

    top1_precision = round((correct_count / total_count) * 100, 2)
    print("-" * 105)
    print(f"\nEvaluation Benchmark Completed:")
    print(f"Total Evaluated Posts: {total_count}")
    print(f"Correct Top-1 Decisions: {correct_count}")
    print(f"Top-1 Precision: {top1_precision}%\n")
    print("=" * 80)

    return {
        "total_posts": total_count,
        "correct_decisions": correct_count,
        "top1_precision_pct": top1_precision,
        "detailed_results": results,
    }


if __name__ == "__main__":
    evaluate_dataset()

