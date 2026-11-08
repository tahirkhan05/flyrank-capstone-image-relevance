"""
ContextLens - Phase 3 Verification Suite
Executes and validates Probes 2, 3, and 4 against the live matching engine and Mismatch Guard.
"""

import sys
import json
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.matching_service import match_post_images, force_pairing_evaluation


def test_probe_2():
    print("=" * 70)
    print("PROBE 2: Query images for 'The behavior of red foxes' article")
    print("=" * 70)
    title = "The behavior of red foxes"
    content = "Red foxes are solitary hunters that feed on rodents and small animals in woodland and field habitats."
    results = match_post_images(title, content, expected_subject="red fox", top_k=6)

    print(f"\nTop Match:")
    top = results[0]
    print(f"  Filename: {top['filename']}")
    print(f"  Subject:  {top['subject']}")
    print(f"  Similarity: {top['similarity_score']}")
    print(f"  Guard Status: {top['guard_status']}")
    print(f"  Guard Reason: {top['guard_reason']}")

    # Assertions
    assert "fox" in top["subject"].lower(), "Probe 2 Failed: Top candidate must be a fox!"
    assert top["guard_status"] == "ACCEPTED", "Probe 2 Failed: Top fox candidate must be ACCEPTED!"
    print("\nPROBE 2 PASSED: Fox image ranks first and is ACCEPTED!\n")


def test_probe_3():
    print("=" * 70)
    print("PROBE 3: Force the wolf as a candidate for the fox post")
    print("=" * 70)
    post_subject = "The behavior of red foxes"
    post_content = "Study on the territory and social behaviour of red foxes."
    forced_wolf = "gray_wolf_01.jpg"

    evaluation = force_pairing_evaluation(post_subject, post_content, forced_wolf)
    print(json.dumps(evaluation, indent=2))

    assert evaluation["result"] == "REJECTED", "Probe 3 Failed: Wolf must be REJECTED on fox post!"
    assert "Animal category mismatch: expected fox, detected wolf" in evaluation["reason"], (
        f"Probe 3 Failed: Unexpected reason: {evaluation['reason']}"
    )
    print("\nPROBE 3 PASSED: Wolf was successfully REJECTED with category-mismatch explanation!\n")


def test_probe_4():
    print("=" * 70)
    print("PROBE 4: Query a post with no suitable image (Unrelated Domain)")
    print("=" * 70)
    title = "Advancements in Quantum Superconducting Qubits"
    content = "Exploring Josephson junctions, quantum decoherence times, and cryogenic dilution refrigerators for next-generation computing."
    results = match_post_images(title, content, expected_subject="quantum computing", top_k=3)

    print("Candidates returned:")
    for r in results:
        print(f"  Filename: {r['filename']} | Sim: {r['similarity_score']} | Status: {r['guard_status']}")
        print(f"  Reason:   {r['guard_reason']}")

    # Assertions: none should be accepted
    assert all(r["guard_status"] in ("NO_CONFIDENT_MATCH", "REJECTED") for r in results), (
        "Probe 4 Failed: No candidate should be accepted for unrelated post!"
    )
    print("\nPROBE 4 PASSED: System answered with 'NO_CONFIDENT_MATCH' below threshold!\n")


def main():
    test_probe_2()
    test_probe_3()
    test_probe_4()
    print("======================================================================")
    print("ALL PHASE 3 PROBES (PROBE 2, PROBE 3, PROBE 4) VERIFIED SUCCESSFULLY!")
    print("======================================================================")


if __name__ == "__main__":
    main()
