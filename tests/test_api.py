import pytest
from fastapi.testclient import TestClient
from main import app
from app.services.matching_service import force_pairing_evaluation

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["system"] == "ContextLens"


def test_list_images():
    response = client.get("/api/images")
    assert response.status_code == 200
    images = response.json()
    assert len(images) >= 50
    # Verify at least one image is flagged low confidence (Probe 1)
    flagged = [img for img in images if img["status"] == "FLAGGED_LOW_CONFIDENCE"]
    assert len(flagged) >= 1, "At least one image must be FLAGGED_LOW_CONFIDENCE"


def test_get_posts():
    response = client.get("/api/posts")
    assert response.status_code == 200
    posts = response.json()
    assert len(posts) >= 1


def test_post_image_suggestions_and_guard():
    # Probe 2: Query images for red fox article
    response = client.get("/api/posts/1/images?top_k=5")
    assert response.status_code == 200
    suggestions = response.json()
    assert len(suggestions) > 0
    top = suggestions[0]
    assert "fox" in top["image_subject"].lower()
    assert top["guard_status"] == "ACCEPTED"


def test_mismatch_guard_forced_rejection():
    # Probe 3: Force wolf on fox post -> must be REJECTED
    eval_result = force_pairing_evaluation(
        post_subject="The behavior of red foxes",
        post_content="Field observations on the ecology and behaviour of red foxes.",
        image_filename="gray_wolf_01.jpg",
    )
    assert eval_result["result"] == "REJECTED"
    assert "Animal category mismatch: expected fox, detected wolf" in eval_result["reason"]


def test_create_and_list_reviews():
    # Review workflow
    response = client.get("/api/posts/1/images")
    suggestions = response.json()
    first_sugg_id = suggestions[0]["id"]

    review_payload = {
        "suggestion_id": first_sugg_id,
        "reviewer_action": "APPROVED",
        "notes": "Verified by reviewer: accurate matching of red fox image to article.",
    }
    rev_resp = client.post("/api/reviews", json=review_payload)
    assert rev_resp.status_code == 201
    rev_data = rev_resp.json()
    assert rev_data["reviewer_action"] == "APPROVED"

    # List reviews
    list_resp = client.get("/api/reviews")
    assert list_resp.status_code == 200
    assert len(list_resp.json()) >= 1


def test_ai_cost_logs():
    # Probe 6: Check cost log
    response = client.get("/api/costs")
    assert response.status_code == 200
    cost_data = response.json()
    assert cost_data["total_calls"] > 0
    assert cost_data["total_tokens_in"] > 0
    assert cost_data["total_estimated_cost_usd"] > 0.0
    assert "VISION_CLASSIFICATION" in cost_data["breakdown_by_operation"]

