"""Node 2: generate candidate images from the validated intent.

TODO: call the image model. Return a few candidates so the user can pick.
"""
from app.schemas.design_intent import Candidate


def generate_images(state: dict) -> dict:
    intent = state["intent"]
    candidates = [
        Candidate(id=f"cand-{i}", image_url=f"placeholder://{intent.subject}/{i}")
        for i in range(1, 4)
    ]
    return {"candidates": candidates}
