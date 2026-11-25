"""Node 4: explain what was made and why, in plain language.

TODO: call the LLM with the intent and candidates, so the user understands
the choices (and so the thesis can evaluate explanation quality).
"""


def explain(state: dict) -> dict:
    intent = state["intent"]
    n = len(state["candidates"])
    return {
        "explanation": (
            f"Made {n} {intent.style} options for '{intent.subject}' at "
            f"{intent.dimensions.width_mm}x{intent.dimensions.height_mm} mm."
        )
    }
