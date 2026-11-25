from typing import Optional, TypedDict

from app.schemas.design_intent import Candidate, DesignIntent


class AgentState(TypedDict, total=False):
    message: str
    intent: Optional[DesignIntent]
    candidates: list[Candidate]
    explanation: str
    error: Optional[str]
