from fastapi import FastAPI, HTTPException

from app.agent.graph import graph
from app.schemas.design_intent import DesignResponse, UserRequest

app = FastAPI(title="KrafNova AI service")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/design", response_model=DesignResponse)
def design(req: UserRequest):
    result = graph.invoke({"message": req.message})
    if not result.get("intent"):
        raise HTTPException(status_code=422, detail=result.get("error", "could not parse intent"))
    return DesignResponse(
        intent=result["intent"],
        candidates=result["candidates"],
        explanation=result["explanation"],
    )
