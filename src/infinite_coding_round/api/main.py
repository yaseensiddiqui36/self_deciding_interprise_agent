"""FastAPI application exposing the self-correcting enterprise data agent."""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException

from infinite_coding_round.agent.graph import run_agent
from infinite_coding_round.api.schemas import AskRequest, AskResponse, RetrievedSource

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Self-Correcting Enterprise Data Agent",
    description="Answers business questions via SQL, document retrieval, or both, "
    "with grounding validation and a single automatic retry on failure.",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    try:
        state = run_agent(request.question)
    except Exception as exc:  # noqa: BLE001 - surface as a 500 with context
        logger.exception("Agent execution failed")
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {exc}") from exc

    return AskResponse(
        answer=state.get("answer", ""),
        tools_used=state.get("tools_used", []),
        generated_sql=state.get("sql_query"),
        retrieved_sources=[RetrievedSource(**s) for s in state.get("retrieved_sources", [])],
        confidence=state.get("confidence", 0.0),
        validation_result=state.get("validation_reason", ""),
        validation_passed=state.get("validation_passed", False),
        retry_count=state.get("retry_count", 0),
    )
