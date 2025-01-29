"""State management for the orchestrator agent."""
from typing import Dict, Any

from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage

from ..types import OrchestratorState, ToolState


class Router(BaseModel):
    """Structured output for the router to select next agent."""
    next: str
    context: Dict[str, Any] = {}

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "next": "research-assistant",
                    "context": {"reason": "Task requires web search"}
                },
                {
                    "next": "FINISH",
                    "context": {"reason": "Task complete"}
                }
            ]
        }
    }


def create_initial_state(messages: list[BaseMessage]) -> OrchestratorState:
    """Create initial state for orchestrator."""
    return OrchestratorState(
        messages=messages,
        agent_ids=[],
        next_agent=None,
        tool_state=ToolState()  # Initialize with empty tool state
    )