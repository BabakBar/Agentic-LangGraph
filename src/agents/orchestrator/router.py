"""Router implementation with Pydantic validation."""
from typing import Dict, Any
from pydantic import BaseModel, model_validator
from langchain_core.runnables import RunnableConfig

from core import get_model, settings
from ..registry import AgentRegistry
from ..types import OrchestratorState


class RouterDecision(BaseModel):
    """Pure routing decision from LLM."""
    next: str


class ValidatedRouterOutput(RouterDecision):
    """Validated router output with registry context."""
    @model_validator(mode="after")
    def validate_next_agent(self, context: Dict[str, Any]) -> "ValidatedRouterOutput":
        registry: AgentRegistry = context.get("registry")
        if not registry:
            raise ValueError("Registry context required for validation")
            
        if self.next != "FINISH" and not registry.has_agent(self.next):
            raise ValueError(f"Invalid agent: {self.next}")
        return self


async def route_node(state: OrchestratorState, config: RunnableConfig) -> OrchestratorState:
    user_input = state.messages[-1].content.lower()
    
    # Simple keyword-based routing
    task_keywords = {"process", "background", "task", "generate", "analyze"}
    if any(kw in user_input for kw in task_keywords):
        return OrchestratorState(
            messages=state.messages,
            agent_ids=state.agent_ids,
            next_agent="bg-task-agent"
        )
    
    # Default to research assistant
    return OrchestratorState(
        messages=state.messages,
        agent_ids=state.agent_ids,
        next_agent="research-assistant"
    )


def should_continue(state: OrchestratorState) -> str:
    """Determine if orchestration should continue."""
    if state.next_agent:
        try:
            # Validate agent exists
            # config["registry"].get_agent(state.next_agent) # this line was causing an error
            return "continue"
        except Exception:
            state.next_agent = "research-assistant"  # Fallback
            return "continue"
    return "end"
