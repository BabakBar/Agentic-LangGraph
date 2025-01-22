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
    """Route to next agent with validation."""
    model = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    registry: AgentRegistry = config.get("registry")
    if not registry:
        raise ValueError("Registry required in config")

    # Get routing decision
    decision = await model.ainvoke(state.model_dump())
    
    # Validate with registry context
    validated = ValidatedRouterOutput(
        **decision,
        context={"registry": registry}
    )
    
    # Update state
    return OrchestratorState(
        messages=state.messages,
        agent_ids=state.agent_ids,
        next_agent=validated.next if validated.next != "FINISH" else None
    )


def should_continue(state: OrchestratorState) -> str:
    """Determine if orchestration should continue."""
    return "continue" if state.next_agent else "end"