"""Pure graph construction for orchestrator."""
from pathlib import Path
from typing import Dict, Optional
from pydantic import BaseModel, Field, model_validator

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from ..types import AgentLike, OrchestratorState
from .router import route_node, should_continue


class OrchestratorConfig(BaseModel):
    """Configuration for orchestrator graph."""
    # Core settings
    checkpoint_dir: Optional[Path] = None
    max_steps: int = Field(default=10, ge=1)
    
    # Routing settings
    model_name: str = "gpt-3.5-turbo"
    llm_routing: bool = Field(
        default=True,
        description="Enable LLM-based routing decisions"
    )
    llm_fallback_threshold: float = Field(
        default=0.7,
        description="Confidence threshold for falling back to keyword routing"
    )
    capability_matching: bool = Field(
        default=True,
        description="Enable capability-based agent routing"
    )
    
    # Monitoring settings
    enable_metrics: bool = Field(
        default=True,
        description="Enable routing metrics collection"
    )
    
    # State versioning
    min_state_version: str = "2.0"
    
    @model_validator(mode="after")
    def validate_config(self) -> "OrchestratorConfig":
        """Validate configuration settings."""
        if self.llm_fallback_threshold < 0.5:
            raise ValueError("LLM fallback threshold must be >= 0.5")
        return self
    
    class Config:
        validate_assignment = True


def migrate_state(state: OrchestratorState, min_version: str) -> OrchestratorState:
    """Migrate state to current version if needed."""
    if not hasattr(state, "schema_version") or state.schema_version < min_version:
        # Create new state with current version
        return OrchestratorState(
            messages=state.messages,
            agent_ids=state.agent_ids,
            next_agent=state.next_agent,
            schema_version=min_version
        )
    return state


def build_orchestrator(
    config: OrchestratorConfig,
    base_agents: Dict[str, AgentLike],
) -> StateGraph:
    """Create orchestrator graph with pure construction.
    
    Args:
        config: Configuration for the orchestrator
        base_agents: Dictionary of base agents to orchestrate
    
    Returns:
        Compiled state graph
    """
    # Initialize graph with our state type
    graph = StateGraph(OrchestratorState)
    
    # Add state migration node
    def migrate_node(state: OrchestratorState) -> OrchestratorState:
        return migrate_state(state, config.min_state_version)
    
    graph.add_node("migrate", migrate_node)
    
    # Add the routing node with config
    def configured_route_node(state: OrchestratorState):
        return route_node(state, {
            "registry": base_agents,
            "model_name": config.model_name,
            "min_confidence": config.llm_fallback_threshold,
            "enable_metrics": config.enable_metrics
        })
    
    graph.add_node("router", configured_route_node)
    
    # Add nodes for each base agent
    for agent_id, agent in base_agents.items():
        graph.add_node(agent_id, agent.graph)
    
    # Set migration as entry point
    graph.set_entry_point("migrate")
    
    # Add edge from migration to router
    graph.add_edge("migrate", "router")
    
    # Add conditional edges from router to agents
    agent_map = {name: name for name in base_agents.keys()}
    graph.add_conditional_edges(
        "router",
        should_continue,
        {
            "continue": "agent_executor",
            "end": END
        }
    )
    
    # Add agent execution node that routes to appropriate agent
    def agent_executor(state: OrchestratorState) -> str:
        """Route to next agent based on state."""
        current_agent = state.routing.current_agent
        if not current_agent:
            raise ValueError("No agent specified in routing state")
        if current_agent not in base_agents:
            raise ValueError(f"Invalid agent: {current_agent}")
        return current_agent
    
    # Add the agent_executor node
    graph.add_node("agent_executor", agent_executor)
    
    # Add conditional edges for agent execution
    graph.add_conditional_edges(
        "agent_executor",
        agent_executor,
        agent_map
    )
    
    # Add edges from agents back to router
    for agent_id in base_agents:
        graph.add_edge(agent_id, "router")
    
    # Configure checkpointing
    checkpointer = None
    if config.checkpoint_dir:
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpointer = MemorySaver()
    
    # Compile the graph
    return graph.compile(checkpointer=checkpointer)
