"""Pure graph construction for orchestrator."""
from pathlib import Path
from typing import Dict
from pydantic import BaseModel, Field

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from ..types import AgentLike, OrchestratorState
from .router import route_node, should_continue


class OrchestratorConfig(BaseModel):
    """Configuration for orchestrator graph."""
    checkpoint_dir: Path | None = None
    max_steps: int = 10
    model_name: str = "gpt-4"
    capability_matching: bool = Field(
        default=True,
        description="Whether to enable capability-based agent routing"
    )

    class Config:
        validate_assignment = True


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
    
    # Add the routing node
    graph.add_node("router", route_node)
    
    # Add nodes for each base agent
    for agent_id, agent in base_agents.items():
        graph.add_node(agent_id, agent.graph)
    
    # Set router as entry point
    graph.set_entry_point("router")
    
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
        if not state.next_agent:
            raise ValueError("No next agent specified")
        if state.next_agent not in base_agents:
            raise ValueError(f"Invalid agent: {state.next_agent}")
        return state.next_agent
    
    # Add the agent_executor node first
    graph.add_node("agent_executor", agent_executor)
    
    # Then add its conditional edges
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