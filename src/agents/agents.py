"""Agent registry and management."""
from typing import Dict, Set, Any

from langgraph.graph.state import CompiledStateGraph

from .bg_task_agent.bg_task_agent import bg_task_agent
from .chatbot import chatbot
from .orchestrator import create_orchestrator
from .research_assistant import research_assistant
from .registry import AgentRegistry
from .types import AgentLike
from schema import AgentInfo

DEFAULT_AGENT = "research-assistant"

class GraphAgent:
    """Wrapper class that implements AgentLike protocol for compiled graphs."""
    def __init__(self, description: str, graph: CompiledStateGraph, capabilities: Set[str]):
        self._description = description
        self._graph = graph
        self._capabilities = capabilities

    @property
    def description(self) -> str:
        return self._description

    @property
    def capabilities(self) -> Set[str]:
        return self._capabilities

    @property
    def graph(self) -> CompiledStateGraph:
        return self._graph

    async def ainvoke(self, state: dict[str, Any]) -> dict[str, Any]:
        return await self._graph.ainvoke(state)

# Initialize registry
_registry = AgentRegistry()

# Register base agents
_base_agents = {
    "chatbot": GraphAgent(
        description="A simple chatbot.",
        graph=chatbot,
        capabilities={"chat", "conversation"}
    ),
    "research-assistant": GraphAgent(
        description="A research assistant with web search and calculator.",
        graph=research_assistant,
        capabilities={"web_search", "calculation", "research"}
    ),
    "bg-task-agent": GraphAgent(
        description="A background task agent.",
        graph=bg_task_agent,
        capabilities={"background_tasks", "async_execution"}
    ),
}

# Register base agents first
for agent_id, agent in _base_agents.items():
    _registry.register_base_agent(agent_id, agent)

# Create orchestrator with registry
orchestrator_graph = create_orchestrator(_registry)

# Register orchestrator separately
_registry.register_orchestrator(
    GraphAgent(
        description="An orchestrator agent that routes tasks between specialized agents.",
        graph=orchestrator_graph,
        capabilities={"routing", "task_delegation"}
    )
)


def get_agent(agent_id: str) -> CompiledStateGraph:
    """Get a compiled agent graph by ID."""
    return _registry.get_runnable(agent_id)


def get_all_agent_info() -> list[AgentInfo]:
    """Get information about all available agents."""
    return [
        AgentInfo(key=agent_id, description=metadata.description)
        for agent_id, metadata in _registry.metadata.items()
    ]
