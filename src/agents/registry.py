"""Agent registry with runtime and serializable components."""
from typing import Any, Dict, Set
from pydantic import BaseModel, PrivateAttr

from .types import AgentLike, AgentMetadata, RegistrationError


class AgentRegistry(BaseModel):
    """Registry managing agent metadata and runtime instances with categories."""
    metadata: Dict[str, AgentMetadata] = {}
    _instances: Dict[str, AgentLike] = PrivateAttr(default_factory=dict)
    _categories: Dict[str, Set[str]] = PrivateAttr(default_factory=dict)

    def _validate_agent(self, agent: AgentLike) -> None:
        """Validate agent implements required interface."""
        if not isinstance(agent, AgentLike):
            raise RegistrationError(f"Agent must implement AgentLike protocol: {agent}")

    def _register(self, agent_id: str, agent: AgentLike) -> None:
        """Internal method to register an agent."""
        self.metadata[agent_id] = AgentMetadata(
            id=agent_id,
            description=agent.description,
            capabilities=list(agent.capabilities)
        )
        self._instances[agent_id] = agent

    def register_base_agent(self, agent_id: str, agent: AgentLike) -> None:
        """Register a base-level agent."""
        if agent_id == "orchestrator":
            raise RegistrationError("'orchestrator' is a reserved agent ID")
        self._validate_agent(agent)
        self._categories.setdefault("base", set()).add(agent_id)
        self._register(agent_id, agent)

    def register_orchestrator(self, agent: AgentLike) -> None:
        """Register orchestrator separately."""
        self._validate_agent(agent)
        self._categories["orchestrator"] = {"orchestrator"}
        self._register("orchestrator", agent)

    def get_agent(self, agent_id: str) -> AgentLike:
        """Get runtime agent instance by ID."""
        if agent_id not in self._instances:
            raise KeyError(f"Agent not found: {agent_id}")
        return self._instances[agent_id]

    def get_runnable(self, agent_id: str) -> Any:
        """Get runnable instance (either StateGraph or CompiledStateGraph) by ID."""
        agent = self.get_agent(agent_id)
        if hasattr(agent, 'graph'):
            return agent.graph
        return agent

    def get_base_agents(self) -> Dict[str, AgentLike]:
        """Get only base-level agents."""
        base_ids = self._categories.get("base", set())
        return {id: self._instances[id] for id in base_ids}

    def list_agents(self) -> list[str]:
        """Get list of registered agent IDs."""
        return list(self.metadata.keys())

    def has_agent(self, agent_id: str) -> bool:
        """Check if agent exists."""
        return agent_id in self.metadata

    def get_agent_capabilities(self, agent_id: str) -> Set[str]:
        """Get agent capabilities."""
        if not self.has_agent(agent_id):
            raise KeyError(f"Agent not found: {agent_id}")
        return self._instances[agent_id].capabilities