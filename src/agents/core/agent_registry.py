"""Agent registry with runtime and serializable components."""
from typing import Any, Dict

from .registry_base import RegistryBase, RegistryError
from ..common.types import AgentLike, AgentMetadata, RegistrationError

class AgentRegistry(RegistryBase[AgentLike, AgentMetadata]):
    """Registry managing agent metadata and runtime instances with categories."""

    def _validate_item(self, agent: AgentLike) -> None:
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
        self._validate_item(agent)
        self._categories.setdefault("base", set()).add(agent_id)
        self._register(agent_id, agent)

    def register_orchestrator(self, agent: AgentLike) -> None:
        """Register orchestrator separately."""
        self._validate_item(agent)
        self._categories["orchestrator"] = {"orchestrator"}
        self._register("orchestrator", agent)

    def get_agent(self, agent_id: str) -> AgentLike:
        """Get runtime agent instance by ID."""
        return self.get_item(agent_id)

    def get_runnable(self, agent_id: str) -> Any:
        """Get runnable instance (either StateGraph or CompiledStateGraph) by ID."""
        agent = self.get_agent(agent_id)
        if hasattr(agent, 'graph'):
            return agent.graph
        return agent

    def get_base_agents(self) -> Dict[str, Any]:
        """Get all base agents as runnable graphs."""
        base_agents = self.get_by_category("base")
        return {
            agent_id: self.get_runnable(agent_id)
            for agent_id in base_agents
        }
