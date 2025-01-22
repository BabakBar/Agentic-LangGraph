"""Orchestrator agent that routes tasks between specialized agents."""
from pathlib import Path

from .graph import build_orchestrator, OrchestratorConfig
from .state import OrchestratorState
from ..registry import AgentRegistry


def create_orchestrator(registry: AgentRegistry, checkpoint_dir: Path | None = None):
    """Create an orchestrator instance with the given registry.
    
    Args:
        registry: Registry of available agents
        checkpoint_dir: Optional directory for checkpointing
    
    Returns:
        Compiled orchestrator graph
    """
    config = OrchestratorConfig(checkpoint_dir=checkpoint_dir)
    base_agents = registry.get_base_agents()
    return build_orchestrator(config, base_agents)


__all__ = [
    "create_orchestrator",
    "OrchestratorState",
    "OrchestratorConfig",
]
