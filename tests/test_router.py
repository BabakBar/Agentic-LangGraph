"""Tests for router implementation."""
import pytest
from datetime import datetime
from typing import Dict, Set, List
from unittest.mock import AsyncMock, Mock, patch

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from src.agents.common.types import (
    AgentLike,
    OrchestratorState,
    RouterDecision,
    ValidatedRouterOutput,
    RoutingError,
    RoutingMetadata
)
from src.agents.registry import AgentRegistry
from src.agents.core.orchestrator.router import (
    RoutingManager,
    ROUTING_PROMPT,
    route_node,
    should_continue
)


class MockAgent:
    """Mock agent for testing."""
    def __init__(self, agent_id: str, description: str, capabilities: Set[str] = None):
        self.agent_id = agent_id
        self._description = description
        self._capabilities = capabilities or set()
        
    @property
    def description(self) -> str:
        return self._description
        
    @property
    def capabilities(self) -> Set[str]:
        return self._capabilities
        
    async def ainvoke(self, state: Dict[str, str]) -> Dict[str, str]:
        return {"response": f"Mock response from {self.agent_id}"}


@pytest.fixture
def mock_registry():
    """Create mock registry with test agents."""
    registry = AgentRegistry()
    
    # Register test agents
    registry.register_base_agent(
        "bg-task-agent",
        MockAgent(
            "bg-task-agent",
            "Handles background tasks and code processing",
            {"code", "task"}
        )
    )
    registry.register_base_agent(
        "research-assistant",
        MockAgent(
            "research-assistant",
            "Performs research and answers questions",
            {"search", "research"}
        )
    )
    
    return registry


@pytest.fixture
def test_config(mock_registry):
    """Create test config with registry."""
    return {
        "registry": mock_registry,
        "model_name": "gpt-3.5-turbo",
        "min_confidence": 0.5,
        "enable_metrics": True
    }


@pytest.fixture
def test_state():
    """Create test orchestrator state."""
    return OrchestratorState(
        messages=[HumanMessage(content="Test message")]
    )


@pytest.fixture
def mock_llm():
    """Create mock LLM that returns valid routing decisions."""
    async def mock_ainvoke(messages: List[SystemMessage]):
        # Return JSON string that matches RouterDecision schema
        return Mock(
            content="""
            {
                "next": "research-assistant",
                "reasoning": "Test reasoning",
                "confidence": 0.8,
                "alternatives": ["bg-task-agent"]
            }
            """
        )
    
    return Mock(ainvoke=AsyncMock(side_effect=mock_ainvoke))


def test_routing_manager_init(mock_registry, test_config):
    """Test RoutingManager initialization."""
    manager = RoutingManager(mock_registry, test_config)
    assert manager.registry == mock_registry
    assert manager.config == test_config


def test_routing_manager_keyword_decision():
    """Test keyword-based routing decisions."""
    manager = RoutingManager(Mock(), {})
    
    # Test task agent routing
    decision = manager.get_keyword_decision("process this data")
    assert decision.next == "bg-task-agent"
    assert decision.confidence == 0.8
    assert decision.reasoning
    assert "research-assistant" in decision.alternatives
    
    # Test research assistant fallback
    decision = manager.get_keyword_decision("what is the weather")
    assert decision.next == "research-assistant"
    assert decision.confidence == 0.6
    assert decision.reasoning
    assert "bg-task-agent" in decision.alternatives


def test_routing_manager_validation(mock_registry):
    """Test routing decision validation."""
    manager = RoutingManager(mock_registry, {"min_confidence": 0.5})
    
    # Test valid decision
    decision = RouterDecision(
        next="bg-task-agent",
        confidence=0.8,
        reasoning="Test",
        alternatives=[]
    )
    validated, error = manager.validate_decision(decision)
    assert validated is not None
    assert error is None
    assert isinstance(validated, ValidatedRouterOutput)
    assert validated.next == "bg-task-agent"
    
    # Test low confidence
    decision.confidence = 0.3
    validated, error = manager.validate_decision(decision)
    assert validated is None
    assert "Low confidence" in error
    
    # Test invalid agent
    decision.next = "invalid-agent"
    decision.confidence = 0.8
    validated, error = manager.validate_decision(decision)
    assert validated is None
    assert "Invalid agent" in error


@pytest.mark.asyncio
async def test_routing_manager_llm_decision(mock_registry, mock_llm, test_state):
    """Test LLM-based routing decisions."""
    manager = RoutingManager(mock_registry, {"model_name": "test"})
    manager.llm = mock_llm
    
    decision = await manager.get_llm_decision(test_state)
    assert isinstance(decision, RouterDecision)
    assert decision.next == "research-assistant"
    assert decision.confidence == 0.8
    assert decision.reasoning == "Test reasoning"
    assert decision.alternatives == ["bg-task-agent"]
    
    # Verify prompt formatting
    mock_llm.ainvoke.assert_called_once()
    messages = mock_llm.ainvoke.call_args[0][0]
    assert any("Available Agents:" in msg.content for msg in messages)
    assert any("Test message" in msg.content for msg in messages)


@pytest.mark.asyncio
async def test_routing_manager_fallback_chain(mock_registry, test_state):
    """Test fallback chain execution."""
    manager = RoutingManager(mock_registry, {"min_confidence": 0.5})
    
    # Test fallback with error
    state = await manager.execute_fallback_chain(test_state, "Test error")
    assert state.routing.fallback_used
    assert state.routing.current_agent in ["research-assistant", "bg-task-agent"]
    assert len(state.routing.errors) == 1
    assert state.routing.errors[0].error == "Test error"
    
    # Test fallback without error
    state = await manager.execute_fallback_chain(test_state)
    assert state.routing.fallback_used
    assert state.routing.current_agent in ["research-assistant", "bg-task-agent"]


@pytest.mark.asyncio
async def test_route_node_integration(test_state, test_config, mock_llm):
    """Test full routing node functionality."""
    with patch("src.agents.core.orchestrator.router.get_model", return_value=mock_llm):
        # Test successful LLM routing
        result = await route_node(test_state, test_config)
        assert result.routing.current_agent == "research-assistant"
        assert not result.routing.fallback_used
        assert len(result.routing.decision_history) == 1
        
        # Test fallback on LLM error
        mock_llm.ainvoke.side_effect = Exception("LLM error")
        result = await route_node(test_state, test_config)
        assert result.routing.current_agent in ["research-assistant", "bg-task-agent"]
        assert result.routing.fallback_used
        assert len(result.routing.errors) == 1
        
        # Test error on missing registry
        with pytest.raises(RoutingError):
            await route_node(test_state, {})


def test_should_continue():
    """Test continuation logic."""
    # Test continue with agent
    state = OrchestratorState(
        messages=[HumanMessage(content="test")],
        routing=RoutingMetadata(current_agent="test-agent")
    )
    assert should_continue(state) == "continue"
    
    # Test end without agent
    state = OrchestratorState(
        messages=[HumanMessage(content="test")],
        routing=RoutingMetadata(current_agent=None)
    )
    assert should_continue(state) == "end"
