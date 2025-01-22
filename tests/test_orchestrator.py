import pytest
from langchain_core.messages import HumanMessage

from src.agents.agents import get_agent
from src.agents.orchestrator.state import OrchestratorState


@pytest.mark.asyncio
async def test_orchestrator_basic_routing():
    """Test that orchestrator can route basic requests to appropriate agents."""
    # Get the orchestrator
    orchestrator = get_agent("orchestrator")
    
    # Test cases with expected routing
    test_cases = [
        (
            "What's the weather in Paris?",
            "research-assistant",  # Should route to research assistant for weather queries
        ),
        (
            "Hello, how are you?",
            "chatbot",  # Should route to chatbot for basic conversation
        ),
        (
            "Can you run this task in the background?",
            "bg-task-agent",  # Should route to background task agent
        ),
    ]
    
    for message, expected_agent in test_cases:
        # Create initial state
        state = {
            "messages": [HumanMessage(content=message)],
            "remaining_steps": 10,
        }
        
        # Get first step of execution
        result = await orchestrator.ainvoke(state)
        
        # Check that router selected correct agent
        assert result["next"] == expected_agent, f"Expected {expected_agent} for message: {message}"


@pytest.mark.asyncio
async def test_orchestrator_completion():
    """Test that orchestrator properly handles task completion."""
    orchestrator = get_agent("orchestrator")
    
    # Create a state with a completed task response
    state = {
        "messages": [
            HumanMessage(content="Hello!"),
            HumanMessage(content="Task is complete.", name="chatbot"),
        ],
        "remaining_steps": 10,
    }
    
    # Execute orchestrator
    result = await orchestrator.ainvoke(state)
    
    # Should indicate completion by setting next to None
    assert result["next"] is None, "Orchestrator should finish when task is complete"


@pytest.mark.asyncio
async def test_orchestrator_state_management():
    """Test that orchestrator properly maintains state between steps."""
    orchestrator = get_agent("orchestrator")
    
    # Initial state
    state = {
        "messages": [HumanMessage(content="What's the weather in Paris?")],
        "remaining_steps": 10,
    }
    
    # First step - should route to research assistant
    result = await orchestrator.ainvoke(state)
    assert result["next"] == "research-assistant"
    
    # Update state with research assistant response
    result["messages"].append(
        HumanMessage(
            content="The weather in Paris is sunny.",
            name="research-assistant"
        )
    )
    
    # Next step - should complete since we have an answer
    final_result = await orchestrator.ainvoke(result)
    assert final_result["next"] is None