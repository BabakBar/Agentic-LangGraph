"""Router implementation with LLM-based routing and validation."""
from typing import Dict, Any, Optional, Tuple, List, Literal
import re
from datetime import datetime
import time
import functools
import logging

from pydantic import BaseModel, Field, model_validator
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Command as BaseCommand
from langchain_core.language_models import BaseLLM

from core.llm import get_model
from core.settings import settings
from ..agent_registry import AgentRegistry
from ...common.types import (
    OrchestratorState,
    AgentError, RouterError, AgentNotFoundError,
    AgentExecutionError, MaxErrorsExceeded
)


logger = logging.getLogger(__name__)

# Define available agents as literals for type safety
AgentType = Literal[
    "research-assistant",
    "chatbot",
    "bg-task-agent",
    "FINISH"
]

class RouterDecision(BaseModel):
    """Structured output for routing decisions."""
    next: AgentType
    confidence: float
    reasoning: str
    capabilities_matched: List[str] = Field(default_factory=list)
    fallback_agents: List[AgentType] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_decision(self) -> "RouterDecision":
        """Validate decision fields."""
        if not self.reasoning:
            raise ValueError("Reasoning must be provided")
        if self.confidence < 0.5:
            raise ValueError(f"Confidence too low: {self.confidence}")
        return self

class OrchestratorRouter:
    """Pure functional router implementation."""
    
    def __init__(self, llm: BaseLLM, agents: Dict[str, Any]):
        """Initialize router with LLM and agent registry."""
        self.llm = llm
        # Create a non-streaming LLM specifically for routing decisions
        self.decision_llm = llm.bind(
            functions=[
                {
                    "name": "route_request",
                    "description": "Route a user request to the appropriate agent",
                    "parameters": RouterDecision.model_json_schema()
                }
            ],
            function_call={"name": "route_request"}
        )
        self.agents = agents
        self.system_prompt = SystemMessage(content="""
        You are a routing agent that directs requests to specialized agents.
        Available agents and their capabilities:
        
        - research-assistant: Web search, weather information, data analysis, fact verification
        - chatbot: General conversation, simple queries, clarifications
        - bg-task-agent: Long-running tasks, background processing, data generation
        
        Select the most appropriate agent based on:
        1. Required capabilities
        2. Task complexity
        3. Expected execution time
        4. Streaming requirements
        
        Provide your decision with:
        - next: The chosen agent
        - confidence: Score between 0-1
        - reasoning: Clear explanation
        - capabilities_matched: List of matched capabilities
        - fallback_agents: Alternative agents if primary fails

        IMPORTANT: If the task is complete or no further action is needed, set next to "FINISH".
        """)
    
    async def route(
        self,
        state: OrchestratorState,
        config: Dict[str, Any]
    ) -> BaseCommand:
        """Get routing decision with fallback handling."""
        try:
            # Validate we have agents registered
            if not self.agents:
                logger.error("No agents registered with router")
                return BaseCommand(
                    goto="chatbot",  # Default to chatbot as fallback
                    update={
                        "routing": {"current_agent": "chatbot", "decisions": [], "error_count": 0},
                        "streaming": {"is_streaming": False, "current_buffer": None, "buffers": {}}
                    }
                )

            # If we're already streaming, continue with the current agent
            if state.streaming.is_streaming:
                current_agent = state.routing.current_agent
                if current_agent:
                    # Check if streaming should continue
                    if state.streaming.current_buffer and state.streaming.current_buffer.is_complete:
                        decision = {
                            "next": "FINISH",
                            "confidence": 1.0,
                            "reasoning": "Streaming complete",
                            "capabilities_matched": [],
                            "fallback_agents": []
                        }
                    else:
                        decision = {
                            "next": current_agent,
                            "confidence": 1.0,
                            "reasoning": "Continue streaming",
                            "capabilities_matched": [],
                            "fallback_agents": []
                        }
                    return BaseCommand(
                        goto=decision["next"],
                        update={
                            "routing": state.routing.model_dump(),
                            "streaming": state.streaming.model_dump()
                        }
                    )
                
                # Handle streaming without current agent
                error_state = state.add_error("Streaming active but no current agent", "RouterError", None)
                raise RouterError("Streaming active but no current agent")
            
            # Get base decision
            decision = await self._get_base_decision(state)
            
            # Override decision for specific patterns
            last_message = state.messages[-1]
            if isinstance(last_message, HumanMessage):
                message_text = last_message.content.lower()
                
                # Check for weather-related queries
                weather_patterns = [
                    r"weather\s+in\s+\w+",
                    r"temperature\s+in\s+\w+",
                    r"how'?s\s+the\s+weather",
                    r"what'?s\s+the\s+weather",
                    r"forecast\s+for\s+\w+"
                ]
                
                if any(re.search(pattern, message_text) for pattern in weather_patterns):
                    logger.info("Detected weather query, routing to research assistant")
                    decision = RouterDecision(
                        next="research-assistant",
                        confidence=1.0,
                        reasoning="Weather-related query detected",
                        capabilities_matched=["weather", "web_search"],
                        fallback_agents=["chatbot"]
                    )
            
            # Convert decision to dict and update routing
            decision_dict = decision.model_dump()
            logger.info(f"Router decision: {decision_dict}")
            
            # Handle streaming continuation
            if state.streaming.is_streaming and state.routing.current_agent:
                logger.info(f"Continuing stream with agent: {state.routing.current_agent}")
                return BaseCommand(
                    goto=state.routing.current_agent,
                    update={
                        "routing": state.routing.model_dump(),
                        "streaming": state.streaming.model_dump()
                    }
                )
            
            # Create routing metadata with current agent
            routing_data = {
                "current_agent": decision.next,
                "decisions": state.routing.decisions + [decision_dict],
                "errors": state.routing.errors,
                "error_count": state.routing.error_count,
                "fallback_count": state.routing.fallback_count,
                "start_time": state.routing.start_time
            }
            
            # Update state preserving all fields
            updated_state = state.model_copy(update={
                "next": decision.next,
                "routing": routing_data
            })
            logger.info(f"Updated routing state: current_agent={updated_state.routing.current_agent}, next={updated_state.next}")
            
            return BaseCommand(
                goto=decision.next,
                update={
                    "routing": routing_data,
                    "streaming": state.streaming.model_dump()
                }
            )
            
        except (RouterError, AgentNotFoundError) as e:
            logger.error(f"Router error: {str(e)}")
            error_state = state.add_error(str(e), "RouterError", None).model_copy()
            
            fallback_decision = {
                "next": "chatbot",
                "confidence": 0.5,
                "reasoning": f"Router error fallback: {str(e)}",
                "capabilities_matched": [],
                "fallback_agents": []
            }
            
            # Update error state with fallback
            updated_state = error_state.model_copy(update={
                "next": "chatbot",
                "routing": error_state.routing.model_copy(
                    update={"current_agent": "chatbot"})
            })
            
            # Handle errors with safe fallback
            logger.warning(
                f"Using fallback agent due to router error: {updated_state.routing.current_agent}"
            )
            
            return BaseCommand(
                goto="chatbot",
                update={
                    "routing": updated_state.routing.model_dump(),
                    "streaming": {"is_streaming": False, "current_buffer": None, "buffers": {}}
                }
            )
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected error in router: {str(e)}")
            error_state = state.add_error(f"Unexpected router error: {str(e)}", "AgentExecutionError", None)
            return BaseCommand(
                goto="error_recovery",
                update={
                    "routing": error_state.routing.model_dump(),
                    "streaming": {"is_streaming": False, "current_buffer": None, "buffers": {}},
                    "errors": error_state.errors
                }
            )
    
    async def _get_base_decision(
        self,
        state: OrchestratorState
    ) -> RouterDecision:
        """Get initial routing decision."""
        # Prepare agent capabilities
        agents_info = "\n".join([
            f"- {id}: {meta.description} (capabilities: {', '.join(meta.capabilities)})"
            for id, meta in self.agents.items()
        ])
        
        messages = [
            self.system_prompt,
            SystemMessage(content=f"Available agents:\n{agents_info}"),
            HumanMessage(content=state.messages[-1].content)
        ]
        
        try:
            response = await self.decision_llm.ainvoke(messages)
            # Extract function call result and parse into RouterDecision
            function_call = response.additional_kwargs.get("function_call", {})
            if not function_call or "arguments" not in function_call:
                raise ValueError("No valid function call in response")
            return RouterDecision.model_validate_json(function_call["arguments"])
        except Exception as e:
            # Log the error and re-raise with more context
            logger.error(f"Error getting routing decision: {str(e)}")
            raise RouterError(f"Failed to get routing decision: {str(e)}")
    
    def _validate_decision(
        self,
        decision: RouterDecision,
        config: Dict[str, Any]
    ) -> bool:
        """Validate routing decision."""
        min_confidence = config.get("min_confidence", 0.7)
        
        # Basic validation
        if decision.confidence < min_confidence:
            return False
            
        # Agent availability
        if decision.next not in self.agents and decision.next != "FINISH":
            return False
            
        # Capability matching
        if config.get("require_capabilities", True):
            agent = self.agents.get(decision.next)
            if agent and not decision.capabilities_matched:
                return False
            
        return True
    
    async def _get_fallback_decision(
        self,
        state: OrchestratorState,
        failed_decision: RouterDecision
    ) -> RouterDecision:
        """Get fallback decision when primary fails."""
        # Use simpler prompt for fallback
        messages = [
            SystemMessage(content="Select fallback agent for failed request"),
            HumanMessage(content=f"""
            Original request: {state.messages[-1].content}
            Failed agent: {failed_decision.next}
            Reason: {failed_decision.reasoning}
            Available fallbacks: {failed_decision.fallback_agents}
            """)
        ]
        
        try:
            response = await self.decision_llm.ainvoke(messages)
            # Extract function call result and parse into RouterDecision
            function_call = response.additional_kwargs.get("function_call", {})
            if not function_call or "arguments" not in function_call:
                raise ValueError("No valid function call in response")
            decision = RouterDecision.model_validate_json(function_call["arguments"])
            decision.confidence *= 0.8  # Reduce confidence for fallback
            return decision
        except Exception as e:
            # Log the error and return a safe fallback
            logger.error(f"Error getting fallback decision: {str(e)}")
            return RouterDecision(
                next="chatbot",
                confidence=0.5,
                reasoning="Error in fallback routing",
                capabilities_matched=[],
                fallback_agents=[]
            )

def create_router(
    llm: Optional[BaseLLM] = None,
    agents: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None
) -> OrchestratorRouter:
    """Create router instance with defaults."""
    if not agents:
        # Default to chatbot if no agents provided
        agents = {
            "chatbot": {
                "description": "A simple chatbot for general conversation",
                "capabilities": ["chat", "conversation"]
            }
        }
    llm = llm or get_model(settings.DEFAULT_MODEL)    
    return OrchestratorRouter(llm, agents)

def should_continue(state: OrchestratorState) -> str:
    """Determine if orchestration should continue."""
    # Continue if we have a next agent to execute
    if state.next and state.next != "FINISH":
        return "continue"
        
    # Check for errors
    if state.routing.error_count > 3:  # Using MAX_ERRORS from state.py
        return "error_recovery"
        
    # End if FINISH was explicitly set
    if state.routing.current_agent == "FINISH":
        return "end"
        
    # Continue if streaming (unless buffer is complete)
    if state.streaming.is_streaming:
        if state.streaming.current_buffer and state.streaming.current_buffer.is_complete:
            return "end"
        return "continue"
    
    # End if no next agent
    return "end"
