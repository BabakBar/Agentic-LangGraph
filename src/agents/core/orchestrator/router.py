"""Router implementation with LLM-based routing and validation."""
from typing import Dict, Any, Optional, Tuple, List, Literal
from datetime import datetime
import time
import functools

from pydantic import BaseModel, Field, model_validator
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseLLM

from core.llm import get_model
from core.settings import settings
from ..agent_registry import AgentRegistry
from ...common.types import OrchestratorState

# Define available agents as literals for type safety
AgentType = Literal[
    "research-assistant",
    "chatbot",
    "bg-task-agent",
    "FINISH"
]

class RouterDecision(BaseModel):
    """Structured output for routing decisions."""
    next_agent: AgentType
    confidence: float = Field(ge=0.0, le=1.0)
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
        self.llm = llm.with_structured_output(RouterDecision)
        self.agents = agents
        self.system_prompt = SystemMessage(content="""
        You are a routing agent that directs requests to specialized agents.
        Available agents and their capabilities:
        
        - research-assistant: Web search, data analysis, fact verification
        - chatbot: General conversation, simple queries, clarifications
        - bg-task-agent: Long-running tasks, background processing, data generation
        
        Select the most appropriate agent based on:
        1. Required capabilities
        2. Task complexity
        3. Expected execution time
        4. Streaming requirements
        
        Provide your decision with:
        - next_agent: The chosen agent
        - confidence: Score between 0-1
        - reasoning: Clear explanation
        - capabilities_matched: List of matched capabilities
        - fallback_agents: Alternative agents if primary fails
        """)
    
    async def route(
        self,
        state: OrchestratorState,
        config: Dict[str, Any]
    ) -> RouterDecision:
        """Get routing decision with fallback handling."""
        try:
            # Handle streaming continuation
            if state.streaming.is_streaming:
                return RouterDecision(
                    next_agent=state.routing.current_agent,
                    confidence=1.0,
                    reasoning="Continue streaming with current agent",
                    capabilities_matched=["streaming"],
                    fallback_agents=[]
                )
            
            # Get base decision
            decision = await self._get_base_decision(state)
            
            # Validate decision
            if not self._validate_decision(decision, config):
                decision = await self._get_fallback_decision(state, decision)
            
            return decision
            
        except Exception as e:
            # Handle errors with safe fallback
            return RouterDecision(
                next_agent="chatbot",
                confidence=0.5,
                reasoning=f"Error in routing: {str(e)}",
                capabilities_matched=[],
                fallback_agents=["research-assistant"]
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
        
        return await self.llm.ainvoke(messages)
    
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
        if decision.next_agent not in self.agents and decision.next_agent != "FINISH":
            return False
            
        # Capability matching
        if config.get("require_capabilities", True):
            agent = self.agents.get(decision.next_agent)
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
            Failed agent: {failed_decision.next_agent}
            Reason: {failed_decision.reasoning}
            Available fallbacks: {failed_decision.fallback_agents}
            """)
        ]
        
        decision = await self.llm.ainvoke(messages)
        decision.confidence *= 0.8  # Reduce confidence for fallback
        return decision

def create_router(
    llm: Optional[BaseLLM] = None,
    agents: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None
) -> OrchestratorRouter:
    """Create router instance with defaults."""
    llm = llm or get_model(settings.DEFAULT_MODEL)
    agents = agents or {}
    config = config or {}
    
    return OrchestratorRouter(llm, agents)

def should_continue(state: OrchestratorState) -> str:
    """Determine if orchestration should continue."""
    # End if no current agent
    if not state.routing.current_agent:
        return "end"
        
    # Continue if streaming
    if state.streaming.is_streaming:
        return "continue"
        
    # Continue if valid agent
    return "continue" if state.routing.current_agent != "FINISH" else "end"
