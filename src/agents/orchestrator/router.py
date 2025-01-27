"""Router implementation with LLM-based routing and validation."""
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
import time
import functools

from pydantic import BaseModel, ValidationError, model_validator
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

from core import get_model, settings
from ..registry import AgentRegistry
from ..types import (
    OrchestratorState,
    RouterDecision,
    ValidatedRouterOutput,
    RoutingError,
    RoutingMetadata
)


# Prompt template for LLM routing
ROUTING_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("""
    Analyze the conversation history and select the most appropriate agent.
    
    Available Agents:
    {agents}
    
    Respond with JSON containing:
    - next: Agent ID to handle the request
    - reasoning: Brief explanation of your choice
    - confidence: Score between 0-1 indicating certainty
    - alternatives: List of other possible agents
    """),
    HumanMessage(content="Latest input: {input}")
])


def track_routing_metrics(fn):
    """Decorator to track routing performance metrics."""
    @functools.wraps(fn)
    async def wrapper(state: OrchestratorState, config: RunnableConfig):
        start = time.time()
        try:
            result = await fn(state, config)
            duration = time.time() - start
            # TODO: Add metric logging
            # log_metric("routing_time", duration)
            return result
        except Exception as e:
            # TODO: Add error logging
            # log_error("routing_error", str(e))
            raise
    return wrapper


class RoutingManager:
    """Manages routing decisions and fallback logic."""
    
    def __init__(self, registry: AgentRegistry, config: Dict[str, Any]):
        self.registry = registry
        self.config = config
        self.llm = get_model(config.get("model_name", "gpt-3.5-turbo"))
    
    async def get_llm_decision(self, state: OrchestratorState) -> RouterDecision:
        """Get routing decision from LLM."""
        agents = "\n".join([
            f"- {id}: {meta.description}"
            for id, meta in self.registry.metadata.items()
        ])
        
        prompt = ROUTING_PROMPT.format_messages(
            agents=agents,
            input=state.messages[-1].content
        )
        
        response = await self.llm.ainvoke(prompt)
        return RouterDecision.model_validate_json(response.content)
    
    def get_keyword_decision(self, user_input: str) -> RouterDecision:
        """Fallback to keyword-based routing."""
        task_keywords = {"process", "background", "task", "generate", "analyze"}
        if any(kw in user_input.lower() for kw in task_keywords):
            return RouterDecision(
                next="bg-task-agent",
                confidence=0.8,
                reasoning="Keyword match: task processing",
                alternatives=["research-assistant"]
            )
        return RouterDecision(
            next="research-assistant",
            confidence=0.6,
            reasoning="Default research agent",
            alternatives=["bg-task-agent"]
        )
    
    def validate_decision(
        self,
        decision: RouterDecision,
        min_confidence: float = 0.5
    ) -> Tuple[ValidatedRouterOutput, Optional[str]]:
        """Validate routing decision against registry and business rules."""
        try:
            if decision.confidence < min_confidence:
                return None, f"Low confidence score: {decision.confidence}"
                
            if not self.registry.has_agent(decision.next):
                return None, f"Invalid agent: {decision.next}"
                
            validated = ValidatedRouterOutput(
                next=decision.next,
                confidence=decision.confidence,
                reasoning=decision.reasoning,
                alternatives=decision.alternatives
            )
            return validated, None
            
        except Exception as e:
            return None, str(e)
    
    async def execute_fallback_chain(
        self,
        state: OrchestratorState,
        error: Optional[str] = None
    ) -> OrchestratorState:
        """Execute fallback chain when primary routing fails."""
        if error:
            state.add_error(error, state.messages[-1].content)
        
        # Try keyword-based routing
        decision = self.get_keyword_decision(state.messages[-1].content)
        validated, error = self.validate_decision(decision)
        
        if error:
            # Final fallback to research assistant
            state.routing.fallback_used = True
            return state.update_routing(RouterDecision(
                next="research-assistant",
                confidence=0.5,
                reasoning="Final fallback",
                alternatives=[]
            ))
        
        state.routing.fallback_used = True
        return state.update_routing(validated)


@track_routing_metrics
async def route_node(state: OrchestratorState, config: RunnableConfig) -> OrchestratorState:
    """Route to next agent using LLM with fallback chain."""
    try:
        registry = config.get("registry")
        if not registry:
            raise RoutingError("Registry not found in config")
        
        router = RoutingManager(registry, config)
        
        # Try LLM-based routing first
        try:
            decision = await router.get_llm_decision(state)
            validated, error = router.validate_decision(
                decision,
                min_confidence=config.get("min_confidence", 0.5)
            )
            
            if error:
                return await router.execute_fallback_chain(state, error)
            
            return state.update_routing(validated)
            
        except Exception as e:
            # Any LLM errors trigger fallback
            return await router.execute_fallback_chain(state, str(e))
        
    except Exception as e:
        # Critical errors default to research assistant
        state.add_error(str(e), state.messages[-1].content)
        return state.update_routing(RouterDecision(
            next="research-assistant",
            confidence=0.5,
            reasoning="Critical error fallback",
            alternatives=[]
        ))


def should_continue(state: OrchestratorState) -> str:
    """Determine if orchestration should continue."""
    if not state.routing.current_agent:
        return "end"
    
    # Continue if we have a valid agent
    return "continue"
