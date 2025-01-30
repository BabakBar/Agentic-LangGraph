"""Pure graph construction for orchestrator."""
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, Optional, Any, List
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field, model_validator
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command

from ...common.types import AgentLike, OrchestratorState
from ..metrics import metrics, NodeMetrics
from .router import create_router, should_continue

logger = logging.getLogger(__name__)

class NodeConfig(BaseModel):
    """Configuration for graph nodes."""
    max_retries: int = Field(default=3, ge=1)
    timeout_seconds: int = Field(default=30, ge=1)
    enable_metrics: bool = Field(default=True)
    checkpoint_dir: Optional[Path] = None

class GraphNode(ABC):
    """Base class for graph nodes."""
    def __init__(self, config: NodeConfig):
        self.config = config
        self.node_metrics: Optional[NodeMetrics] = None
        
    async def execute(self, state: OrchestratorState) -> Command:
        """Execute node with metrics and error handling."""
        # Start metrics collection
        if self.config.enable_metrics:
            self.node_metrics = metrics.start_node(self.__class__.__name__)
            
        try:
            result = await self._execute(state)
            
            # Record success metrics
            if self.node_metrics:
                self.node_metrics.complete(success=True)
                
            return result
            
        except Exception as e:
            # Record error metrics
            if self.node_metrics:
                self.node_metrics.complete(success=False)
                
            return await self._handle_error(state, e)
            
    @abstractmethod
    async def _execute(self, state: OrchestratorState) -> Command:
        """Implement actual node logic."""
        pass
        
    async def _handle_error(self, state: OrchestratorState, error: Exception) -> Command:
        """Handle node execution errors."""
        state.add_error(str(error), state.messages[-1].content)
        if self.node_metrics and self.node_metrics.error_count > self.config.max_retries:
            return Command(goto="error_recovery")
        return Command(goto="router")

class RouterNode(GraphNode):
    """Node for routing decisions."""
    def __init__(self, router: Any, config: NodeConfig):
        super().__init__(config)
        self.router = router
        
    async def _execute(self, state: OrchestratorState) -> Command:
        """Execute routing logic."""
        logger.debug("RouterNode executing with state: %s", state)
        start_time = datetime.now()
        decision = await self.router.route(state, {
            "min_confidence": 0.7,
            "require_capabilities": True
        })
        
        # Record routing metrics
        if self.config.enable_metrics:
            routing_time = (datetime.now() - start_time).total_seconds()
            metrics.add_router_decision(
                agent=decision.next_agent,
                confidence=decision.confidence,
                routing_time=routing_time,
                is_fallback=state.routing.fallback_used
            )
        logger.debug("Routing decision: %s", decision)
            
        return Command(
            goto="executor" if decision.next_agent != "FINISH" else END,
            update={"routing": state.update_routing(decision)}
        )

class AgentExecutorNode(GraphNode):
    """Node for executing agent actions."""
    def __init__(self, agents: Dict[str, AgentLike], config: NodeConfig):
        super().__init__(config)
        self.agents = agents
        
    async def _execute(self, state: OrchestratorState) -> Command:
        """Execute selected agent."""
        logger.debug("AgentExecutorNode executing with agent: %s", state.routing.current_agent)
        agent = self.agents[state.routing.current_agent]
        # Convert state to dict while preserving messages
        agent_state = {
            "messages": list(state.messages),  # Convert frozen list to mutable
            "config": state.model_dump().get("config", {})
        }
        result = await agent.ainvoke(agent_state)
        logger.debug("Agent execution result: %s", result)
        
        # Handle streaming
        if state.streaming.is_streaming:
            return Command(goto="stream_handler")
            
        return Command(goto="router")

class StreamingNode(GraphNode):
    """Node for handling streaming responses."""
    async def _execute(self, state: OrchestratorState) -> Command:
        """Handle streaming state."""
        logger.debug("StreamingNode state: %s", state.streaming)
        if not state.streaming.is_streaming:
            return Command(goto="router")
            
        buffer = state.streaming.current_buffer
        if buffer and buffer.is_complete:
            # Record streaming metrics
            if self.config.enable_metrics:
                metrics.add_stream_metrics(
                    stream_time=self.node_metrics.execution_time if self.node_metrics else 0,
                    token_count=len(buffer.tokens),
                    success=not bool(buffer.error)
                )
                
            state.end_stream()
            return Command(goto="router")
            
        return Command(goto="executor")

class ErrorRecoveryNode(GraphNode):
    """Node for handling errors."""
    async def _execute(self, state: OrchestratorState) -> Command:
        """Handle error recovery."""
        logger.debug("ErrorRecoveryNode executing with error count: %d", state.routing.error_count)
        if state.routing.error_count > self.config.max_retries:
            # Switch to fallback agent
            return Command(
                goto="router",
                update={
                    "routing": state.update_routing({
                        "next_agent": "chatbot",
                        "confidence": 0.5,
                        "reasoning": "Error recovery fallback"
                    })
                }
            )
        return Command(goto="router")

class OrchestratorGraph:
    """Builder for orchestrator graph."""
    def __init__(self, config: NodeConfig):
        self.config = config
        self.graph = StateGraph(OrchestratorState)
        
    def add_router(self, router: Any) -> None:
        """Add routing node."""
        node = RouterNode(router, self.config)
        self.graph.add_node("router", node.execute)
        
    def add_executor(self, agents: Dict[str, AgentLike]) -> None:
        """Add agent executor node."""
        node = AgentExecutorNode(agents, self.config)
        self.graph.add_node("executor", node.execute)
        
    def add_streaming(self) -> None:
        """Add streaming handler node."""
        node = StreamingNode(self.config)
        self.graph.add_node("stream_handler", node.execute)
        
    def add_error_recovery(self) -> None:
        """Add error recovery node."""
        node = ErrorRecoveryNode(self.config)
        self.graph.add_node("error_recovery", node.execute)
        
    def add_edges(self) -> None:
        """Add graph edges."""
        # Router edges
        self.graph.add_conditional_edges(
            "router",
            should_continue,
            {
                "continue": "executor",
                "end": END,
                "error_recovery": "error_recovery"
            }
        )
        
        # Executor edges
        self.graph.add_conditional_edges(
            "executor",
            lambda s: (
                "stream_handler" if s.streaming.is_streaming
                else "error_recovery" if s.routing.error_count > self.config.max_retries
                else "router"
            ),
            {
                "stream_handler": "stream_handler",
                "error_recovery": "error_recovery",
                "router": "router"
            }
        )
        
        # Streaming edges
        self.graph.add_conditional_edges(
            "stream_handler",
            lambda s: (
                "error_recovery" if s.routing.error_count > self.config.max_retries
                else "router" if not s.streaming.is_streaming
                else "executor"
            ),
            {
                "error_recovery": "error_recovery",
                "router": "router",
                "executor": "executor"
            }
        )
        
        # Error recovery edges
        self.graph.add_edge("error_recovery", "router")
        
    def build(self) -> StateGraph:
        """Build final graph."""
        return self.graph.compile()

def build_orchestrator(
    config: Optional[NodeConfig] = None,
    base_agents: Optional[Dict[str, AgentLike]] = None
) -> StateGraph:
    """Create orchestrator graph."""
    config = config or NodeConfig()
    base_agents = base_agents or {}
    
    # Create graph builder
    builder = OrchestratorGraph(config)
    
    # Create router
    router = create_router(agents=base_agents)
    
    # Add nodes
    builder.add_router(router)
    builder.add_executor(base_agents)
    builder.add_streaming()
    builder.add_error_recovery()
    
    # Add edges
    builder.add_edges()
    
    return builder.build()
