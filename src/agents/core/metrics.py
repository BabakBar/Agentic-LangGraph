"""Metrics collection and monitoring for orchestrator."""
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

@dataclass
class NodeMetrics:
    """Metrics for a single node execution."""
    node_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    execution_time: Optional[float] = None
    error_count: int = 0
    success_count: int = 0
    custom_metrics: Dict[str, float] = field(default_factory=dict)

    def complete(self, success: bool = True) -> None:
        """Complete node execution metrics."""
        self.end_time = datetime.now()
        self.execution_time = (self.end_time - self.start_time).total_seconds()
        if success:
            self.success_count += 1
        else:
            self.error_count += 1

    def add_custom_metric(self, name: str, value: float) -> None:
        """Add a custom metric."""
        self.custom_metrics[name] = value

@dataclass
class RouterMetrics:
    """Metrics for router decisions."""
    total_decisions: int = 0
    successful_routes: int = 0
    fallback_routes: int = 0
    average_confidence: float = 0.0
    decisions_by_agent: Dict[str, int] = field(default_factory=dict)
    average_routing_time: float = 0.0
    routing_times: List[float] = field(default_factory=list)

    def add_decision(
        self,
        agent: str,
        confidence: float,
        routing_time: float,
        is_fallback: bool = False
    ) -> None:
        """Add a routing decision."""
        self.total_decisions += 1
        if is_fallback:
            self.fallback_routes += 1
        else:
            self.successful_routes += 1

        # Update agent counts
        self.decisions_by_agent[agent] = self.decisions_by_agent.get(agent, 0) + 1

        # Update confidence
        self.average_confidence = (
            (self.average_confidence * (self.total_decisions - 1) + confidence)
            / self.total_decisions
        )

        # Update routing time
        self.routing_times.append(routing_time)
        self.average_routing_time = sum(self.routing_times) / len(self.routing_times)

@dataclass
class StreamingMetrics:
    """Metrics for streaming operations."""
    total_streams: int = 0
    successful_streams: int = 0
    failed_streams: int = 0
    average_stream_time: float = 0.0
    stream_times: List[float] = field(default_factory=list)
    total_tokens: int = 0
    tokens_per_second: float = 0.0

    def add_stream(
        self,
        stream_time: float,
        token_count: int,
        success: bool = True
    ) -> None:
        """Add streaming metrics."""
        self.total_streams += 1
        if success:
            self.successful_streams += 1
        else:
            self.failed_streams += 1

        # Update timing metrics
        self.stream_times.append(stream_time)
        self.average_stream_time = sum(self.stream_times) / len(self.stream_times)

        # Update token metrics
        self.total_tokens += token_count
        if stream_time > 0:
            self.tokens_per_second = token_count / stream_time

class MetricsCollector:
    """Collect and aggregate metrics."""
    def __init__(self):
        self.node_metrics: Dict[str, NodeMetrics] = {}
        self.router_metrics = RouterMetrics()
        self.streaming_metrics = StreamingMetrics()
        self.start_time = datetime.now()

    def start_node(self, node_name: str) -> NodeMetrics:
        """Start collecting metrics for a node."""
        metrics = NodeMetrics(
            node_name=node_name,
            start_time=datetime.now()
        )
        self.node_metrics[node_name] = metrics
        return metrics

    def add_router_decision(
        self,
        agent: str,
        confidence: float,
        routing_time: float,
        is_fallback: bool = False
    ) -> None:
        """Add a routing decision metric."""
        self.router_metrics.add_decision(
            agent=agent,
            confidence=confidence,
            routing_time=routing_time,
            is_fallback=is_fallback
        )

    def add_stream_metrics(
        self,
        stream_time: float,
        token_count: int,
        success: bool = True
    ) -> None:
        """Add streaming metrics."""
        self.streaming_metrics.add_stream(
            stream_time=stream_time,
            token_count=token_count,
            success=success
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        total_time = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "uptime_seconds": total_time,
            "nodes": {
                name: {
                    "execution_time": metrics.execution_time,
                    "error_count": metrics.error_count,
                    "success_count": metrics.success_count,
                    "custom_metrics": metrics.custom_metrics
                }
                for name, metrics in self.node_metrics.items()
            },
            "router": {
                "total_decisions": self.router_metrics.total_decisions,
                "successful_routes": self.router_metrics.successful_routes,
                "fallback_routes": self.router_metrics.fallback_routes,
                "average_confidence": self.router_metrics.average_confidence,
                "decisions_by_agent": self.router_metrics.decisions_by_agent,
                "average_routing_time": self.router_metrics.average_routing_time
            },
            "streaming": {
                "total_streams": self.streaming_metrics.total_streams,
                "successful_streams": self.streaming_metrics.successful_streams,
                "failed_streams": self.streaming_metrics.failed_streams,
                "average_stream_time": self.streaming_metrics.average_stream_time,
                "total_tokens": self.streaming_metrics.total_tokens,
                "tokens_per_second": self.streaming_metrics.tokens_per_second
            }
        }

# Global metrics collector instance
metrics = MetricsCollector()