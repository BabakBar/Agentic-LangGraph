"""Lightweight metrics interface for monitoring dashboard."""
from typing import Dict, Any

class MetricsInterface:
    """Interface for accessing metrics data."""
    
    @staticmethod
    def get_summary() -> Dict[str, Any]:
        """Get metrics summary from the agent service."""
        # This would normally fetch from the agent service API
        # For now return empty metrics structure
        return {
            "uptime_seconds": 0,
            "nodes": {},
            "router": {
                "decisions_by_agent": {},
                "average_confidence": 0,
                "fallback_routes": 0,
                "total_decisions": 0,
                "average_routing_time": 0
            },
            "streaming": {
                "successful_streams": 0,
                "total_streams": 0,
                "total_tokens": 0,
                "tokens_per_second": 0,
                "average_stream_time": 0
            }
        }

metrics = MetricsInterface()