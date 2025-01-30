"""Monitoring dashboard for orchestrator metrics."""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

from ..agents.core.metrics import metrics

def draw_metrics_dashboard():
    """Draw the monitoring dashboard."""
    st.title("Orchestrator Monitoring")
    
    # Get current metrics
    summary = metrics.get_summary()
    
    # System Overview
    with st.expander("🔄 System Overview", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Uptime",
                f"{summary['uptime_seconds'] / 3600:.1f}h"
            )
            
        with col2:
            total_success = sum(
                m["success_count"]
                for m in summary["nodes"].values()
            )
            st.metric("Total Successful Operations", total_success)
            
        with col3:
            total_errors = sum(
                m["error_count"]
                for m in summary["nodes"].values()
            )
            st.metric("Total Errors", total_errors)
    
    # Routing Performance
    with st.expander("🔀 Routing Performance", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            # Agent Distribution Pie Chart
            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=list(summary["router"]["decisions_by_agent"].keys()),
                        values=list(summary["router"]["decisions_by_agent"].values()),
                        hole=.3
                    )
                ]
            )
            fig.update_layout(title="Agent Distribution")
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Routing Metrics
            st.metric(
                "Average Confidence",
                f"{summary['router']['average_confidence']:.2f}"
            )
            st.metric(
                "Fallback Rate",
                f"{summary['router']['fallback_routes'] / max(summary['router']['total_decisions'], 1):.1%}"
            )
            st.metric(
                "Average Routing Time",
                f"{summary['router']['average_routing_time']*1000:.0f}ms"
            )
    
    # Streaming Performance
    with st.expander("📊 Streaming Performance", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            # Success Rate Gauge
            success_rate = (
                summary["streaming"]["successful_streams"]
                / max(summary["streaming"]["total_streams"], 1)
            )
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=success_rate * 100,
                title={"text": "Streaming Success Rate"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "darkgreen"},
                    "steps": [
                        {"range": [0, 50], "color": "lightgray"},
                        {"range": [50, 80], "color": "gray"},
                        {"range": [80, 100], "color": "lightgreen"}
                    ],
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Streaming Metrics
            st.metric(
                "Total Tokens",
                summary["streaming"]["total_tokens"]
            )
            st.metric(
                "Tokens/Second",
                f"{summary['streaming']['tokens_per_second']:.1f}"
            )
            st.metric(
                "Average Stream Time",
                f"{summary['streaming']['average_stream_time']*1000:.0f}ms"
            )
    
    # Node Performance
    with st.expander("🔍 Node Performance", expanded=True):
        # Execution Time Bar Chart
        exec_times = {
            name: metrics["execution_time"]
            for name, metrics in summary["nodes"].items()
            if metrics["execution_time"] is not None
        }
        if exec_times:
            fig = go.Figure([
                go.Bar(
                    x=list(exec_times.keys()),
                    y=list(exec_times.values()),
                    text=[f"{t*1000:.0f}ms" for t in exec_times.values()],
                    textposition="auto"
                )
            ])
            fig.update_layout(
                title="Node Execution Times",
                yaxis_title="Seconds"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Success/Error Rates
        for node_name, node_metrics in summary["nodes"].items():
            with st.container():
                st.subheader(f"Node: {node_name}")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Success Count",
                        node_metrics["success_count"]
                    )
                    
                with col2:
                    st.metric(
                        "Error Count",
                        node_metrics["error_count"]
                    )
                    
                with col3:
                    success_rate = (
                        node_metrics["success_count"]
                        / max(
                            node_metrics["success_count"]
                            + node_metrics["error_count"],
                            1
                        )
                    )
                    st.metric(
                        "Success Rate",
                        f"{success_rate:.1%}"
                    )
                
                # Custom Metrics
                if node_metrics["custom_metrics"]:
                    st.write("Custom Metrics:")
                    for name, value in node_metrics["custom_metrics"].items():
                        st.metric(name, f"{value:.2f}")
    
    # Auto-refresh
    if st.toggle("Auto-refresh", value=False):
        st.empty()
        st.rerun()