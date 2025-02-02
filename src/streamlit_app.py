"""Streamlit app for interacting with the orchestrator agent."""
import asyncio
import os
import logging
from collections.abc import AsyncGenerator
from typing import Optional

import streamlit as st
from dotenv import load_dotenv
from pydantic import ValidationError
from streamlit.runtime.scriptrunner import get_script_run_ctx

from client import AgentClient, AgentClientError
from schema import ChatHistory, ChatMessage
from schema.task_data import TaskData, TaskDataStatus
from client.monitoring_dashboard import draw_metrics_dashboard
from client.logging_config import setup_logging

# Set up logging with environment LOG_LEVEL
setup_logging(os.getenv("LOG_LEVEL", "INFO"))

logging.getLogger('watchdog').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

APP_TITLE = "Agentic Orixa"
WELCOME_MESSAGE = """
Just ask me anything!
"""

async def main() -> None:
    """Main app entry point."""
    logger.info("Starting Streamlit application")
    st.set_page_config(
        page_title=APP_TITLE,
        layout="wide",
        menu_items={},
    )

    # Hide the streamlit upper-right chrome
    st.html(
        """
        <style>
        [data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
            }
        </style>
        """,
    )
    if st.get_option("client.toolbarMode") != "minimal":
        st.set_option("client.toolbarMode", "minimal")
        await asyncio.sleep(0.1)
        st.rerun()

    # Initialize session state variables
    if "last_message" not in st.session_state:
        st.session_state.last_message = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_feedback" not in st.session_state:
        st.session_state.last_feedback = (None, None)

    # Initialize agent client
    if "agent_client" not in st.session_state:
        load_dotenv()
        agent_url = os.getenv("AGENT_URL")
        if not agent_url:
            host = os.getenv("HOST", "0.0.0.0")
            port = os.getenv("PORT", 80)
            agent_url = f"http://{host}:{port}"
        try:
            with st.spinner("Connecting to agent service..."):
                st.session_state.agent_client = AgentClient(base_url=agent_url, agent="orchestrator")
        except AgentClientError as e:
            st.error(f"Error connecting to agent service: {e}")
            st.markdown("The service might be booting up. Try again in a few seconds.")
            st.stop()
    agent_client: AgentClient = st.session_state.agent_client

    # Initialize thread state
    if "thread_id" not in st.session_state:
        thread_id = st.query_params.get("thread_id")
        if not thread_id:
            thread_id = get_script_run_ctx().session_id
            messages = []
        else:
            try:
                messages: ChatHistory = agent_client.get_history(thread_id=thread_id).messages
            except AgentClientError:
                st.error("No message history found for this Thread ID.")
                messages = []
        st.session_state.messages = messages
        st.session_state.thread_id = thread_id

    # Sidebar
    with st.sidebar:
        st.header(APP_TITLE)
        ""
        "AI orchestrator powered by LangGraph"
        with st.expander("i️ About", expanded=False):
            st.markdown("""
            This AI orchestrator:
            - Routes tasks to specialized agents
            - Handles streaming responses
            - Manages tool execution
            - Provides error recovery
            """)
        with st.popover("⚙️ Settings", use_container_width=True):
            model_idx = agent_client.info.models.index(agent_client.info.default_model)
            model = st.selectbox("LLM to use", options=agent_client.info.models, index=model_idx)
            use_streaming = st.toggle("Stream results", value=True)

    # Tab interface
    tab1, tab2 = st.tabs(["💬 Chat", "📊 Monitoring"])

    with tab1:  # Chat Interface
        messages = st.session_state.messages

        if len(messages) == 0:
            with st.chat_message("ai"):
                st.markdown(WELCOME_MESSAGE)

        # Draw messages
        async def amessage_iter() -> AsyncGenerator[ChatMessage, None]:
            for m in messages:
                yield m

        await draw_messages(amessage_iter())

        # Handle new user input
        if user_input := st.chat_input():
            thread_id = st.session_state.thread_id
            messages.append(ChatMessage(type="human", content=user_input))
            st.chat_message("human").write(user_input)
            
            message_container = st.chat_message("ai")
            st.session_state.last_message = message_container
            
            try:
                with message_container:
                    if use_streaming:
                        try:
                            stream = agent_client.astream(
                                message={"message": user_input, "model": model, "thread_id": thread_id}
                            )
                            await draw_messages(stream, is_new=True)
                        except asyncio.CancelledError:
                            logger.info("Stream cancelled by user")
                            return
                    else:
                        try:
                            response = await agent_client.ainvoke(
                                message=user_input,
                                model=model,
                                thread_id=thread_id,
                            )
                            messages.append(response)
                            st.write(response.content)
                        except asyncio.CancelledError:
                            logger.info("Request cancelled by user")
                            return
            except AgentClientError as e:
                st.error(f"Error: {str(e)}")
                if "routing" in str(e).lower():
                    st.info("The orchestrator will try to recover and route to a fallback agent.")
                st.stop()

        # Show feedback widget
        if len(messages) > 0 and st.session_state.last_message is not None:
            with st.session_state.last_message:
                await handle_feedback()

    with tab2:  # Monitoring Dashboard
        draw_metrics_dashboard()

async def draw_messages(messages_agen, is_new=False):
    """Draw messages from the async generator."""
    try:
        placeholder = st.empty() 
        current_content = ""
        
        async for msg in messages_agen:
            try:
                # Handle both dict and ChatMessage types
                msg_type = msg.type if isinstance(msg, ChatMessage) else msg.get("type")
                msg_content = msg.content if isinstance(msg, ChatMessage) else msg.get("content", "")
                
                if msg_type == "error":
                    st.error(f"Error: {msg_content}")
                    continue
                    
                # Handle other message types
                if msg_type in ["message", "token", "ai"]:
                    # Extract content from structured messages
                    if isinstance(msg_content, dict) and "content" in msg_content:
                        msg_content = msg_content["content"]
                    elif isinstance(msg_content, dict) and "text" in msg_content:
                        msg_content = msg_content["text"]
                    
                    # Append new content
                    if msg_content:
                        current_content += str(msg_content)
                    
                    # Update display
                    with placeholder.container():
                        st.markdown(current_content)
                        
                # Allow other tasks to run
                await asyncio.sleep(0)
                        
            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=True)
                continue
                    
    except asyncio.CancelledError:
        logger.info("Message stream cancelled")
        return
    except Exception as e:
        st.error(f"Error processing messages: {str(e)}")
        logger.error(f"Error in draw_messages: {e}", exc_info=True)

async def handle_feedback() -> None:
    """Handle user feedback collection."""
    if "last_feedback" not in st.session_state:
        st.session_state.last_feedback = (None, None)

    latest_run_id = st.session_state.messages[-1].run_id
    feedback = st.feedback("stars", key=latest_run_id)

    if feedback is not None and (latest_run_id, feedback) != st.session_state.last_feedback:
        normalized_score = (feedback + 1) / 5.0
        agent_client: AgentClient = st.session_state.agent_client
        try:
            await agent_client.acreate_feedback(
                run_id=latest_run_id,
                key="human-feedback-stars",
                score=normalized_score,
                kwargs={"comment": "In-line human feedback"},
            )
        except AgentClientError as e:
            st.error(f"Error recording feedback: {e}")
            st.stop()
        st.session_state.last_feedback = (latest_run_id, feedback)
        st.toast("Feedback recorded", icon="⭐")

if __name__ == "__main__":
    asyncio.run(main())
