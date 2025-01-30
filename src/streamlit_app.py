"""Streamlit app for interacting with the orchestrator agent."""
import asyncio
import os
from collections.abc import AsyncGenerator
from typing import Optional

import streamlit as st
from dotenv import load_dotenv
from pydantic import ValidationError
from streamlit.runtime.scriptrunner import get_script_run_ctx

from client import AgentClient, AgentClientError
from schema import ChatHistory, ChatMessage
from schema.task_data import TaskData, TaskDataStatus
from core.monitoring_dashboard import draw_metrics_dashboard

APP_TITLE = "Agentic Orixa"
WELCOME_MESSAGE = """
Welcome! I'm an AI orchestrator that can help you with various tasks. I'll automatically:
- Route your questions to the most appropriate agent
- Handle web searches and calculations
- Process background tasks
- Manage streaming responses

Just ask me anything, and I'll take care of the rest!
"""

async def main() -> None:
    """Main app entry point."""
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
                st.session_state.agent_client = AgentClient(base_url=agent_url)
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
        messages: list[ChatMessage] = st.session_state.messages

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
            messages.append(ChatMessage(type="human", content=user_input))
            st.chat_message("human").write(user_input)
            try:
                if use_streaming:
                    stream = agent_client.astream(
                        message=user_input,
                        model=model,
                        thread_id=st.session_state.thread_id,
                        agent="orchestrator"  # Always use orchestrator
                    )
                    await draw_messages(stream, is_new=True)
                else:
                    response = await agent_client.ainvoke(
                        message=user_input,
                        model=model,
                        thread_id=st.session_state.thread_id,
                        agent="orchestrator"  # Always use orchestrator
                    )
                    messages.append(response)
                    st.chat_message("ai").write(response.content)
                st.rerun()
            except AgentClientError as e:
                st.error(f"Error: {str(e)}")
                if "routing" in str(e).lower():
                    st.info("The orchestrator will try to recover and route to a fallback agent.")
                st.stop()

        # Show feedback widget
        if len(messages) > 0 and st.session_state.last_message:
            with st.session_state.last_message:
                await handle_feedback()

    with tab2:  # Monitoring Dashboard
        draw_metrics_dashboard()

async def draw_messages(
    messages_agen: AsyncGenerator[ChatMessage | str, None],
    is_new: bool = False,
) -> None:
    """Draw chat messages with streaming and tool call support."""
    # Track message state
    last_message_type = None
    st.session_state.last_message = None
    streaming_content = ""
    streaming_placeholder = None
    routing_status = None

    # Process messages
    while msg := await anext(messages_agen, None):
        # Handle streaming tokens
        if isinstance(msg, str):
            if not streaming_placeholder:
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")
                with st.session_state.last_message:
                    streaming_placeholder = st.empty()
                    routing_status = st.status("🤖 Orchestrator Processing", state="running")

            streaming_content += msg
            streaming_placeholder.write(streaming_content)
            continue

        if not isinstance(msg, ChatMessage):
            st.error(f"Unexpected message type: {type(msg)}")
            st.write(msg)
            st.stop()

        # Handle different message types
        match msg.type:
            case "human":
                last_message_type = "human"
                st.chat_message("human").write(msg.content)

            case "ai":
                if is_new:
                    st.session_state.messages.append(msg)

                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")

                with st.session_state.last_message:
                    # Update routing status if present
                    if routing_status and msg.metadata.get("routing_decision"):
                        decision = msg.metadata["routing_decision"]
                        routing_status.update(
                            label=f"🤖 Routed to: {decision['next_agent']}",
                            state="complete",
                            expanded=False
                        )
                        routing_status.markdown(f"""
                        **Confidence:** {decision['confidence']:.2f}
                        **Reason:** {decision['reasoning']}
                        """)

                    # Write message content
                    if msg.content:
                        if streaming_placeholder:
                            streaming_placeholder.write(msg.content)
                            streaming_content = ""
                            streaming_placeholder = None
                        else:
                            st.write(msg.content)

                    # Handle tool calls
                    if msg.tool_calls:
                        call_results = {}
                        for tool_call in msg.tool_calls:
                            status = st.status(
                                f"""🔧 Tool: {tool_call["name"]}""",
                                state="running" if is_new else "complete",
                            )
                            call_results[tool_call["id"]] = status
                            status.write("Input:")
                            status.write(tool_call["args"])

                        for _ in range(len(call_results)):
                            tool_result: ChatMessage = await anext(messages_agen)
                            if tool_result.type != "tool":
                                st.error(f"Unexpected message type: {tool_result.type}")
                                st.write(tool_result)
                                st.stop()

                            if is_new:
                                st.session_state.messages.append(tool_result)
                            status = call_results[tool_result.tool_call_id]
                            status.write("Output:")
                            status.write(tool_result.content)
                            status.update(state="complete")

            case "custom":
                try:
                    task_data: TaskData = TaskData.model_validate(msg.custom_data)
                except ValidationError:
                    st.error("Invalid custom data received")
                    st.write(msg.custom_data)
                    st.stop()

                if is_new:
                    st.session_state.messages.append(msg)

                if last_message_type != "task":
                    last_message_type = "task"
                    st.session_state.last_message = st.chat_message(
                        name="task",
                        avatar="🔄"
                    )
                    with st.session_state.last_message:
                        status = TaskDataStatus()

                status.add_and_draw_task_data(task_data)

            case _:
                st.error(f"Unknown message type: {msg.type}")
                st.write(msg)
                st.stop()

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
