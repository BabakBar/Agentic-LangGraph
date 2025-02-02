import logging
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph

from core import get_model, settings

logger = logging.getLogger(__name__)


class AgentState(MessagesState, total=False):
    """`total=False` is PEP589 specs.

    documentation: https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    preprocessor = RunnableLambda(
        lambda state: state.get("messages", state.get("input", {}).get("messages", [])),
        name="StateModifier",
    )
    return preprocessor | model


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m)
    response = await model_runnable.ainvoke(state, config)

    logger.info("Chatbot processing response")

    # Get current state
    streaming_state = state.get("streaming", {"is_streaming": False, "buffers": {}})
    routing_state = state.get("routing", {})
    
    # Determine if we should terminate
    is_streaming = streaming_state.get("is_streaming", False)
    if is_streaming:
        current_buffer = streaming_state.get("current_buffer")
        should_terminate = current_buffer and current_buffer.get("is_complete", False)
    else:
        should_terminate = True  # Always terminate for non-streaming responses

    # Prepare result state
    result_state = {
        "messages": [response],
        "routing": routing_state,
        "streaming": streaming_state,
        "next": "FINISH" if should_terminate else None,
        "tool_state": state.get("tool_state", {}),
        "schema_version": state.get("schema_version", "2.0")
    }

    logger.info(
        f"Chatbot completed. Streaming: {is_streaming}, "
        f"Should terminate: {should_terminate}, "
        f"Next state: {result_state['next']}"
    )

    return result_state


# Define the graph
agent = StateGraph(AgentState)
agent.add_node("model", acall_model)
agent.set_entry_point("model")

# End the conversation after model response
agent.add_edge("model", END)

chatbot = agent.compile(
    checkpointer=MemorySaver(),
)
