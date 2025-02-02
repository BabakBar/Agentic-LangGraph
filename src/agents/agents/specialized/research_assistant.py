from datetime import datetime
import logging
from typing import Literal

from langchain_community.tools import DuckDuckGoSearchResults, OpenWeatherMapQueryRun
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode

from agents.tools.calculator import calculator
from core import get_model, settings

logger = logging.getLogger(__name__)

class AgentState(MessagesState, total=False):
    """`total=False` is PEP589 specs.

    documentation: https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """

    remaining_steps: RemainingSteps


# Initialize tools list
tools = []

# Add web search tool
try:
    web_search = DuckDuckGoSearchResults(name="WebSearch")
    tools.append(web_search)
    logger.info("Web search tool initialized")
except Exception as e:
    logger.error(f"Failed to initialize web search tool: {e}")

# Add calculator tool
tools.append(calculator)

# Add weather tool if API key is available
if settings.OPENWEATHERMAP_API_KEY:
    wrapper = OpenWeatherMapAPIWrapper(
        openweathermap_api_key=settings.OPENWEATHERMAP_API_KEY.get_secret_value()
    )
    tools.append(OpenWeatherMapQueryRun(name="Weather", api_wrapper=wrapper))
    logger.info("Weather tool initialized")

current_date = datetime.now().strftime("%B %d, %Y")
instructions = f"""
    You are a helpful research assistant with the ability to search the web and use other tools.
    Today's date is {current_date}.

    NOTE: THE USER CAN'T SEE THE TOOL RESPONSE.

    A few things to remember:
    - Please include markdown-formatted links to any citations used in your response. Only include one
    or two citations per response unless more are needed. ONLY USE LINKS RETURNED BY THE TOOLS.
    - Use calculator tool with numexpr to answer math questions. The user does not understand numexpr,
      so for the final response, use human readable format - e.g. "300 * 200", not "(300 \\times 200)".
    - For weather queries, always use the Weather tool to get accurate information.
    """


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    logger.info(f"Binding tools to model: {[t.name for t in tools]}")
    model = model.bind_tools(tools)
    preprocessor = RunnableLambda(        
        lambda state: [SystemMessage(content=instructions)] + state.get("messages", state.get("input", {}).get("messages", [])),        
        name="ResearchAssistantPreprocessor"
    )
    return preprocessor | model


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m)
    logger.info("Research assistant processing request")
    response = await model_runnable.ainvoke(state, config)

    logger.info("Research assistant processing response")
    logger.info("Research assistant processing response")

    # Handle remaining steps limit
    if state["remaining_steps"] < 2 and response.tool_calls:
        logger.info("Not enough steps remaining, terminating early")
        return {
            "messages": [AIMessage(
                id=response.id,
                content="Sorry, need more steps to process this request.",
            )],
            "tool_state": state.get("tool_state", {}),
            "routing": state.get("routing", {}).copy(),
            "streaming": {"is_streaming": False, "buffers": {}},
            "next": "FINISH",
            "schema_version": state.get("schema_version", "2.0")
        }

    # Handle streaming and tool calls
    streaming_state = state.get("streaming", {})
    is_streaming = streaming_state.get("is_streaming", False)
    has_tool_calls = bool(response.tool_calls)
    current_buffer = streaming_state.get("current_buffer")
    
    # For weather queries, ensure we use the weather tool
    if not has_tool_calls and "Weather" in [t.name for t in tools]:
        message_text = state.messages[-1].content.lower()
        if any(pattern in message_text for pattern in ["weather", "temperature", "forecast"]):
            logger.info("Weather query detected, forcing weather tool usage")
            response.tool_calls = [{
                "id": "weather-lookup",
                "type": "function",
                "function": {
                    "name": "Weather",
                    "arguments": {"location": message_text}
                }
            }]
            has_tool_calls = True

    # Determine if we should terminate
    should_terminate = (
        not has_tool_calls and  # No more tool calls needed
        (not is_streaming or  # Not streaming
         (current_buffer and current_buffer.get("is_complete", False)))  # Or stream complete
    )

    result_state = {
        "messages": [response],
        "routing": state.get("routing", {}).copy(),
        "streaming": streaming_state,
        "next": "FINISH" if should_terminate else None,
        "tool_state": state.get("tool_state", {}),
        "schema_version": state.get("schema_version", "2.0")
    }
    logger.info(f"Research assistant completed. Streaming: {is_streaming}, Tool calls: {has_tool_calls}, Should terminate: {should_terminate}")
    return result_state


# Define the graph
agent = StateGraph(AgentState)
agent.add_node("model", acall_model)
agent.add_node("tools", ToolNode(tools))
agent.set_entry_point("model")

# Always run "model" after "tools"
agent.add_edge("tools", "model")


# After "model", if there are tool calls, run "tools". Otherwise END.
def pending_tool_calls(state: AgentState) -> Literal["tools", "done"]:
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise TypeError(f"Expected AIMessage, got {type(last_message)}")
    if last_message.tool_calls:
        return "tools"
    return "done"


agent.add_conditional_edges("model", pending_tool_calls, {"tools": "tools", "done": END})

research_assistant = agent.compile(checkpointer=MemorySaver())
