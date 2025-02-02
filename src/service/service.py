"""Service implementation for the agent API."""
import json
import asyncio
import logging
import os
import warnings
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated, Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langchain_core._api import LangChainBetaWarning
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.state import CompiledStateGraph
from langsmith import Client as LangsmithClient

from agents import DEFAULT_AGENT, get_agent, get_all_agent_info
from core import settings
from core.logging_config import setup_logging
from schema import (
    ChatHistory,
    ChatHistoryInput,
    ChatMessage,
    Feedback,
    FeedbackResponse,
    ServiceMetadata,
    StreamInput,
    UserInput,
)
from service.utils import (
    convert_message_content_to_string,
    langchain_to_chat_message,
    remove_tool_calls,
)
from agents.core.orchestrator.state import create_initial_state, OrchestratorState, RoutingMetadata, StreamingState, CURRENT_VERSION

warnings.filterwarnings("ignore", category=LangChainBetaWarning)
logger = logging.getLogger(__name__)

# Set up logging with environment LOG_LEVEL
setup_logging(os.getenv("LOG_LEVEL", "INFO"))

# Ensure data directory exists
os.makedirs("/app/data", exist_ok=True)
DB_PATH = "/app/data/checkpoints.db"

def verify_bearer(
    http_auth: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(HTTPBearer(description="Please provide AUTH_SECRET api key.", auto_error=False)),
    ],
) -> None:
    if not settings.AUTH_SECRET:
        return
    auth_secret = settings.AUTH_SECRET.get_secret_value()
    if not http_auth or http_auth.credentials != auth_secret:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Construct agent with Sqlite checkpointer
    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as saver:
        agents = get_all_agent_info()
        for a in agents:
            agent = get_agent(a.key)
            agent.checkpointer = saver
        yield

app = FastAPI(lifespan=lifespan)
router = APIRouter(dependencies=[Depends(verify_bearer)])

@router.get("/info")
async def info() -> ServiceMetadata:
    models = list(settings.AVAILABLE_MODELS)
    models.sort()
    return ServiceMetadata(
        agents=get_all_agent_info(),
        models=models,
        default_agent=DEFAULT_AGENT,
        default_model=settings.DEFAULT_MODEL,
    )

@router.get("/debug/orchestrator")
async def debug_orchestrator():
    """Debug endpoint to check orchestrator state."""
    agent = get_agent("orchestrator")
    return {"status": "ok", "agent_type": str(type(agent))}

def _parse_input(user_input: UserInput) -> tuple[dict[str, Any], UUID]:
    run_id = uuid4()
    thread_id = user_input.thread_id or str(uuid4())
    
    # Get list of available agent IDs from registered agents
    agent_ids = [agent.key for agent in get_all_agent_info()]
    
    logger.debug("Creating initial state in _parse_input")
    # Create initial state with agent_ids using model_copy since OrchestratorState is frozen
    state = create_initial_state([HumanMessage(content=user_input.message)]).model_copy(update={"agent_ids": agent_ids})
    # Convert to dictionary before passing to LangGraph
    initial_state = state.model_dump()
    logger.debug(f"Initial state type: {type(initial_state)}")
    logger.debug(f"Initial state dict: {initial_state}")
    
    kwargs = {
        "input": initial_state,  # Pass dictionary instead of model instance
        "config": RunnableConfig(
            configurable={"thread_id": thread_id, "model": user_input.model},
            run_id=run_id,
            version=CURRENT_VERSION
        ),
    }
    logger.debug(f"Kwargs for LangGraph: {kwargs}")
    return kwargs, run_id

@router.post("/{agent_id}/invoke")
@router.post("/invoke")
async def invoke(user_input: UserInput, agent_id: str = DEFAULT_AGENT) -> ChatMessage:
    """Invoke an agent with user input to retrieve a final response."""
    agent: CompiledStateGraph = get_agent(agent_id)
    kwargs, run_id = _parse_input(user_input)
    try:
        response = await agent.ainvoke(**kwargs)
        output = langchain_to_chat_message(response["messages"][-1])
        output.run_id = str(run_id)
        return output
    except Exception as e:
        logger.error(f"An exception occurred: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error")

async def message_generator(
    user_input: StreamInput, agent_id: str = DEFAULT_AGENT
) -> AsyncGenerator[str, None]:
    """Generate a stream of messages from the agent."""
    try:
        agent = get_agent(agent_id)
        if not agent:
            yield f"data: {json.dumps({'type': 'error', 'content': f'Agent {agent_id} not found'})}\n\n"
            return
            
        kwargs, run_id = _parse_input(user_input)
        
        # Process streamed events from the graph and yield messages over the SSE stream.
        logger.debug(f"Starting stream for agent {agent_id} with kwargs: {kwargs}")
        
        async for event in agent.astream_events(**kwargs, version="v2"):
            if event and event["event"] == "on_chain_end":
                if "data" in event and "output" in event["data"]:
                    output = event["data"]["output"]
                    logger.debug(f"Chain end output type: {type(output)}")
                    if hasattr(output, "model_dump"):
                        logger.debug(f"Chain end output dump: {output.model_dump()}")
            logger.debug(f"Received event: {event}")
            if not event:
                continue

            new_messages = []
            # Yield messages written to the graph state after node execution finishes.
            if (
                event["event"] == "on_chain_end"
                and any(t.startswith("graph:step:") for t in event.get("tags", []))
            ):
                logger.debug("Processing chain end event with messages")
                output = event["data"]["output"]
                
                # Handle Command objects
                if hasattr(output, "update") and isinstance(output.update, dict):
                    if "messages" in output.update:
                        new_messages = output.update.get("messages", [])
                # Handle direct message updates
                elif isinstance(output, dict) and "messages" in output:
                    new_messages = output["messages"]
                else:
                    logger.debug(f"Skipping event with unhandled output type: {type(output)}")

            # Also yield intermediate messages from agents.utils.CustomData.adispatch().
            if event["event"] == "on_custom_event" and "custom_data_dispatch" in event.get("tags", []):
                new_messages = [event["data"]]

            for message in new_messages:
                try:
                    chat_message = langchain_to_chat_message(message)
                    chat_message.run_id = str(run_id)
                except Exception as e:
                    logger.error(f"Error parsing message: {e}")
                    yield f"data: {json.dumps({'type': 'error', 'content': 'Unexpected error'})}\n\n"
                    continue
                # LangGraph re-sends the input message, which feels weird, so drop it
                if chat_message.type == "human" and chat_message.content == user_input.message:
                    continue
                yield f"data: {json.dumps({'type': 'message', 'content': chat_message.model_dump()})}\n\n"

            # Yield tokens streamed from LLMs.
            if event["event"] == "on_chat_model_stream" and user_input.stream_tokens:
                logger.debug("Processing chat model stream event")
                content = remove_tool_calls(event["data"]["chunk"].content)
                if content:
                    yield f"data: {json.dumps({'type': 'token', 'content': convert_message_content_to_string(content)})}\n\n"
                continue

    except asyncio.CancelledError:
        logger.info("Stream cancelled by client")
        try:
            yield f"data: {json.dumps({'type': 'status', 'content': 'Stream cancelled'})}\n\n"
        except Exception:
            pass
        finally:
            return
            
    except Exception as e:
        logger.error(f"Stream generation error: {e}")
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

@router.post(
    "/{agent_id}/stream", response_class=StreamingResponse
)
@router.post("/stream", response_class=StreamingResponse)
async def stream(user_input: StreamInput, agent_id: str = DEFAULT_AGENT) -> StreamingResponse:
    """Stream an agent's response to a user input."""
    try:
        return StreamingResponse(
            message_generator(user_input, agent_id),
            media_type="text/event-stream",
        )
    except Exception as e:
        logger.error(f"Stream error: {e}")
        async def error_generator():
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
        return StreamingResponse(
            error_generator(),
            media_type="text/event-stream"
        )

@router.post("/feedback")
async def feedback(feedback: Feedback) -> FeedbackResponse:
    """Record feedback for a run to LangSmith."""
    client = LangsmithClient()
    kwargs = feedback.kwargs or {}
    client.create_feedback(
        run_id=feedback.run_id,
        key=feedback.key,
        score=feedback.score,
        **kwargs,
    )
    return FeedbackResponse()

@router.post("/history")
def history(input: ChatHistoryInput) -> ChatHistory:
    """Get chat history."""
    agent: CompiledStateGraph = get_agent(DEFAULT_AGENT)
    try:
        state_snapshot = agent.get_state(
            config=RunnableConfig(
                configurable={
                    "thread_id": input.thread_id,
                }
            )
        )
        messages: list[AnyMessage] = state_snapshot.values["messages"]
        chat_messages: list[ChatMessage] = [langchain_to_chat_message(m) for m in messages]
        return ChatHistory(messages=chat_messages)
    except Exception as e:
        logger.error(f"An exception occurred: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

app.include_router(router)

class AgentService:
    async def stream_response(self, message):
        try:
            # Initialize state if needed
            state = None
            if isinstance(message, str):
                message = {"content": message}
            
            # Create initial state
            state = create_initial_state([HumanMessage(content=message["content"])]).model_dump()
            logger.debug(f"AgentService stream_response initial state type: {type(state)}")
            logger.debug(f"AgentService stream_response initial state: {state}")
            
            try:
                response = await self.orchestrator.process_message(state, message)
                
                if isinstance(response, dict):
                    yield response
                elif isinstance(response, OrchestratorState):
                    # Extract the last message from state
                    if response.messages:
                        last_message = response.messages[-1]
                        yield {"type": "message", "content": str(last_message.content)}
                else:
                    yield {"type": "message", "content": str(response)}
                    
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                yield {"type": "error", "content": f"Error processing message: {str(e)}"}
                
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield {"type": "error", "content": str(e)}

    async def invoke(self, message):
        try:
            if isinstance(message, str):
                message = {"content": message}
            
            state = create_initial_state([HumanMessage(content=message["content"])]).model_dump()
            logger.debug(f"AgentService invoke initial state type: {type(state)}")
            logger.debug(f"AgentService invoke initial state: {state}")
            response = await self.orchestrator.process_message(state, message)
            return {"type": "message", "content": str(response.messages[-1].content) if response.messages else "No response"}
            
        except Exception as e:
            logger.error(f"Invoke error: {e}")
            return {"type": "error", "content": f"Error processing message: {str(e)}"}
