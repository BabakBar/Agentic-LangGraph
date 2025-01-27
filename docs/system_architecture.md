# Agentic Orixa System Architecture

## Overview
Agentic Orixa is a multi-agent system built on FastAPI and LangGraph, designed to provide flexible, extensible agent capabilities through a unified interface. The system supports multiple LLM providers and agent types, with built-in support for conversation history, feedback collection, and real-time streaming.

## Core Components

### 1. Service Layer (FastAPI)
- **Endpoints**:
  - `/info`: Service metadata and capabilities
  - `/invoke`: Single-turn agent interactions
  - `/stream`: Real-time streaming with SSE
  - `/feedback`: LangSmith feedback integration
  - `/history`: Conversation history management
- **Features**:
  - Bearer token authentication
  - SQLite-based state persistence
  - LangGraph integration for agent orchestration

### 2. Agent Layer
- **Agent Types**:
  - Chatbot: Basic conversational agent
  - Research Assistant: Web search and calculator capabilities
  - Background Task Agent: Asynchronous task processing
- **Architecture**:
  - Central registry for agent management
  - Type-safe configuration using dataclasses
  - LangGraph's CompiledStateGraph for implementation

### 3. Configuration Management
- **Settings**:
  - Environment variables and .env file support
  - Multiple LLM provider configurations
  - Development mode detection
- **Model Management**:
  - Cached model instantiation
  - Type-safe model selection
  - Streaming support

### 4. Data Structures
- **Core Models**:
  - AgentInfo: Metadata about available agents
  - ServiceMetadata: Service capabilities and configuration
  - UserInput: Base input structure for agent interactions
- **Communication Patterns**:
  - ChatMessage: Standardized message format
  - ToolCall: Tool invocation specification
  - Feedback: LangSmith integration structure

## Deployment Architecture
- **Docker**:
  - Separate containers for service and app
  - Docker Compose for orchestration
- **CI/CD**:
  - GitHub Actions for testing
  - Codecov for coverage reporting

## Development Environment
- **Tools**:
  - Pre-commit hooks for code quality
  - LangGraph Studio integration
  - UV dependency management
- **Testing**:
  - Unit tests for core functionality
  - Integration tests for service endpoints
  - End-to-end tests for Docker deployment

## Key Features
- Multi-agent support with extensible architecture
- Real-time streaming with Server-Sent Events
- Conversation history persistence
- LangSmith integration for feedback collection
- Multiple LLM provider support
- Type-safe configuration and data structures


### Data Flow

The data flow when a user asks a question in the UI is as follows:

1.  **User Input:** The user enters a question in the Streamlit app (`src/streamlit_app.py`).
2.  **Message Creation:** The app creates a `ChatMessage` object with `type="human"` and the user's input.
3.  **Agent Client Request:** The app's `AgentClient` (`src/client/client.py`) sends the message to the backend service using either `astream` (for streaming responses) or `ainvoke` (for a single response). The request is sent to the `/stream` or `/invoke` endpoint of the service, along with the selected agent and model.
4.  **Service Processing:**
    *   The backend service (`src/service/service.py`) receives the request.
    *   It authenticates the request using a bearer token.
    *   It retrieves the specified agent using `get_agent` from `src/agents/agents.py`.
    *   For `/invoke`, the service calls the agent's `ainvoke` method and returns the final response as a `ChatMessage`.
    *   For `/stream`, the service calls the agent's `astream_events` method and streams the response back to the client. The `message_generator` function parses the events and yields `ChatMessage` objects or string tokens.
5.  **Agent Execution:** The selected agent (either a base agent or the orchestrator) processes the user's input.
    *   **Base Agent (e.g., `research-assistant`):** The agent's logic is defined in its corresponding file (`src/agents/research_assistant.py`). The agent may use tools, call other agents, or generate a response directly. The execution flow is: `Agent Input --> Model --> Tool Call Check --> (Tool Execution --> Model) * --> Final Response`
    *   **Orchestrator:** The orchestrator's logic is defined in `src/agents/orchestrator/graph.py`. It routes the task to the appropriate base agent. The execution flow is: `Agent Input --> Router --> Agent Executor --> Base Agent --> Router --> Final Response`
6.  **Response Handling:**
    *   For `/invoke`, the service sends the final `ChatMessage` back to the client.
    *   For `/stream`, the service sends a stream of data back to the client, which includes `ChatMessage` objects and string tokens.
7.  **Client Processing:** The `AgentClient` parses the response and returns it to the Streamlit app.
8.  **UI Update:** The Streamlit app's `draw_messages` function receives the response and updates the UI. It displays the agent's messages, tool call inputs/outputs, and any custom data.
9.  **Feedback:** The user can provide feedback using the feedback widget, which sends a request to the `/feedback` endpoint of the service.

### Components

The main components of the system are:

*   **Streamlit App (`src/streamlit_app.py`):** The user interface for interacting with the system.
*   **Agent Client (`src/client/client.py`):** A client for communicating with the backend service.
*   **Backend Service (`src/service/service.py`):** A FastAPI application that handles requests and interacts with the agents.
*   **Agent Registry (`src/agents/agents.py`):** Manages the available agents.
*   **Base Agents:**
    *   `research-assistant` (`src/agents/research_assistant.py`): A research assistant with web search and calculator capabilities.
    *   `bg-task-agent` (`src/agents/bg_task_agent/bg_task_agent.py`): A background task agent.
    *   `chatbot` (`src/agents/chatbot.py`): A simple chatbot.
*   **Orchestrator (`src/agents/orchestrator/graph.py`):** An agent that routes tasks to other agents.
*   **Router (`src/agents/orchestrator/router.py`):** The logic for routing tasks to different agents.