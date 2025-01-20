# Source Code Structure and Implementation

## Overview

The `src/` directory contains the core implementation of the AI agent service toolkit. It's organized into several key components that work together to provide the agent functionality.

## Directory Structure

```
src/
├── agents/            # Agent implementations and utilities
├── client/            # Client library for interacting with the service
├── core/              # Core functionality and settings
├── schema/            # Data structure definitions
├── service/           # FastAPI service implementation
├── run_agent.py       # Direct agent execution
├── run_client.py      # Client usage examples
├── run_service.py     # Service startup script
└── streamlit_app.py   # Streamlit web interface
```

## Core Components

### 1. Agents (src/agents/)
Contains the core agent implementations and supporting utilities.

#### Key Files:
- `agents.py`: Agent registry and configuration
- `research_assistant.py`: Example research assistant agent
- `chatbot.py`: Example chatbot agent
- `tools.py`: Shared tool implementations
- `utils.py`: Utility functions for agents
- `bg_task_agent/`: Background task agent implementation

#### Agent Architecture:
1. **Agent Registry**
   - Centralized agent management
   - Configuration through environment variables
   - Dynamic agent loading

2. **Tool Integration**
   - Standardized tool interface
   - Built-in tools (web search, calculator, etc.)
   - Custom tool registration

3. **State Management**
   - Conversation history tracking
   - Thread-based isolation
   - Persistent state storage

### 2. Client (src/client/)
Provides a client library for interacting with the agent service.

#### Key Features:
- Synchronous and asynchronous interfaces
- Streaming support
- Type-safe API
- Error handling

#### Implementation Patterns:
- Request/response handling
- Streaming protocol implementation
- Authentication integration
- Configuration management

### 3. Core (src/core/)
Contains shared functionality and configuration.

#### Key Components:
- `settings.py`: Application settings using Pydantic
- `llm.py`: LLM configuration and management
- `__init__.py`: Core exports and initialization

### 4. Schema (src/schema/)
Defines the data structures used throughout the system.

#### Key Models:
- Chat messages
- Tool calls
- API requests/responses
- Error structures
- Configuration models

### 5. Service (src/service/)
Implements the FastAPI service layer.

#### Key Features:
- REST API endpoints
- Real-time streaming
- Authentication
- Rate limiting
- LangSmith integration

#### Implementation Details:
- Endpoint design
- Middleware configuration
- Error handling
- Performance optimization

## Workflow Patterns

### 1. Agent Execution Flow
1. Client sends request to service
2. Service routes request to appropriate agent
3. Agent processes request using tools
4. Results are streamed back to client

### 2. Tool Execution
1. Agent identifies required tools
2. Tools are executed with appropriate parameters
3. Results are processed and incorporated into response
4. Tool execution status is tracked

### 3. Streaming Protocol
1. Client initiates streaming request
2. Service establishes streaming connection
3. Agent generates response tokens
4. Tokens are streamed to client in real-time

## Extension Patterns

### Adding New Agents
1. Create new agent class in `src/agents/`
2. Implement required methods
3. Register agent in `agents.py`
4. Add configuration options
5. Update documentation

### Adding New Tools
1. Create tool implementation
2. Add to `tools.py`
3. Register with appropriate agents
4. Update schema definitions
5. Add tests

### Modifying Service Behavior
1. Update endpoint definitions
2. Add middleware as needed
3. Modify request/response handling
4. Update client implementation

## Development Practices

### Testing Strategy
1. Unit tests for individual components
2. Integration tests for service endpoints
3. End-to-end tests for complete workflows
4. Performance testing for critical paths

### Debugging Patterns
1. LangSmith tracing
2. Structured logging
3. Error handling middleware
4. Debug endpoints

### Performance Optimization
1. Async/await patterns
2. Connection pooling
3. Caching strategies
4. Load testing

## Example Workflows

### 1. Adding a New Feature
1. Define requirements
2. Create schema definitions
3. Implement core functionality
4. Add service endpoints
5. Update client library
6. Add tests
7. Update documentation

### 2. Debugging an Issue
1. Reproduce issue
2. Enable LangSmith tracing
3. Check logs
4. Isolate problematic component
5. Implement fix
6. Add regression test

### 3. Performance Optimization
1. Identify bottleneck
2. Profile code
3. Implement optimization
4. Verify improvement
5. Add monitoring
