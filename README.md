# Agentic Orixa

An AI agent service built with LangGraph, FastAPI and Streamlit.

## System Architecture

For detailed architecture documentation, see [System Architecture](docs/system_architecture.md)

### Core Components

1. **Service Layer** (FastAPI)
   - REST API endpoints
   - Real-time streaming
   - Authentication and rate limiting
   - LangSmith integration

2. **Agent Layer**
   - Multiple agent types
   - Tool integration
   - State management
   - Conversation history

3. **Configuration Management**
   - Environment variables
   - Multiple LLM providers
   - Type-safe settings

4. **Data Structures**
   - Protocol schema definitions
   - Type-safe models
   - Validation patterns

## Implementation Details

### Agent Implementation

For detailed agent documentation, see [Agent Implementation](docs/agent_implementation.md)

- Centralized agent registry
- Tool integration patterns
- State management
- Testing and debugging

### Service Implementation

For detailed service documentation, see [Service Implementation](docs/service_implementation.md)

- API endpoint design
- Authentication patterns
- Deployment strategies
- Performance optimization

## Quickstart

You can run the application either locally or using Docker. Both methods provide identical functionality and user experience.

### Prerequisites

- Python 3.11 or 3.12 or 3.13
- pip or uv package manager
- At least one LLM API key (OpenAI, Anthropic, Google, etc.)

### Local Setup (Recommended for Development)

1. **Install Dependencies**:

   Windows (PowerShell):
   ```powershell
   # Install uv if needed
   pip install --user uv

   # Create venv and install all dependencies
   uv venv -p 3.13
   .\.venv\Scripts\Activate.ps1
   uv pip install --upgrade pip
   uv sync --all-extras --frozen
   ```

   macOS/Linux (bash):
   ```bash
   # Install uv if needed
   pip install --user uv

   # Create venv and install all dependencies
   uv venv -p 3.13
   source .venv/bin/activate
   uv pip install --upgrade pip
   uv sync --all-extras --frozen
   ```

3. **Configure Environment**:
   ```bash
   # Copy environment file
   cp .env.example .env
   
   # Edit .env with your settings:
   # MODE=dev
   # DEFAULT_MODEL=gpt-4o-mini
   # HOST=0.0.0.0
   # PORT=8000
   # Add your LLM API key(s)
   ```

4. **Run the Application**:

   Terminal 1 (Backend):
   ```bash
   python src/run_service.py
   # Service will run on http://localhost:8000
   ```

   Terminal 2 (Frontend):
   ```bash
   export AGENT_URL="http://localhost:8000"  # For macOS/Linux
   # or
   $env:AGENT_URL="http://localhost:8000"    # For Windows PowerShell
   
   streamlit run src/streamlit_app.py
   # UI will be available at http://localhost:8501
   ```

## Troubleshooting

### Port Conflicts
Stop existing services:
```bash
# Linux/macOS
kill $(lsof -t -i:8000) $(lsof -t -i:8501)

# Windows PowerShell
Stop-Process -Id (Get-NetTCPConnection -LocalPort 8000,8501).OwningProcess -Force
```

- Backend API: <http://localhost:8080/redoc>
- Streamlit UI: <http://localhost:8501>

### Feature Parity

Both local and Docker setups provide:

- Identical API endpoints and functionality
- Same Streamlit UI experience
- Real-time streaming capabilities
- LangSmith integration (if configured)
- Hot reload for development

The main differences are:

- Local setup uses ports 8000/8501
- Docker setup uses ports 8080/8501
- Local setup provides easier debugging
- Docker setup ensures consistent environment

## Architecture Diagram

<img src="media/agent_architecture.png" width="600">

## Key Features

1. **LangGraph Agent**: Customizable agent framework
2. **FastAPI Service**: Robust API implementation
3. **Advanced Streaming**: Token and message streaming
4. **Streamlit Interface**: User-friendly chat UI
5. **Multiple Agent Support**: Flexible agent management
6. **Asynchronous Design**: Efficient request handling
7. **Feedback Mechanism**: LangSmith integration
8. **Dynamic Metadata**: Service configuration discovery
9. **Docker Support**: Easy development and deployment
10. **Testing**: Comprehensive test coverage

## Customization

To customize the agent:

1. Add new agents to `src/agents`
2. Register agents in `src/agents/agents.py`
3. Adjust Streamlit interface in `src/streamlit_app.py`

## Documentation

- [System Architecture](docs/system_architecture.md)
- [Agent Implementation](docs/agent_implementation.md)
- [Service Implementation](docs/service_implementation.md)
