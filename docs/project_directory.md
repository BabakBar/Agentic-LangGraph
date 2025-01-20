Agentic Orixa/
├── .github/
│   └── workflows/
│       └── test.yml              # GitHub Actions workflow for testing
│
├── docker/
│   ├── Dockerfile.app           # Dockerfile for Streamlit app
│   └── Dockerfile.service       # Dockerfile for FastAPI service
│
├── media/
│   ├── app_screenshot.png       # Screenshot for README
│   └── agent_architecture.png   # Architecture diagram
│
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── agents.py           # Agent registry and configuration
│   │   ├── research_assistant.py
│   │   └── chatbot.py
│   │
│   ├── client/
│   │   ├── __init__.py
│   │   └── client.py           # AgentClient implementation
│   │
│   ├── core/
│   │   ├── __init__.py         # Exports settings and get_model
│   │   ├── llm.py             # LLM configuration
│   │   └── settings.py        # Global settings using Pydantic
│   │
│   ├── schema/
│   │   ├── __init__.py
│   │   └── messages.py        # Protocol schema definitions
│   │
│   ├── service/
│   │   ├── __init__.py        # Exports FastAPI app
│   │   └── service.py         # FastAPI service implementation
│   │
│   ├── run_agent.py          # Script to run agent directly
│   ├── run_client.py         # Example client usage
│   ├── run_service.py        # Script to start FastAPI service
│   └── streamlit_app.py      # Streamlit web interface
│
├── tests/
│   ├── __init__.py
│   ├── test_agent.py
│   ├── test_client.py
│   └── test_service.py
│
├── .dockerignore             # Docker ignore patterns
├── .env.example             # Example environment variables
├── .gitignore              # Git ignore patterns
├── .pre-commit-config.yaml  # Pre-commit hooks configuration
├── codecov.yml             # Codecov configuration
├── compose.yaml            # Docker Compose configuration
├── langgraph.json         # LangGraph Studio configuration
├── LICENSE                # Project license
├── pyproject.toml         # Python project metadata and dependencies
├── README.md             # Project documentation
└── uv.lock               # UV dependency lock file


1- Docker Configuration
docker/ contains separate Dockerfiles for the service and app
compose.yaml defines the multi-container setup with watch mode

2- Source Code (src/)
agents/: Contains agent implementations
client/: API client implementation
core/: Core functionality and settings
schema/: Data structure definitions
service/: FastAPI service implementation

3- Entry Points
run_service.py: Starts the FastAPI server
streamlit_app.py: Launches the web interface

4- Configuration Files
pyproject.toml: Project metadata and dependencies
langgraph.json: LangGraph Studio configuration
.pre-commit-config.yaml: Code quality checks
codecov.yml: Coverage reporting configuration
