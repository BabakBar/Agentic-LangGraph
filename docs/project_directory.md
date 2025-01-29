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
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── manager.py          # Agent registry and management
│   │   │   ├── agent_registry.py   # Base agent registry implementation
│   │   │   ├── registry_base.py    # Generic registry functionality
│   │   │   └── orchestrator/       # Agent orchestration system
│   │   │       ├── __init__.py
│   │   │       ├── graph.py        # Graph construction for orchestrator
│   │   │       └── router.py       # Routing logic between agents
│   │   ├── agents/
│   │   │   ├── base/
│   │   │   │   └── chatbot.py      # Base chatbot implementation
│   │   │   └── specialized/
│   │   │       └── research_assistant.py  # Research assistant with tools
│   │   └── tools/                  # LangChain-compatible tool implementations
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
│   │   ├── schema.py         # Core schema definitions
│   │   └── models.py         # Model-specific schemas
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

5- Recent Architectural Changes

Tool System Refactor:
- Migrated from custom ToolProtocol to LangChain's @tool decorator
- Tools now directly compatible with LangGraph's ToolNode
- Example: calculator.py refactored to use @tool for better integration

Import Structure:
- Absolute imports used for top-level packages (e.g., 'schema', 'core')
- Relative imports for internal modules within packages
- Fixed circular dependencies between tools and agents
- Example: manager.py now uses 'from schema.schema import AgentInfo'

Agent Registry:
- Enhanced AgentRegistry with get_base_agents() method
- Returns runnable graphs for orchestrator compatibility
- Fixed interaction between registry and orchestrator

Docker Considerations:
- src/ directory copied to /app/ in container
- Import paths work in both local and Docker environments
- No relative imports beyond top-level package

Development Tips:
- Use 'python src/run_service.py' for faster local testing
- Docker build needed only for final verification
- Watch mode enabled for automatic reloading
