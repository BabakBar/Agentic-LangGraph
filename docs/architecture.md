# System Architecture

## Current Implementation

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#BBE8F2', 'edgeLabelBackground':'#FFF', 'tertiaryColor': '#FFF'}}}%%

flowchart TD
    subgraph External[External Systems]
        direction TB
        LLM_API[LLM API]
        DB[(SQLite DB)]
        Auth[Auth Service]
    end

    subgraph Frontend[Frontend Layer]
        direction TB
        WebUI[Streamlit Web Interface]
        CLI[Command Line Interface]
    end

    subgraph Backend[Backend Layer]
        direction TB
        subgraph Services[Core Services]
            direction TB
            AgentService[FastAPI Agent Service]
            AuthService[Auth Service]
        end

        subgraph Agents[Agent Layer]
            direction TB
            ChatAgent[Chat Agent]
            ResearchAgent[Research Agent]
            TaskAgent[Task Agent]
        end

        subgraph Core[Core Components]
            direction TB
            LLM[LLM Interface]
            Settings[Settings Manager]
            Schema[Schema Definitions]
            Tools[Tool Manager]
        end
    end

    %% Data Flows
    WebUI -->|HTTP| AgentService
    CLI -->|HTTP| AgentService

    AgentService -->|Auth| AuthService
    
    ChatAgent -->|Chat| LLM_API
    ResearchAgent -->|Research| LLM_API
    TaskAgent -->|Tasks| LLM_API
    
    Tools -->|Integration| External

    LLM -->|Completion| LLM_API
    Settings -->|Config| DB
    Schema -->|Models| DB
    AuthService -->|OAuth| Auth

    Services -->|Logs & State| DB
    Agents -->|History & Results| DB

    %% Style
    classDef external fill:#F5F5F5,stroke:#333,stroke-width:2px
    classDef frontend fill:#E3F2FD,stroke:#333,stroke-width:2px
    classDef backend fill:#FFF3E0,stroke:#333,stroke-width:2px
    classDef services fill:#FFE0B2,stroke:#333,stroke-width:2px
    classDef agents fill:#FFCC80,stroke:#333,stroke-width:2px
    classDef core fill:#FFE57F,stroke:#333,stroke-width:2px

    class External external
    class Frontend frontend
    class Backend backend
    class Services services
    class Agents agents
    class Core core
```

## Current Components

### External Systems

- **LLM API**: Multiple LLM providers (OpenAI, Anthropic, Google, etc.)
- **SQLite DB**: Local state persistence and checkpointing
- **Auth Service**: Basic bearer token authentication

### Frontend Layer

- **Web Interface (Streamlit)**: Primary user interface
- **Command Line Interface**: Development and testing interface

### Backend Layer

#### Core Services

- **Agent Service (FastAPI)**: Main API gateway and request router
- **Auth Service**: Basic authentication and authorization

#### Agent Layer

- **Chat Agent**: Basic conversational agent
- **Research Agent**: Web search and calculator capabilities
- **Task Agent**: Background task execution

#### Core Components

- **LLM Interface**: Multi-provider LLM abstraction
- **Settings Manager**: Environment-based configuration
- **Schema Definitions**: Pydantic data models
- **Tool Manager**: Basic tool integration system

## Suggested Enhancements

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#E6F3FF', 'edgeLabelBackground':'#FFF', 'tertiaryColor': '#FFF'}}}%%

flowchart TD
    subgraph Proposed[Proposed Additions]
        direction TB
        VectorDB[(Vector DB)]
        Queue[Message Queue]
        Monitor[Monitoring System]
        MemoryService[Memory Service]
        EventBus[Event Bus]
    end

    subgraph Benefits[Key Benefits]
        direction TB
        Semantic[Semantic Search]
        AsyncOps[Async Operations]
        Metrics[System Metrics]
        Context[Context Management]
        Decoupling[System Decoupling]
    end

    VectorDB --> Semantic
    Queue --> AsyncOps
    Monitor --> Metrics
    MemoryService --> Context
    EventBus --> Decoupling

    %% Style
    classDef proposed fill:#E6F3FF,stroke:#333,stroke-width:2px
    classDef benefits fill:#F8F8F8,stroke:#333,stroke-width:2px

    class Proposed proposed
    class Benefits benefits
```

### Proposed Enhancements

1. **Vector Database**
   - Semantic search capabilities
   - Conversation memory embeddings
   - Document storage and retrieval
   - Suggested implementations: Chroma, Weaviate, or Pinecone

2. **Message Queue**
   - Asynchronous task processing
   - Better scalability
   - Reliable message delivery
   - Suggested implementations: Redis, RabbitMQ, or Apache Kafka

3. **Monitoring System**
   - System metrics collection
   - Performance monitoring
   - Error tracking
   - Suggested implementations: Prometheus + Grafana

4. **Memory Service**
   - Advanced context management
   - Long-term conversation storage
   - Memory retrieval strategies
   - Integration with Vector DB

5. **Event Bus**
   - Decoupled component communication
   - Event sourcing
   - System extensibility
   - Suggested implementations: Redis Pub/Sub, Apache Kafka

## Implementation Priorities

1. **Phase 1: Memory Enhancement**
   - Implement Vector DB
   - Create Memory Service
   - Enhance context management

2. **Phase 2: Scalability**
   - Add Message Queue
   - Implement Event Bus
   - Enhance async operations

3. **Phase 3: Observability**
   - Add Monitoring System
   - Implement metrics collection
   - Set up dashboards

## Current Data Flows

1. User interactions flow through frontend interfaces to Agent Service
2. Agent Service handles authentication and routing
3. Agents utilize LLM API for processing
4. Tools provide external integrations
5. SQLite DB stores state and history
6. Authentication flows through Auth Service

## Future Data Flows

1. Vector DB will handle semantic search and embeddings
2. Message Queue will manage async operations
3. Event Bus will handle internal communication
4. Monitoring will collect system-wide metrics
5. Memory Service will manage advanced context