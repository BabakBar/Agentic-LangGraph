# Orchestrator Implementation

The orchestrator is a specialized agent that routes tasks between other agents based on their capabilities. This implementation uses Pydantic for type safety and validation, with a clean separation between runtime and serializable components.

## Architecture

### Core Components

#### Type System (`types.py`)
```python
@runtime_checkable
class AgentLike(Protocol):
    @property
    def description(self) -> str: ...
    async def ainvoke(self, state: dict) -> dict: ...

class AgentMetadata(BaseModel):
    id: str
    description: str
    capabilities: list[str]
```

#### Registry Management (`registry.py`)
```python
class AgentRegistry(BaseModel):
    metadata: Dict[str, AgentMetadata]
    _instances: Dict[str, AgentLike] = PrivateAttr()
```
- Separates serializable metadata from runtime instances
- Uses Pydantic's PrivateAttr for non-serializable components
- Provides runtime validation of agent registration

#### State Management (`state.py`)
```python
class OrchestratorState(BaseModel):
    messages: list[BaseMessage]
    agent_ids: list[str]
    next_agent: str | None
```
- Pure Pydantic models for type safety
- Serializable state without runtime components
- Clear validation boundaries

### Routing System

#### Router Implementation (`router.py`)
```python
class RouterDecision(BaseModel):
    next: str

class ValidatedRouterOutput(RouterDecision):
    @model_validator(mode="after")
    def validate_next_agent(self, context: dict) -> "ValidatedRouterOutput":
        registry: AgentRegistry = context["registry"]
        if self.next != "FINISH" and not registry.has_agent(self.next):
            raise ValueError(f"Invalid agent: {self.next}")
```
- Two-phase routing with validation
- Context-based validation using Pydantic
- Early validation of LLM outputs

#### Graph Construction (`graph.py`)
```python
class OrchestratorConfig(BaseModel):
    checkpoint_dir: Path | None
    max_steps: int = 10
    model_name: str = "gpt-4"

def build_orchestrator(config: OrchestratorConfig, registry: AgentRegistry) -> StateGraph:
    """Pure function for graph construction"""
```
- Configuration through Pydantic models
- Pure function approach to graph building
- Explicit dependency injection

## Flow Control

1. **Initialization**
   - Register base agents with metadata
   - Create orchestrator with registry
   - Configure routing and validation

2. **Routing Process**
   ```python
   async def route_node(state: OrchestratorState, config: RunnableConfig):
       decision = await get_router_decision(state)
       validated = ValidatedRouterOutput(**decision, context={"registry": registry})
       return OrchestratorState(next_agent=validated.next)
   ```

3. **Agent Execution**
   ```python
   def agent_executor(state: OrchestratorState) -> str:
       if not state.next_agent or not registry.has_agent(state.next_agent):
           raise ValueError("Invalid agent")
       return state.next_agent
   ```

## Key Features

1. **Type Safety**
   - Runtime protocol checking
   - Pydantic validation
   - Clear error messages

2. **Dependency Management**
   - No circular imports
   - Clear dependency flow
   - Easy testing

3. **Serialization**
   - Safe state persistence
   - Metadata separation
   - Checkpoint support

4. **Validation**
   - Early validation
   - Context-aware checks
   - Runtime safety

## Usage Example

```python
from agents.registry import AgentRegistry
from agents.orchestrator import create_orchestrator

# Create registry
registry = AgentRegistry()

# Register agents
registry.register("research", research_agent)
registry.register("calculator", calculator_agent)

# Create orchestrator
orchestrator = create_orchestrator(registry)

# Use orchestrator
result = await orchestrator.ainvoke({
    "messages": [{"role": "user", "content": "Research quantum computing"}]
})
```

## Testing

The new architecture enables comprehensive testing:

1. **Unit Tests**
   - Test registry operations
   - Validate router decisions
   - Check state transitions

2. **Integration Tests**
   - Test agent registration
   - Verify routing flow
   - Check error handling

3. **Validation Tests**
   - Test invalid agent names
   - Check state validation
   - Verify serialization

## Error Handling

The system provides clear error messages for common issues:

1. **Registration Errors**
   ```python
   raise TypeError("Agent must implement AgentLike protocol")
   ```

2. **Routing Errors**
   ```python
   raise ValueError("Invalid agent name")
   ```

3. **State Errors**
   ```python
   raise ValueError("No next agent specified")
   ```

## Future Improvements

1. Add agent capability matching
2. Implement agent versioning
3. Add runtime metrics collection
4. Enhance checkpoint recovery
5. Add circuit breaker pattern