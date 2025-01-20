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
