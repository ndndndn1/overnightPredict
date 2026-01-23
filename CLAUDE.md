# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OvernightPredict is a recursive agent system implementing the OODA loop (Observe-Orient-Decide-Act) with meta-cognition capabilities. It evaluates and adjusts prediction strategies in real-time across multiple LLM providers (OpenAI, DeepSeek, Claude/Anthropic).

## Commands

### Running the Application
```bash
python main.py                    # Interactive CLI
python main.py --dashboard        # TUI Dashboard
python main.py --create openai --prompt "..."   # Quick session
python main.py --parallel openai,claude --prompt "..."  # Parallel sessions
```

### Testing
```bash
pytest                                    # Run all tests
pytest --cov=src --cov-report=html       # Run with coverage
pytest tests/unit/test_entities.py -v    # Run specific test file
```

### Type Checking
```bash
mypy src/
```

### Setup
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp config/.env.example .env  # Then edit with API keys
```

## Architecture

The project follows Clean Architecture with four layers:

```
src/
├── domain/           # Business rules (entities, interfaces, value objects, events)
├── application/      # Use cases and services
├── infrastructure/   # External adapters (LLM providers, storage, context sharing)
└── presentation/     # UI (CLI, TUI dashboard, controllers)
```

### Core Services (src/application/services/)

| Service | Role |
|---------|------|
| **Orchestrator** | Central coordinator managing multiple parallel sessions |
| **SessionManager** | Runs OODA loop for individual sessions |
| **Forecaster** | Predicts future questions using context-aware strategy selection |
| **Executor** | Executes tasks and generates code via LLM |
| **Evaluator** | Measures semantic similarity between predictions and actuals |
| **MetaTuner** | Adjusts strategies based on performance metrics |

### OODA Loop Flow

The loop runs in `SessionManager.process_question()`:
1. **Observe**: Receive question, check pending predictions for matches
2. **Orient**: Evaluate prediction accuracy (semantic similarity), update context
3. **Decide**: Use predicted answer if accuracy >= threshold, or execute fresh
4. **Act**: Return result with metadata, generate new predictions, apply strategy adjustments

### Key Design Patterns

- **Strategy Pattern**: Dynamic prediction strategy replacement (IPredictionStrategy)
- **Factory Pattern**: LLMProviderFactory, ContextSharingFactory
- **Repository Pattern**: ISessionRepository with SQLite implementation
- **Adapter Pattern**: LLM provider adapters for OpenAI, DeepSeek, Claude
- **Event-Driven**: IEventBus with domain events

### Infrastructure Implementations

- **LLM Providers**: `src/infrastructure/llm_providers/` - OpenAI, DeepSeek, Claude with rate limiting
- **Storage**: SQLite (aiosqlite), in-memory event bus and context store
- **Context Sharing**: File-based, Cloud (S3), Redis-based options
- **Rate Limiting**: Token bucket and sliding window implementations

### Key Entities (src/domain/entities/)

- **Session**: Coding session with state and metrics
- **Prediction**: Predicted question/task with confidence and similarity scoring
- **Question**: User's actual question with source tracking
- **Task**: Work item with priority and status

## Configuration

Main configuration in `src/core/config/settings.py` using pydantic-settings. Key environment variables:

- `OPENAI_API_KEY`, `OPENAI_ENABLED` - OpenAI provider
- `DEEPSEEK_API_KEY`, `DEEPSEEK_ENABLED` - DeepSeek provider
- `CLAUDE_AUTH_TYPE` (api_key/oauth/session_key), `CLAUDE_API_KEY` - Claude provider
- `PREDICTION_ACCURACY_THRESHOLD` (default 0.7) - Similarity threshold for matching
- `CONTEXT_SHARING_TYPE` (file/cloud_bucket/redis) - Context sharing backend
