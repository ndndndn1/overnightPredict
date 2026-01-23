# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OvernightPredict is an AI-powered autonomous code generation system that predicts developer questions, pre-generates answers, validates predictions against actual questions, and automatically adjusts prediction strategies based on accuracy metrics. It supports parallel session orchestration for enterprise projects.

## Build & Development Commands

```bash
# Install dependencies
make install          # Production only
make dev              # With dev dependencies (preferred for development)

# Run tests
make test             # Run tests with coverage
pytest tests/test_core.py::test_session_state -v  # Run single test

# Code quality
make lint             # ruff + mypy
make format           # ruff/black formatting

# Run the application
make run              # Start overnight session
make serve            # Start API server (port 8000)
make serve-dev        # API server with auto-reload

# Docker
make docker-up        # Start with docker-compose (includes Redis)
make docker-down      # Stop
```

## Architecture

### Core Pipeline Flow
1. **SessionOrchestrator** (`src/sessions/orchestrator.py`) - Manages parallel sessions, auto-scaling, and health monitoring
2. **OvernightEngine** (`src/core/engine.py`) - Main orchestrator: prediction loop, answer generation, accuracy evaluation
3. **QuestionPredictor** (`src/predictors/question.py`) - Predicts questions using 4 strategies: context-based, pattern-matching, semantic-similarity, hybrid
4. **AnswerGenerator** (`src/generators/answer.py`) - Pre-generates answers for predicted questions
5. **AccuracyEvaluator** (`src/evaluators/accuracy.py`) - Compares predictions vs actual questions using embeddings
6. **StrategyManager** (`src/strategies/manager.py`) - Dynamically selects and adapts prediction strategies based on accuracy
7. **CodeGenerator** (`src/generators/code.py`) - Produces production code from Q&A pairs
8. **CheckpointManager** (`src/sessions/checkpoint.py`) - Session state persistence to SQLite

### Key Data Models (`src/core/models.py`)
- `SessionState` - Complete state of a coding session
- `Prediction` / `PredictionResult` - Predicted questions and evaluation results
- `ProjectContext` - Enterprise project definition
- `CodeArtifact` - Generated code file with metadata

### AI Client (`src/utils/ai_client.py`)
Abstraction layer supporting:
- **AnthropicClient** - Claude API (API key or session token auth)
- **OpenAIClient** - GPT-4 API
- **MockAIClient** - Testing

### Configuration (`src/core/config.py`)
Hierarchical config from env vars → YAML (`config/settings.yaml`) → defaults. Key configs: `AIConfig`, `PredictionConfig`, `SessionConfig`.

## Authentication

Two methods supported:
1. **API Key**: Set `ANTHROPIC_API_KEY` environment variable
2. **Session Token**: Set `ANTHROPIC_SESSION_TOKEN` for Claude.ai subscription accounts
   - Credentials stored at `~/.overnight/credentials.json`
   - CLI: `overnight login --method session|api_key`

## CLI Entry Point

Main CLI: `overnight` command (defined in pyproject.toml)

```bash
overnight start --name "Project" --component auth --component api --sessions 3
overnight login --method api_key
overnight auth-status
overnight serve --reload
```

## Testing

- Framework: pytest with pytest-asyncio
- Fixtures in `tests/conftest.py`: `settings`, `mock_ai_client`, `engine`, `session`, `orchestrator`
- Tests organized by component: `test_core.py`, `test_evaluators.py`, `test_predictors.py`

## Key Patterns

- **Lazy Initialization**: AI clients and embedding services created on first use
- **Strategy Pattern**: Runtime strategy selection with performance-based adaptation
- **Checkpoint Pattern**: Periodic state persistence for fault tolerance
- **Full Async**: All I/O operations use async/await
