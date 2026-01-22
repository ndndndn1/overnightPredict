# OvernightPredict

**Recursive Agent System with Self-Correction (메타인지 기반 자율 수정 시스템)**

An enterprise-grade system that implements meta-cognition capabilities to evaluate and adjust its own prediction strategies in real-time.

## Features

### Core Capabilities

- **OODA Loop Implementation**: Forecast → Execute → Evaluate → Tune cycle running in parallel
- **Meta-Cognition**: Real-time self-evaluation and strategy adjustment
- **Multi-Provider Support**: OpenAI, DeepSeek, Claude/Anthropic
- **Parallel Sessions**: Run multiple sessions simultaneously with shared context
- **Clean Architecture**: Domain-driven design with clear boundaries
- **Strategy Pattern**: Dynamic algorithm replacement based on performance

### Architecture Components

| Component | Role | Description |
|-----------|------|-------------|
| **Orchestrator** | Coordinator | Manages multiple sessions in parallel |
| **Forecaster** | Prefrontal Cortex | Predicts future questions/tasks based on context |
| **Executor** | Worker | Executes tasks and generates code |
| **Evaluator** | Critic | Measures semantic similarity between predicted and actual questions |
| **MetaTuner** | Optimizer | Adjusts strategies when accuracy drops |

### Key Features

- 📊 **Prediction Accuracy Tracking**: Semantic similarity-based evaluation
- 🔄 **Auto-Strategy Switching**: Automatically changes strategies when performance degrades
- 🌐 **Context Sharing**: Share progress across sessions (file, cloud, Redis)
- ⏱️ **Rate Limit Handling**: Automatic wait and retry for Claude Code limits
- 📈 **Real-time Dashboard**: TUI dashboard for monitoring all sessions

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/overnightPredict.git
cd overnightPredict

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Copy the example configuration and set your API keys:

```bash
cp config/.env.example .env
```

Edit `.env` with your credentials:

```env
# Required: At least one provider
OPENAI_API_KEY=sk-your-key-here
OPENAI_ENABLED=true

# Optional: Additional providers
CLAUDE_API_KEY=sk-ant-your-key-here
CLAUDE_ENABLED=true

DEEPSEEK_API_KEY=your-key-here
DEEPSEEK_ENABLED=false
```

## Usage

### Interactive CLI

```bash
python main.py
```

This starts the interactive CLI where you can:
1. Configure providers
2. Create sessions
3. Process questions
4. Monitor status
5. Launch the dashboard

### Quick Session Creation

```bash
# Create a session with OpenAI
python main.py --create openai --prompt "Build a REST API for user management"

# Create parallel sessions across providers
python main.py --parallel openai,claude --prompt "Implement a caching system"
```

### TUI Dashboard

```bash
python main.py --dashboard
```

Launch the real-time monitoring dashboard to:
- View all sessions status
- Start/stop sessions
- Send questions to sessions
- Monitor prediction accuracy

## Architecture

```
src/
├── domain/                 # Enterprise business rules
│   ├── entities/          # Session, Prediction, Question, Task
│   ├── value_objects/     # AccuracyScore, Context
│   └── interfaces/        # Ports for infrastructure
│
├── application/           # Application business rules
│   ├── services/          # Orchestrator, Forecaster, Executor, Evaluator, MetaTuner
│   ├── use_cases/         # Application use cases
│   └── dto/               # Data transfer objects
│
├── infrastructure/        # External interfaces
│   ├── llm_providers/     # OpenAI, DeepSeek, Claude implementations
│   ├── storage/           # SQLite repository, Event bus
│   ├── context_sharing/   # File, Cloud, Redis sharing
│   └── rate_limiting/     # Token bucket, Sliding window
│
└── presentation/          # UI Layer
    ├── cli/               # Command-line interface
    ├── dashboard/         # TUI dashboard (Textual)
    └── controllers/       # Session controllers
```

## OODA Loop Flow

```
┌─────────────────────────────────────────────────────────────┐
│                        OBSERVE                               │
│  - Receive actual question from user                        │
│  - Check pending predictions for matches                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                        ORIENT                                │
│  - Evaluate prediction accuracy (semantic similarity)        │
│  - Update context with new information                       │
│  - Track accuracy metrics                                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                        DECIDE                                │
│  - Use predicted answer if accuracy >= threshold            │
│  - Or execute fresh with LLM                                │
│  - Determine if strategy change needed                      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                         ACT                                  │
│  - Return answer to user                                    │
│  - Generate new predictions (lookahead)                     │
│  - Apply strategy adjustments if needed                     │
└─────────────────────────────────────────────────────────────┘
```

## Context Sharing

Enable sessions to share context and progress:

```env
# File-based (same machine)
CONTEXT_ENABLED=true
CONTEXT_SHARING_TYPE=file
CONTEXT_SHARED_PATH=.overnight/shared

# Cloud-based (distributed)
CONTEXT_ENABLED=true
CONTEXT_SHARING_TYPE=cloud_bucket
CONTEXT_BUCKET_NAME=my-bucket
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_entities.py -v
```

## API Reference

### SessionController

```python
from src.presentation.controllers import SessionController

controller = SessionController(orchestrator)

# Create session
session_id = await controller.create_session(
    provider="openai",
    name="My Session",
    initial_prompt="Build a web scraper",
)

# Process question
answer = await controller.ask(session_id, "How should I handle pagination?")

# Get status
status = controller.get_status(session_id)
print(f"Accuracy: {status['metrics']['prediction_accuracy']:.1%}")
```

### Orchestrator

```python
from src.application.services.orchestrator import Orchestrator

# Create parallel sessions
group_id = await orchestrator.create_session_group(
    providers=[LLMProvider.OPENAI, LLMProvider.CLAUDE],
    initial_prompt="Implement authentication system",
)

# Share context across group
await orchestrator.broadcast_context(
    group_id=group_id,
    content="User model: id, email, password_hash",
    context_type=ContextType.CODE,
)
```

## Configuration Reference

| Variable | Description | Default |
|----------|-------------|---------|
| `PREDICTION_ACCURACY_THRESHOLD` | Similarity threshold for matching | 0.7 |
| `PREDICTION_LOOKAHEAD_COUNT` | Questions to predict ahead | 5 |
| `PREDICTION_MIN_ACCURACY_FOR_KEEP` | Min accuracy before strategy switch | 0.6 |
| `ORCHESTRATOR_MAX_SESSIONS` | Maximum concurrent sessions | 10 |
| `ORCHESTRATOR_SESSION_TIMEOUT` | Session timeout in seconds | 3600 |

## License

MIT License - See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest`
5. Submit a pull request
