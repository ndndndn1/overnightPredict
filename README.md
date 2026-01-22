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

### Intelligent Strategy Selection

The Forecaster now implements context-aware strategy selection:

| Context Type | Detection Keywords | Description |
|--------------|-------------------|-------------|
| `error_debugging` | Error context present | Prioritizes strategies for debugging scenarios |
| `testing` | test, unittest, pytest, spec | Optimized for test-related predictions |
| `api_development` | api, endpoint, route, request, response | API-focused strategy selection |
| `refactoring` | refactor, optimize, clean, improve | Code improvement scenarios |
| `bug_fixing` | bug, fix, error, issue, problem | Bug resolution predictions |
| `feature_development` | feature, implement, add, create, build | New feature development |

**Scoring Weights:**
- **Accuracy (40%)**: Historical prediction accuracy
- **Context-Specific (35%)**: Performance in similar context types
- **Latency (15%)**: Response time optimization for large contexts
- **Experience (10%)**: Reliability based on prediction volume

### Key Features

- 📊 **Prediction Accuracy Tracking**: Semantic similarity-based evaluation
- 🔄 **Auto-Strategy Switching**: Automatically changes strategies when performance degrades
- 🧠 **Context-Aware Strategy Selection**: Intelligent strategy scoring based on context type (error debugging, testing, API development, refactoring, bug fixing, feature development)
- 📈 **Performance-Based Scoring**: Weighted scoring system (accuracy 40%, context-specific 35%, latency 15%, experience 10%)
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
DEEPSEEK_API_KEY=your-key-here
DEEPSEEK_ENABLED=false

# Claude - See "Claude Authentication" section below
CLAUDE_ENABLED=true
CLAUDE_AUTH_TYPE=api_key
CLAUDE_API_KEY=sk-ant-your-key-here
```

## Claude Authentication

Claude supports three authentication methods to accommodate different use cases:

### Option 1: API Key (Default)

Standard Anthropic API key authentication for API access:

```env
CLAUDE_AUTH_TYPE=api_key
CLAUDE_API_KEY=sk-ant-your-anthropic-key-here
```

### Option 2: OAuth (Claude Pro/Team Subscription)

Browser-based OAuth login for Claude Pro or Team subscribers. No API key needed - uses your existing subscription:

```env
CLAUDE_AUTH_TYPE=oauth
CLAUDE_OAUTH_CALLBACK_PORT=8080
```

When you start a session with Claude, a browser window will open for you to log in to your Claude account.

### Option 3: Session Key (From Browser)

Direct session key authentication using cookies from an active claude.ai session:

```env
CLAUDE_AUTH_TYPE=session_key
CLAUDE_SESSION_KEY=your-session-key-here
```

**How to get your session key:**
1. Open [claude.ai](https://claude.ai) in your browser and log in
2. Open Developer Tools (F12) → Application → Cookies
3. Find the `sessionKey` cookie and copy its value

### Subscription Account Settings (Optional)

For OAuth and Session Key authentication, you can optionally configure:

```env
CLAUDE_ACCOUNT_EMAIL=your-email@example.com
CLAUDE_SUBSCRIPTION_TYPE=pro  # free, pro, team, enterprise
CLAUDE_DAILY_MESSAGE_LIMIT=100  # Leave empty for auto-detection
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

The OODA loop is implemented in `SessionManager.process_question()` with full traceability:

```
┌─────────────────────────────────────────────────────────────┐
│                        OBSERVE                               │
│  - Receive actual question from user                        │
│  - Check pending predictions for matches                    │
│  - Returns: matched_prediction or None                      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                        ORIENT                                │
│  - Evaluate prediction accuracy (semantic similarity)        │
│  - Update context with new information                       │
│  - Track accuracy metrics via MetaTuner                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                        DECIDE                                │
│  - Use predicted answer if accuracy >= threshold            │
│  - Or execute fresh with LLM via Executor                   │
│  - Track: used_prediction, prediction_id, prediction_accuracy│
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                         ACT                                  │
│  - Return QuestionProcessingResult with full metadata       │
│  - Generate new predictions via Forecaster (lookahead)      │
│  - Apply strategy adjustments via MetaTuner if needed       │
└─────────────────────────────────────────────────────────────┘
```

### QuestionProcessingResult

Each question processing returns detailed metadata:

```python
@dataclass
class QuestionProcessingResult:
    question_id: str           # Unique identifier for the question
    answer: str                # The generated or predicted answer
    used_prediction: bool      # Whether a prediction was used
    prediction_id: str | None  # ID of matched prediction (if used)
    prediction_accuracy: float | None  # Similarity score (if matched)
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

### Claude Authentication

| Variable | Description | Default |
|----------|-------------|---------|
| `CLAUDE_AUTH_TYPE` | Authentication method: `api_key`, `oauth`, `session_key` | `api_key` |
| `CLAUDE_API_KEY` | Anthropic API key (for api_key auth) | - |
| `CLAUDE_OAUTH_CALLBACK_PORT` | Local port for OAuth callback | `8080` |
| `CLAUDE_SESSION_KEY` | Session key from browser (for session_key auth) | - |
| `CLAUDE_SUBSCRIPTION_TYPE` | Subscription tier: `free`, `pro`, `team`, `enterprise` | - |
| `CLAUDE_DAILY_MESSAGE_LIMIT` | Daily message limit for subscription | auto |

### Prediction Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `PREDICTION_ACCURACY_THRESHOLD` | Similarity threshold for matching | 0.7 |
| `PREDICTION_LOOKAHEAD_COUNT` | Questions to predict ahead | 5 |
| `PREDICTION_MIN_ACCURACY_FOR_KEEP` | Min accuracy before strategy switch | 0.6 |

### Orchestrator Settings

| Variable | Description | Default |
|----------|-------------|---------|
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
