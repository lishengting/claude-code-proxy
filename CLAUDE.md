# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a proxy server that enables Claude Code to work with OpenAI-compatible API providers. It converts Claude API requests to OpenAI API calls, allowing you to use various LLM providers through the Claude Code CLI.

Key features:
- Full Claude API Compatibility for `/v1/messages` endpoint
- Multiple provider support (OpenAI, Azure OpenAI, local models like Ollama)
- Smart model mapping (BIG, MIDDLE, SMALL models)
- Function calling support with proper conversion
- Streaming responses with SSE support
- Image input support (base64 encoded)
- Client API key validation

## Project Architecture

```
claude-code-proxy/
├── src/
│   ├── main.py                 # Application entry point and FastAPI setup
│   ├── api/endpoints.py        # API route handlers
│   ├── core/                   # Core functionality
│   │   ├── client.py           # OpenAI client with cancellation support
│   │   ├── config.py           # Configuration management
│   │   ├── constants.py        # Application constants
│   │   ├── model_manager.py    # Claude to OpenAI model mapping
│   ├── conversion/             # Request/response format conversion
│   │   ├── request_converter.py
│   │   ├── response_converter.py
│   ├── models/                 # Pydantic models for request/response validation
│   │   ├── claude.py           # Claude API data models
│   │   ├── openai.py           # OpenAI API data models
│   └── test_claude_to_openai.py # Integration tests
├── tests/
│   └── test_main.py            # Test suite
├── start_proxy.py              # Startup script
├── Dockerfile                  # Container definition
├── docker-compose.yml          # Docker deployment configuration
├── pyproject.toml              # Project dependencies and metadata
├── requirements.txt            # Dependencies list
└── .env.example                # Configuration template
```

## Common Development Tasks

### Running the Application

```bash
# Install dependencies
uv sync

# Start the proxy server
uv run start_proxy.py

# Or with Docker
docker compose up -d
```

### Testing

```bash
# Run all tests
python tests/test_main.py

# Run specific test functions
python -c "
import asyncio
from tests.test_main import test_basic_chat
asyncio.run(test_basic_chat())
"
```

### Code Formatting and Type Checking

```bash
# Format code
uv run black src/
uv run isort src/

# Type checking
uv run mypy src/
```

## API Endpoints

- `POST /v1/messages` - Main Claude API endpoint
- `POST /v1/messages/count_tokens` - Token counting
- `GET /health` - Health check
- `GET /test-connection` - OpenAI connectivity test
- `GET /` - Root endpoint with configuration info

## Configuration

Key environment variables:
- `OPENAI_API_KEY` - Required API key for target provider
- `ANTHROPIC_API_KEY` - Optional client validation key
- `OPENAI_BASE_URL` - API endpoint (default: https://api.openai.com/v1)
- `BIG_MODEL`, `MIDDLE_MODEL`, `SMALL_MODEL` - Model mappings
- `HOST`, `PORT` - Server settings

## Model Mapping

The proxy maps Claude models to configured OpenAI models:
- Models with "haiku" → `SMALL_MODEL` (default: gpt-4o-mini)
- Models with "sonnet" → `MIDDLE_MODEL` (default: gpt-4o)
- Models with "opus" → `BIG_MODEL` (default: gpt-4o)

## Integration with Claude Code

To use with Claude Code:
```bash
# Start the proxy
uv run start_proxy.py

# Use Claude Code with the proxy
ANTHROPIC_BASE_URL=http://localhost:8082 claude
```

## Recent Changes

- Updated MIDDLE_MODEL config to default to BIG_MODEL value for consistency