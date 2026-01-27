# LLM Pivot

A unified interface for interacting with both online API-based and local LLM models.

## Overview

LLM Pivot provides a consistent API for working with different types of language models:
- **Online Models**: OpenAI, Anthropic, and other API-based services
- **Local Models**: Ollama and other locally-hosted models

This project is private and requires collaboration access. Contact the repository administrator for access.

## Quick Start

### Installation

#### Method 1: Git Dependency (Recommended)

Check the [Releases](https://github.com/Aquarids/llm-pivot/releases) page for the latest version.

**For SSH access (private repository):**

Add to your `requirements.txt`:

```txt
llm-pivot @ git+ssh://git@github.com/Aquarids/llm-pivot.git@v0.0.1
```

Or install it directly:

```bash
pip install git+ssh://git@github.com/Aquarids/llm-pivot.git@v0.0.1
```

**Prerequisites:**
- Ensure your SSH key is added to GitHub: [GitHub SSH Keys](https://github.com/settings/keys)
- Test SSH connection: `ssh -T git@github.com`

#### Method 2: Local Development

```bash
# Clone the repository
git clone https://github.com/Aquarids/llm-pivot.git
cd llm-pivot

# Create environment
conda env create -f environment.yml
conda activate llm-pivot

# Install in editable mode
pip install -e .
```

In your project:

```bash
cd /path/to/your/project
conda activate your-project-env
pip install -e /path/to/llm-pivot
```

### Quick Start

#### Basic Usage

```python
from llmpivot import LLMPivot, PivotConfig

config = PivotConfig(
    model_type="online",
    model_id="gpt-4",
    api_key="your-api-key",
    base_url="https://api.openai.com/v1/"
)

with LLMPivot(config) as llm:
    messages = [
        {"role": "user", "content": "Hello, how are you?"}
    ]
    response = llm.dialogue(messages)
    print(response)
```

#### Online API Models

```python
from llmpivot import LLMPivot, PivotConfig

api_config = PivotConfig(
    model_type="online",
    model_id="gpt-4",
    api_key="sk-xxx",
    base_url="https://api.openai.com/v1/",
    temperature=0.7,
    max_tokens=2000
)

llm = LLMPivot(api_config)

messages = [{"role": "user", "content": "Explain quantum computing"}]
response = llm.generate(messages)
print(response['content'])
```

#### Local Models (Ollama)

```python
from llmpivot import LLMPivot, PivotConfig

local_config = PivotConfig(
    model_type="local",
    model_id="qwen2.5:7b",
    base_url="http://localhost:11434"
)

llm = LLMPivot(local_config)

messages = [{"role": "user", "content": "What is machine learning?"}]
response = llm.dialogue(messages)
print(response)
```

#### Streaming Response

```python
messages = [{"role": "user", "content": "Write a short story"}]

for chunk in llm.stream_generate(messages):
    print(chunk, end='', flush=True)
```

#### Function Calling

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"]
            }
        }
    }
]

messages = [{"role": "user", "content": "What's the weather in Beijing?"}]
tool_calls = llm.call_function(messages, tools)
print(tool_calls)
```

### Configuration

#### PivotConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| model_type | str | "online" | "online" or "local" |
| model_id | str | "gpt-4" | Model identifier |
| api_key | str | None | API key for online models |
| base_url | str | None | API base URL |
| temperature | float | 0.7 | Sampling temperature (0.0-2.0) |
| max_tokens | int | None | Maximum tokens to generate |
| timeout | int | 60 | Request timeout in seconds |
| max_retries | int | 1 | Maximum retry attempts |
| enable_log | bool | True | Enable logging |


## Development

### Configuration File

Create `tests/local_config.ini`:

```ini
[online]
model_id = gpt-4
api_key = sk-your-api-key
base_url = https://api.openai.com/v1/
enable_log = false

[local]
model_id = qwen2.5:7b
base_url = http://localhost:11434
enable_log = false
```

### Running Tests

```bash
# All tests
pytest

# Only API tests
pytest -m api

# Only local model tests
pytest -m local

# Skip integration tests
pytest -m "not integration"
```

### Release Guide

#### Update Version Number

Edit `pyproject.toml`:

```toml
[project]
name = "llmpivot"
version = "0.2.0"
```

#### Create Git Tag

```bash
# Create annotated tag
git tag -a v0.2.0 -m "Release version 0.2.0"

# Push tag to remote
git push origin v0.2.0
```

#### Create GitHub Release (Optional)

Go to GitHub repository → Releases → Create a new release:
- Tag: `v0.2.0`
- Title: `Release v0.2.0`
- Description: `Changelog`

## License

Internal use only. Not for public distribution.
