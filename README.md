# LLM Pivot

A unified interface for interacting with both online API-based and local LLM models.

## Overview

LLM Pivot provides a consistent API for working with different types of models:
- **Online Models**: OpenAI, Anthropic, and other API-based services
- **Local Models**: Ollama (LLM), Infinity (Embedding / Reranking), and other locally-hosted models

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

### Basic Usage

#### Online LLM API Models

```python
from llmpivot import LLMPivot, PivotConfig

# Basic configuration
config = PivotConfig(
    model_type="online",
    model_id="gpt-4",
    api_key="sk-xxx",
    base_url="https://api.openai.com/v1/"
)

# Use context manager for automatic cleanup
with LLMPivot(config) as llm:
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    response = llm.dialogue(messages)
    print(response)
```

#### Local LLM Models (Ollama)

```python
from llmpivot import LLMPivot, PivotConfig

config = PivotConfig(
    model_type="local",
    model_id="qwen2.5:7b",
    base_url="http://localhost:11434"
)

with LLMPivot(config) as llm:
    messages = [{"role": "user", "content": "What is machine learning?"}]
    response = llm.dialogue(messages)
    print(response)
```

#### Embedding Models

```python
from llmpivot import EmbedPivot, EmbedConfig

config = EmbedConfig(
    model_type="online",
    model_id="Qwen3-Embedding-0.6B",
    api_key="sk-xxx",
    base_url="https://api.example.com"
)

with EmbedPivot(config) as embed:
    vector = embed.embedding("Hello, world!")
    print(len(vector))  # Vector dimension

    vectors = embed.embedding(["Hello", "World"])
    print(len(vectors))  # 2
```

#### Rerank Models

```python
from llmpivot import RerankPivot, RerankConfig

config = RerankConfig(
    model_type="online",
    model_id="bce-reranker-base_v1",
    api_key="sk-xxx",
    base_url="https://api.example.com"
)

with RerankPivot(config) as rerank:
    documents = [
        "Python is a programming language.",
        "The Eiffel Tower is in Paris.",
        "Machine learning is a subset of AI.",
    ]
    results = rerank.rerank_documents("What is machine learning?", documents)
    print(results)
    # [{"index": 2, "text": "...", "relevance_score": 0.98}, ...]

    scores = rerank.similarity("AI includes machine learning.", documents)
    print(scores)  # [0.72, 0.11, 0.95]
```


### Advanced Usage

#### Custom Default Parameters

Set default parameters for all LLM calls:

```python
config = PivotConfig(
    model_type="online",
    model_id="gpt-4",
    api_key="sk-xxx",
    llm_default_params={
        "temperature": 0.3,
        "max_tokens": 1000,
        "top_p": 0.9,
    }
)

llm = LLMPivot(config)

# Uses default parameters (temperature=0.3, max_tokens=1000, top_p=0.9)
response = llm.generate(messages=[{"role": "user", "content": "Hello"}])
```

#### Override Parameters at Runtime

```python
# Override specific parameters for this call
response = llm.generate(
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.8,
    max_tokens=2000,
)

# Use streaming internally (still returns complete string)
response = llm.dialogue(
    messages=[{"role": "user", "content": "Explain quantum computing"}],
    stream=True,
    temperature=0.7
)
print(response)
```

#### Streaming Response

```python
messages = [{"role": "user", "content": "Write a short story"}]

# Stream generation for chunk-by-chunk processing
for chunk in llm.stream_generate(messages, temperature=0.7):
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
# Output: [{"id": "call_xxx", "name": "get_weather", "arguments": '{"city": "Beijing"}'}]
```

#### Provider-Specific Features

**DeepSeek Thinking Mode:**

```python
response = llm.generate(
    messages=[{"role": "user", "content": "Solve this complex problem"}],
    extra_body={"thinking": {"type": "enabled"}}
)
```

**JSON Mode:**

```python
response = llm.generate(
    messages=[{"role": "user", "content": "Generate a JSON object"}],
    response_format={"type": "json_object"}
)
```

**Ollama-Specific Parameters:**

```python
config = PivotConfig(
    model_type="local",
    model_id="llama3",
    llm_default_params={
        "temperature": 0.7,
        "top_k": 40,
        "repeat_penalty": 1.1,
        "num_ctx": 4096,
    }
)
```

#### Embeddings

```python
# Single text
embedding = llm.embedding("Hello world")
print(len(embedding))  # Vector dimension

# Multiple texts
texts = ["Hello", "World", "AI"]
embeddings = llm.embedding(texts)
print(len(embeddings))  # 3
print(len(embeddings[0]))  # Vector dimension
```

### Configuration

#### PivotConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| model_type | str | "online" | "online" or "local" |
| model_id | str | "gpt-4" | Model identifier |
| api_key | str | None | API key for online models |
| base_url | str | None | API base URL or Ollama host |
| timeout | int | 60 | Request timeout in seconds |
| max_retries | int | 1 | Maximum retry attempts |
| llm_default_params | dict | {"temperature": 0.7} | Default parameters for LLM calls |
| log_env | str | "local" | Log environment identifier |
| enable_log | bool | True | Enable logging |
| log_level | str | None | Log level (DEBUG, INFO, WARNING, ERROR) |

#### EmbedConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| model_type | str | "online" | "online" or "local" |
| model_id | str | "Qwen3-Embedding-0.6B" | Model identifier |
| api_key | str | None | API key for online models |
| base_url | str | None | API base URL or Infinity host |
| timeout | int | 60 | Request timeout in seconds |
| max_retries | int | 1 | Maximum retry attempts |
| embed_default_params | dict | {"encoding_format": "float"} | Default parameters for embedding calls |
| log_env | str | "local" | Log environment identifier |
| enable_log | bool | True | Enable logging |
| log_level | str | None | Log level (DEBUG, INFO, WARNING, ERROR) |

#### RerankConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| model_type | str | "online" | "online" or "local" |
| model_id | str | "bce-reranker-base_v1" | Model identifier |
| api_key | str | None | API key for online models |
| base_url | str | None | API base URL or Infinity host |
| timeout | int | 60 | Request timeout in seconds |
| max_retries | int | 1 | Maximum retry attempts |
| rerank_default_params | dict | {"top_n": 5, "normalize": true} | Default parameters for rerank calls |
| log_env | str | "local" | Log environment identifier |
| enable_log | bool | True | Enable logging |
| log_level | str | None | Log level (DEBUG, INFO, WARNING, ERROR) |


#### Common LLM Parameters (via llm_default_params or kwargs)

**Universal Parameters:**
- `temperature` (float): Sampling temperature (0.0-2.0)
- `max_tokens` (int): Maximum tokens to generate
- `top_p` (float): Nucleus sampling threshold

**Online API (OpenAI-compatible):**
- `frequency_penalty` (float): Frequency penalty (-2.0 to 2.0)
- `presence_penalty` (float): Presence penalty (-2.0 to 2.0)
- `response_format` (dict): Response format specification
- `extra_body` (dict): Provider-specific parameters

**Local (Ollama):**
- `top_k` (int): Top-k sampling
- `repeat_penalty` (float): Repetition penalty
- `num_ctx` (int): Context window size
- `seed` (int): Random seed for reproducibility

### API Reference

#### Core Methods

```python
# Generate with full response details
response = llm.generate(messages, tools=None, **kwargs)
# Returns: {"content": str, "role": str, "tool_calls": list, "usage": dict, ...}

# Simple dialogue (returns only content string)
content = llm.dialogue(messages, stream=False, **kwargs)
# Returns: str

# Dialogue with streaming API (uses streaming internally, returns complete string)
content = llm.dialogue(messages, stream=True, **kwargs)
# Returns: str

# Function calling
tool_calls = llm.call_function(messages, tools, **kwargs)
# Returns: [{"id": str, "name": str, "arguments": str}, ...]

# Streaming generation
for chunk in llm.stream_generate(messages, tools=None, **kwargs):
    print(chunk)  # str

# Embeddings
embedding = llm.embedding(text, **kwargs)
# Returns: List[float] or List[List[float]]

# Perplexity calculation
ppl = llm.perplexity(text, **kwargs)
# Returns: float
```

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

[online_embed]
model_id = Qwen3-Embedding-0.6B
api_key = sk-your-api-key
base_url = https://api.example.com
enable_log = false

[local_embed]
model_id = Qwen3-Embedding-0.6B
base_url = http://localhost:7997
enable_log = false

[online_rerank]
model_id = bce-reranker-base_v1
api_key = sk-your-api-key
base_url = https://api.example.com
enable_log = false

[local_rerank]
model_id = bce-reranker-base_v1
base_url = http://localhost:7997
enable_log = false

```

### Running Tests

Local build in editable mode
```bash
pip install -e .
```

Then run tests
```bash
# All tests
pytest

# Only API tests
pytest -m api

# Only local model tests
pytest -m local

# Skip integration tests
pytest -m "not integration"

# Verbose output
pytest -v
```

### Release Guide

#### Update Version Number

Edit `pyproject.toml`:

```toml
[project]
name = "llm-pivot"
version = "0.2.0"
```

#### Create Git Tag

```bash
# Create annotated tag
git tag -a v0.2.0 -m "Release version 0.2.0"

# Push tag to remote
git push origin v0.2.0
```

#### Create GitHub Release

Go to GitHub repository → Releases → Create a new release:
- Tag: `v0.2.0`
- Title: `Release v0.2.0`
- Description: Add changelog and release notes

## Examples

### Multi-turn Conversation

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"},
]

response1 = llm.dialogue(messages)
messages.append({"role": "assistant", "content": response1})
messages.append({"role": "user", "content": "Show me an example"})

response2 = llm.dialogue(messages)
print(response2)
```

### Batch Processing

```python
questions = [
    "What is AI?",
    "What is ML?",
    "What is DL?"
]

for question in questions:
    response = llm.dialogue([{"role": "user", "content": question}])
    print(f"Q: {question}\nA: {response}\n")
```

### Error Handling

```python
from llmpivot import LLMPivot, PivotConfig

config = PivotConfig(
    model_type="online",
    model_id="gpt-4",
    api_key="sk-xxx",
    max_retries=3,
    timeout=30
)

try:
    with LLMPivot(config) as llm:
        response = llm.dialogue([{"role": "user", "content": "Hello"}])
        print(response)
except Exception as e:
    print(f"Error: {e}")
```

## License

Internal use only. Not for public distribution.
