# azure-ai-inference-plus

**The easier way to use Azure AI Inference SDK** ✨

[![PyPI Version](https://img.shields.io/pypi/v/azure-ai-inference-plus)](https://pypi.org/project/azure-ai-inference-plus/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/azure-ai-inference-plus)](https://pypi.org/project/azure-ai-inference-plus/)
[![License: MIT](https://img.shields.io/pypi/l/azure-ai-inference-plus)](https://opensource.org/licenses/MIT)

Enhanced wrapper that makes Azure AI Inference SDK simple and reliable with **automatic retry**, **JSON validation**, and **reasoning separation**.

## Why Use This Instead?

✅ **Reasoning separation** - automatically splits thinking from output (`.content` and `.reasoning`)  
✅ **Automatic retries** - never lose requests to transient failures  
✅ **JSON that works** - guaranteed valid JSON or automatic retry  
✅ **One import** - no need for multiple Azure SDK imports  
✅ **100% compatible** - drop-in replacement for Azure AI Inference SDK

## Installation

```bash
pip install azure-ai-inference-plus
```

Supports Python 3.10+

## Quick Start

```python
from azure_ai_inference_plus import ChatCompletionsClient, SystemMessage, UserMessage

# Uses environment variables: AZURE_AI_ENDPOINT, AZURE_AI_API_KEY
client = ChatCompletionsClient()

response = client.complete(
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="What's the capital of France?"),
    ],
    max_tokens=100,
    model="Codestral-2501"
)

print(response.choices[0].message.content)
# "The capital of France is Paris..."
```

**Or with manual credentials (everything from one import!):**

```python
from azure_ai_inference_plus import ChatCompletionsClient, SystemMessage, UserMessage, AzureKeyCredential

client = ChatCompletionsClient(
    endpoint="https://your-resource.services.ai.azure.com/models",
    credential=AzureKeyCredential("your-api-key")
)
```

## 🎯 Key Features

### 🧠 Automatic Reasoning Separation

**Game changer for reasoning models like DeepSeek-R1** - automatically separates thinking from output:

```python
response = client.complete(
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="What's 2+2? Think step by step."),
    ],
    model="DeepSeek-R1",
    reasoning_tags=["<think>", "</think>"]  # ✨ Auto-separation
)

# Clean output without reasoning clutter
print(response.choices[0].message.content)
# "2 + 2 equals 4."

# Access the reasoning separately
print(response.choices[0].message.reasoning)
# "Let me think about this step by step. 2 + 2 is a basic addition..."
```

### ✅ Guaranteed Valid JSON

No more JSON parsing errors - automatic validation and retry.

**Simple JSON (standard models like GPT-4o):**

```python
response = client.complete(
    messages=[
        SystemMessage(content="You are a helpful assistant that returns JSON."),
        UserMessage(content="Give me Tokyo info as JSON with keys: name, country, population"),
    ],
    max_tokens=500,
    model="gpt-4o",
    response_format="json_object"  # ✨ Auto-validation + retry
)

# Always valid JSON, no try/catch needed!
import json
data = json.loads(response.choices[0].message.content)  # ✅ Works perfectly
```

**JSON with reasoning models (like DeepSeek-R1):**

```python
response = client.complete(
    messages=[
        SystemMessage(content="You are a helpful assistant that returns JSON."),
        UserMessage(content="Give me Paris info as JSON with keys: name, country, population"),
    ],
    max_tokens=2000,  # More tokens needed for reasoning + JSON
    model="DeepSeek-R1",
    response_format="json_object",  # ✨ Clean JSON guaranteed
    reasoning_tags=["<think>", "</think>"]  # Required for reasoning separation
)

# Pure JSON - reasoning automatically stripped
data = json.loads(response.choices[0].message.content)  # {"name": "Paris", ...}

# But reasoning is still accessible
thinking = response.choices[0].message.reasoning  # "Let me think about Paris..."
```

_Note: JSON responses are automatically cleaned of markdown wrappers (like \`\`\`json blocks) for reliable parsing._

### 🔄 Smart Automatic Retries

Built-in retry with exponential backoff - no configuration needed:

```python
# Automatically retries on failures - just works!
response = client.complete(
    messages=[UserMessage(content="Tell me a joke")],
    model="Phi-4"
)
```

### ⚙️ Custom Retry (If Needed)

```python
from azure_ai_inference_plus import RetryConfig

# Override default behavior
client = ChatCompletionsClient(
    retry_config=RetryConfig(max_retries=5, delay_seconds=2.0)
)
```

### 📢 Retry Callbacks (Optional Observability)

Get notified when retries happen - perfect for logging and monitoring:

```python
from azure_ai_inference_plus import RetryConfig

def on_chat_retry(attempt, max_retries, exception, delay):
    print(f"🔄 Chat retry {attempt}/{max_retries}: {type(exception).__name__} - waiting {delay:.1f}s")

def on_json_retry(attempt, max_retries, message):
    print(f"📝 JSON retry {attempt}/{max_retries}: {message}")

# Add callbacks to your retry config
client = ChatCompletionsClient(
    retry_config=RetryConfig(
        max_retries=3,
        on_chat_retry=on_chat_retry,    # Called for general failures
        on_json_retry=on_json_retry     # Called for JSON validation failures
    )
)

# Now you'll see retry notifications:
# 🔄 Chat retry 1/3: HttpResponseError - waiting 1.0s
# 📝 JSON retry 2/3: Retry 2 after JSON validation failed
```

**Why callbacks?** The library doesn't print anything by default (clean for production), but callbacks let you add your own logging, metrics, or notifications exactly how you want them.

## 🚀 Embeddings Too

```python
from azure_ai_inference_plus import EmbeddingsClient

client = EmbeddingsClient()
response = client.embed(
    input=["Hello world", "Python is great"],
    model="text-embedding-3-large"
)
```

## Environment Setup

Create a `.env` file:

```bash
AZURE_AI_ENDPOINT=https://your-resource.services.ai.azure.com/models
AZURE_AI_API_KEY=your-api-key-here
```

## Migration from Azure AI Inference SDK

**2 simple steps:**

1. `pip install azure-ai-inference-plus`
2. Change your import:

   ```python
   # Before
   from azure.ai.inference import ChatCompletionsClient
   from azure.ai.inference.models import SystemMessage, UserMessage
   from azure.core.credentials import AzureKeyCredential

   # After
   from azure_ai_inference_plus import ChatCompletionsClient, SystemMessage, UserMessage, AzureKeyCredential
   ```

That's it! Your existing code works unchanged with automatic retries and JSON validation.

### Manual Credential Setup

```python
from azure_ai_inference_plus import ChatCompletionsClient, AzureKeyCredential

client = ChatCompletionsClient(
    endpoint="https://your-resource.services.ai.azure.com/models",
    credential=AzureKeyCredential("your-api-key")
)
```

## Examples

Check out the [`examples/`](examples/) directory for complete demonstrations:

- [`basic_usage.py`](examples/basic_usage.py) - **Reasoning separation**, JSON validation, and retry features
- [`embeddings_example.py`](examples/embeddings_example.py) - Embeddings with retry and credential setup
- [`callbacks_example.py`](examples/callbacks_example.py) - **Retry callbacks** for logging and monitoring

All examples show real-world usage patterns and advanced features.

## License

[MIT](./LICENSE)

## Contributing

Contributions are welcome! Whether it's bug fixes, feature additions, or documentation improvements, we appreciate your help in making this project better. For major changes or new features, please open an issue first to discuss what you would like to change.
