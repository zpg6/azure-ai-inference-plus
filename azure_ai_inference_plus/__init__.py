"""
Azure AI Inference Plus - Enhanced wrapper for Azure AI Inference SDK
"""

# Re-export commonly used classes from Azure AI Inference SDK for convenience
from azure.ai.inference.models import (
    AssistantMessage,
    ChatRequestMessage,
    JsonSchemaFormat,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from azure.core.credentials import AzureKeyCredential

from .client import (
    AzureChatCompletionsClientKwargs,
    AzureEmbeddingsClientKwargs,
    ChatClient,
    ChatCompletionsClient,
    EmbeddingsClient,
)
from .config import RetryConfig
from .exceptions import (
    AzureAIInferencePlusError,
    ConfigurationError,
    JSONValidationError,
    RetryExhaustedError,
)

__version__ = "1.0.4"
__all__ = [
    "AzureChatCompletionsClientKwargs",
    "AzureEmbeddingsClientKwargs",
    "ChatCompletionsClient",
    "ChatClient",  # Alias for ChatCompletionsClient
    "EmbeddingsClient",
    "RetryConfig",
    "AzureAIInferencePlusError",
    "JSONValidationError",
    "RetryExhaustedError",
    "ConfigurationError",
    # Re-exported message classes
    "SystemMessage",
    "UserMessage",
    "AssistantMessage",
    "ToolMessage",
    "ChatRequestMessage",
    "JsonSchemaFormat",
    # Re-exported credential class
    "AzureKeyCredential",
]
