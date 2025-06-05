"""
Azure AI Inference Plus - Enhanced wrapper for Azure AI Inference SDK
"""

from .client import ChatCompletionsClient, EmbeddingsClient, ChatClient
from .config import RetryConfig
from .exceptions import (
    AzureAIInferencePlusError,
    JSONValidationError,
    RetryExhaustedError,
    ConfigurationError,
)

# Re-export commonly used classes from Azure AI Inference SDK for convenience
from azure.ai.inference.models import (
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolMessage,
    ChatRequestMessage,
    JsonSchemaFormat,
)
from azure.core.credentials import AzureKeyCredential

__version__ = "1.0.0"
__all__ = [
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
