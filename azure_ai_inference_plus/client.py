"""Enhanced client classes that extend Azure AI Inference clients"""

import os
from typing import Any, Dict, List, Literal, Optional, Union

from azure.ai.inference import ChatCompletionsClient as AzureChatCompletionsClient
from azure.ai.inference import EmbeddingsClient as AzureEmbeddingsClient
from azure.ai.inference.models import (
    AssistantMessage,
    JsonSchemaFormat,
    SystemMessage,
    UserMessage,
)
from azure.core.credentials import AzureKeyCredential

from .config import RetryConfig
from .exceptions import ConfigurationError
from .utils import (
    build_endpoint_url,
    process_response_with_reasoning,
    retry_with_config,
)


class ChatCompletionsClient(AzureChatCompletionsClient):
    """
    Enhanced ChatCompletionsClient with retry mechanism and JSON validation.

    This class extends the original Azure ChatCompletionsClient with:
    - Automatic retry logic for transient failures
    - JSON response validation when response_format="json_object"
    - Environment variable support for credentials
    - Improved endpoint URL handling
    - Optional callbacks for retry events (on_chat_retry, on_json_retry)

    Example with callbacks:
        def on_retry(attempt, max_retries, exception, delay):
            print(f"Retrying {attempt}/{max_retries} after {type(exception).__name__}: {exception}")

        def on_json_retry(attempt, max_retries, message):
            print(f"JSON retry {attempt}/{max_retries}: {message}")

        retry_config = RetryConfig(
            max_retries=3,
            on_chat_retry=on_retry,
            on_json_retry=on_json_retry
        )
        client = ChatCompletionsClient(retry_config=retry_config)
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        credential: Optional[AzureKeyCredential] = None,
        api_version: str = "2024-05-01-preview",
        retry_config: Optional[RetryConfig] = None,
        **kwargs,
    ):
        """
        Initialize the enhanced ChatCompletionsClient.

        Args:
            endpoint: Azure AI endpoint URL (can be set via AZURE_AI_ENDPOINT env var)
            credential: AzureKeyCredential (can be created from AZURE_AI_API_KEY env var)
            api_version: API version to use
            retry_config: Retry configuration (uses defaults if not provided)
            **kwargs: Additional arguments passed to the base client
        """
        # Handle endpoint from environment
        if endpoint is None:
            endpoint = os.getenv("AZURE_AI_ENDPOINT")

        if endpoint is None:
            raise ConfigurationError(
                "Endpoint must be provided or set via AZURE_AI_ENDPOINT environment variable"
            )

        # Handle credential from environment
        if credential is None:
            api_key = os.getenv("AZURE_AI_API_KEY")
            if api_key is None:
                raise ConfigurationError(
                    "Credential must be provided or API key set via AZURE_AI_API_KEY environment variable"
                )
            credential = AzureKeyCredential(api_key)

        # Build proper endpoint URL
        endpoint = build_endpoint_url(endpoint)

        # Set up retry configuration
        self.retry_config = retry_config or RetryConfig()

        # Initialize the base client
        super().__init__(
            endpoint=endpoint, credential=credential, api_version=api_version, **kwargs
        )

    def complete(
        self,
        messages: List[Union[SystemMessage, UserMessage, AssistantMessage]],
        model: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: bool = False,
        response_format: Optional[
            Union[Literal["text", "json_object"], JsonSchemaFormat]
        ] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[Dict[int, float]] = None,
        user: Optional[str] = None,
        seed: Optional[int] = None,
        reasoning_tags: Optional[List[str]] = None,
        retry_config: Optional[RetryConfig] = None,
        **kwargs,
    ):
        """
        Generate chat completion with enhanced retry mechanism and reasoning separation.

        This method has the same signature as the original complete() method,
        but adds automatic retry logic, JSON validation, and reasoning parsing.

        When response_format="json_object", the response will be automatically
        validated as JSON and retried if invalid.

        When reasoning_tags is provided, reasoning content between the tags will be:
        - For JSON mode: Removed entirely to ensure clean JSON output
        - For non-JSON mode: Separated into a 'reasoning' field on the message

        Args:
            reasoning_tags: Optional list with [start_tag, end_tag] to parse reasoning
                          content. Example: ["<think>", "</think>"]
        """
        # Use provided retry config or fall back to instance config
        config = retry_config or self.retry_config

        # Check if JSON validation is needed and if we have reasoning tags
        is_json_mode = response_format == "json_object"
        json_validation = is_json_mode
        has_reasoning_tags = reasoning_tags and len(reasoning_tags) == 2

        # Prepare arguments exactly as the original method expects
        # Filter out None values, but handle stream specially (only add if True)
        optional_params = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": stop,
            "response_format": response_format,
            "tools": tools,
            "tool_choice": tool_choice,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "logit_bias": logit_bias,
            "user": user,
            "seed": seed,
        }

        # Filter out None values
        filtered_params = {k: v for k, v in optional_params.items() if v is not None}

        # Only add stream if explicitly set to True
        if stream is True:
            filtered_params["stream"] = stream

        completion_kwargs = {
            "messages": messages,
            "model": model,
            **filtered_params,
            **kwargs,
        }

        # Apply retry decorator to the base class method
        @retry_with_config(
            config, json_validation=json_validation, reasoning_tags=reasoning_tags
        )
        def _complete():
            result = super(ChatCompletionsClient, self).complete(**completion_kwargs)

            # Process reasoning if tags are provided
            if has_reasoning_tags:
                result = process_response_with_reasoning(
                    result, reasoning_tags, is_json_mode
                )

            return result

        return _complete()


class EmbeddingsClient(AzureEmbeddingsClient):
    """
    Enhanced EmbeddingsClient with retry mechanism.

    This class extends the original Azure EmbeddingsClient with:
    - Automatic retry logic for transient failures
    - Environment variable support for credentials
    - Improved endpoint URL handling
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        credential: Optional[AzureKeyCredential] = None,
        retry_config: Optional[RetryConfig] = None,
        **kwargs,
    ):
        """
        Initialize the enhanced EmbeddingsClient.

        Args:
            endpoint: Azure AI endpoint URL (can be set via AZURE_AI_ENDPOINT env var)
            credential: AzureKeyCredential (can be created from AZURE_AI_API_KEY env var)
            retry_config: Retry configuration (uses defaults if not provided)
            **kwargs: Additional arguments passed to the base client
        """
        # Handle endpoint from environment
        if endpoint is None:
            endpoint = os.getenv("AZURE_AI_ENDPOINT")

        if endpoint is None:
            raise ConfigurationError(
                "Endpoint must be provided or set via AZURE_AI_ENDPOINT environment variable"
            )

        # Handle credential from environment
        if credential is None:
            api_key = os.getenv("AZURE_AI_API_KEY")
            if api_key is None:
                raise ConfigurationError(
                    "Credential must be provided or API key set via AZURE_AI_API_KEY environment variable"
                )
            credential = AzureKeyCredential(api_key)

        # Set up retry configuration
        self.retry_config = retry_config or RetryConfig()

        # Initialize the base client
        super().__init__(endpoint=endpoint, credential=credential, **kwargs)

    def embed(
        self,
        input: Union[str, List[str]],
        model: str,
        encoding_format: Optional[str] = None,
        dimensions: Optional[int] = None,
        user: Optional[str] = None,
        retry_config: Optional[RetryConfig] = None,
        **kwargs,
    ):
        """
        Generate embeddings with enhanced retry mechanism.

        This method has the same signature as the original embed() method,
        but adds automatic retry logic for transient failures.
        """
        # Use provided retry config or fall back to instance config
        config = retry_config or self.retry_config

        # Prepare arguments exactly as the original method expects
        # Filter out None values
        optional_params = {
            "encoding_format": encoding_format,
            "dimensions": dimensions,
            "user": user,
        }
        filtered_params = {k: v for k, v in optional_params.items() if v is not None}

        embed_kwargs = {"input": input, "model": model, **filtered_params, **kwargs}

        # Apply retry decorator to the base class method
        @retry_with_config(config, json_validation=False)
        def _embed():
            return super(EmbeddingsClient, self).embed(**embed_kwargs)

        return _embed()


# Backward compatibility aliases
ChatClient = ChatCompletionsClient  # For those who prefer the shorter name
