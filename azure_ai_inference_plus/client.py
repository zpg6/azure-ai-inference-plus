"""Enhanced client classes that extend Azure AI Inference clients"""

import os
from typing import Any, Dict, List, Literal, Optional, TypedDict, Union, Unpack

from azure.ai.inference import ChatCompletionsClient as AzureChatCompletionsClient
from azure.ai.inference import EmbeddingsClient as AzureEmbeddingsClient
from azure.ai.inference.models import (
    AssistantMessage,
    ChatCompletionsNamedToolChoice,
    ChatCompletionsToolChoicePreset,
    ChatCompletionsToolDefinition,
    JsonSchemaFormat,
    SystemMessage,
    UserMessage,
)
from azure.core.credentials import AzureKeyCredential, TokenCredential

from .config import RetryConfig
from .exceptions import ConfigurationError
from .utils import (
    build_endpoint_url,
    process_response_with_reasoning,
    retry_with_config,
)


class AzureChatCompletionsClientKwargs(TypedDict, total=False):
    """
    Keyword arguments that can be passed to the Azure ChatCompletionsClient constructor.

    Only includes parameters that users would reasonably want to configure
    when using our enhanced wrapper library.

    Args:
        temperature: The sampling temperature to use that controls the apparent creativity of
            generated completions. Higher values will make output more random while lower values
            will make results more focused and deterministic. Supported range is [0, 1].
            Default value is None.
        max_tokens: The maximum number of tokens to generate. Default value is None.
        model: ID of the specific AI model to use, if more than one model is available on the
            endpoint. Default value is None.
        frequency_penalty: A value that influences the probability of generated tokens appearing
            based on their cumulative frequency in generated text. Positive values will make
            tokens less likely to appear as their frequency increases and decrease the likelihood
            of the model repeating the same statements verbatim. Supported range is [-2, 2].
            Default value is None.
        presence_penalty: A value that influences the probability of generated tokens appearing
            based on their existing presence in generated text. Positive values will make tokens
            less likely to appear when they already exist and increase the model's likelihood to
            output new topics. Supported range is [-2, 2]. Default value is None.
        top_p: An alternative to sampling with temperature called nucleus sampling. This value
            causes the model to consider the results of tokens with the provided probability mass.
            As an example, a value of 0.15 will cause only the tokens comprising the top 15% of
            probability mass to be considered. Supported range is [0, 1]. Default value is None.
        seed: If specified, the system will make a best effort to sample deterministically such
            that repeated requests with the same seed and parameters should return the same result.
            Determinism is not guaranteed. Default value is None.
        stop: A collection of textual sequences that will end completions generation.
            Default value is None.
        tools: The available tool definitions that the chat completions request can use,
            including caller-defined functions. Default value is None.
        tool_choice: If specified, the model will configure which of the provided tools it can
            use for the chat completions response. Default value is None.
        response_format: The format that the AI model must output. AI chat completions models
            typically output unformatted text by default. This is equivalent to setting "text"
            as the response_format. To output JSON format, without adhering to any schema, set to
            "json_object". To output JSON format adhering to a provided schema, set this to an
            object of the class JsonSchemaFormat. Default value is None.
        model_extras: Additional, model-specific parameters that are not in the standard request
            payload. They will be added as-is to the root of the JSON in the request body.
            Default value is None.
        headers: Additional HTTP request headers to include. Default value is None.
        logging_enable: Whether to enable detailed logging for debugging. Default value is False.
    """

    # Default chat completion behavior (commonly customized)
    temperature: Optional[float]
    max_tokens: Optional[int]
    model: Optional[str]

    # Advanced completion parameters (less common but useful)
    frequency_penalty: Optional[float]
    presence_penalty: Optional[float]
    top_p: Optional[float]
    seed: Optional[int]
    stop: Optional[List[str]]

    # Tool usage (for function calling)
    tools: Optional[List[ChatCompletionsToolDefinition]]
    tool_choice: Optional[
        Union[str, ChatCompletionsToolChoicePreset, ChatCompletionsNamedToolChoice]
    ]

    # Format and model-specific options
    response_format: Optional[Union[Literal["text", "json_object"], JsonSchemaFormat]]
    model_extras: Optional[Dict[str, Any]]

    # HTTP/SDK configuration
    headers: Optional[Dict[str, str]]
    logging_enable: bool


class AzureEmbeddingsClientKwargs(TypedDict, total=False):
    """
    Keyword arguments that can be passed to the Azure EmbeddingsClient constructor.

    Only includes parameters that users would reasonably want to configure
    when using our enhanced wrapper library.

    Args:
        dimensions: The number of dimensions the resulting output embeddings should have.
            Default value is None.
        encoding_format: The desired format for the returned embeddings. Known values are:
            "base64", "binary", "float", "int8", "ubinary", and "uint8". Default value is None.
        input_type: The type of the input. Known values are: "text", "query", and "document".
            Default value is None.
        model: ID of the specific AI model to use, if more than one model is available on the
            endpoint. Default value is None.
        model_extras: Additional, model-specific parameters that are not in the standard request
            payload. They will be added as-is to the root of the JSON in the request body.
            Default value is None.
        headers: Additional HTTP request headers to include. Default value is None.
        logging_enable: Whether to enable detailed logging for debugging. Default value is False.
    """

    # Embedding-specific parameters
    dimensions: Optional[int]
    encoding_format: Optional[
        str
    ]  # Could be more specific with Literal types if needed
    input_type: Optional[str]  # Could be more specific with Literal types if needed
    model: Optional[str]
    model_extras: Optional[Dict[str, Any]]

    # HTTP/SDK configuration
    headers: Optional[Dict[str, str]]
    logging_enable: bool


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
        credential: Optional[Union[AzureKeyCredential, TokenCredential]] = None,
        api_version: str = "2024-05-01-preview",
        retry_config: Optional[RetryConfig] = None,
        connection_timeout: Optional[float] = None,
        **kwargs: Unpack[AzureChatCompletionsClientKwargs],
    ):
        """
        Initialize the enhanced ChatCompletionsClient.

        Args:
            endpoint: Azure AI endpoint URL (can be set via AZURE_AI_ENDPOINT env var)
            credential: AzureKeyCredential or TokenCredential (can be created from AZURE_AI_API_KEY env var)
            api_version: API version to use
            retry_config: Retry configuration (uses defaults if not provided)
            connection_timeout: HTTP connection timeout in seconds (default: 300)
            **kwargs: Additional arguments passed to the base Azure ChatCompletionsClient
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

        # Configure timeout if provided
        if connection_timeout is not None:
            # Azure SDK uses connection_timeout parameter
            kwargs["connection_timeout"] = connection_timeout

        # Initialize the base client
        super().__init__(
            endpoint=endpoint,
            credential=credential,
            api_version=api_version,
            **kwargs,
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
        credential: Optional[Union[AzureKeyCredential, TokenCredential]] = None,
        retry_config: Optional[RetryConfig] = None,
        connection_timeout: Optional[float] = None,
        **kwargs: Unpack[AzureEmbeddingsClientKwargs],
    ):
        """
        Initialize the enhanced EmbeddingsClient.

        Args:
            endpoint: Azure AI endpoint URL (can be set via AZURE_AI_ENDPOINT env var)
            credential: AzureKeyCredential or TokenCredential (can be created from AZURE_AI_API_KEY env var)
            retry_config: Retry configuration (uses defaults if not provided)
            connection_timeout: HTTP connection timeout in seconds (default: 300)
            **kwargs: Additional arguments passed to the base Azure EmbeddingsClient
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

        # Configure timeout if provided
        if connection_timeout is not None:
            kwargs["connection_timeout"] = connection_timeout

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
