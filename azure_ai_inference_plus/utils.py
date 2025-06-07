"""Utility functions for Azure AI Inference Plus"""

import functools
import json
import re
import time
from typing import Any, Callable, List, Optional, Tuple, TypeVar
from urllib.parse import urljoin, urlparse

from .config import RetryConfig
from .exceptions import JSONValidationError, RetryExhaustedError

T = TypeVar("T")


def build_endpoint_url(endpoint: str) -> str:
    """
    Build a proper endpoint URL from various input formats.

    Args:
        endpoint: The endpoint URL in various formats

    Returns:
        Properly formatted endpoint URL
    """
    if not endpoint:
        raise ValueError("Endpoint cannot be empty")

    # Add https:// if no scheme provided
    if not endpoint.startswith(("http://", "https://")):
        endpoint = f"https://{endpoint}"

    # Parse URL to validate and normalize
    parsed = urlparse(endpoint)
    if not parsed.netloc:
        raise ValueError(f"Invalid endpoint URL: {endpoint}")

    # Ensure endpoint ends with proper path for models
    if not parsed.path or parsed.path == "/":
        if "models.ai.azure.com" in parsed.netloc:
            # For Azure AI Foundry endpoints, add /models path
            endpoint = urljoin(endpoint.rstrip("/"), "/models")
        elif "openai.azure.com" in parsed.netloc:
            # For Azure OpenAI endpoints, ensure proper deployments path
            if "/openai/deployments/" not in endpoint:
                endpoint = urljoin(endpoint.rstrip("/"), "/openai/deployments/")

    return endpoint


def strip_json_markdown_wrappers(content: str) -> str:
    """
    Strip markdown code block wrappers from JSON content.

    Models often wrap JSON in markdown like:
    ```json
    {"key": "value"}
    ```

    Args:
        content: The content that might have markdown wrappers

    Returns:
        Content with markdown wrappers removed
    """
    content = content.strip()

    # Remove markdown code blocks (```json...``` or ```...```)
    if content.startswith("```"):
        lines = content.split("\n")
        if len(lines) > 2:
            # Remove first line (```json or ```)
            lines = lines[1:]
            # Remove last line if it's just ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines).strip()

    return content


def validate_json_response(response_content: str) -> bool:
    """
    Validate that response content is valid JSON.
    Automatically strips common markdown wrappers before validation.

    Args:
        response_content: The response content to validate

    Returns:
        True if valid JSON, False otherwise
    """
    try:
        # First, strip any markdown wrappers
        cleaned_content = strip_json_markdown_wrappers(response_content)
        json.loads(cleaned_content)
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def parse_reasoning_from_content(
    content: str, reasoning_tags: List[str]
) -> Tuple[Optional[str], str]:
    """
    Parse reasoning content from response based on reasoning tags and remove it from content.

    Args:
        content: The response content to parse
        reasoning_tags: List with start and end tags [start_tag, end_tag]

    Returns:
        Tuple of (reasoning_content, cleaned_content)
        - reasoning_content: The extracted reasoning text (None if no reasoning found)
        - cleaned_content: Content with reasoning removed
    """
    if not reasoning_tags or len(reasoning_tags) != 2:
        return None, content

    start_tag, end_tag = reasoning_tags

    # Escape special regex characters in tags
    start_pattern = re.escape(start_tag)
    end_pattern = re.escape(end_tag)

    # Pattern to match reasoning blocks (non-greedy match)
    pattern = f"{start_pattern}(.*?){end_pattern}"

    # Find all reasoning blocks
    reasoning_matches = re.findall(pattern, content, re.DOTALL)

    if not reasoning_matches:
        return None, content

    # Extract reasoning content (join multiple blocks if present)
    reasoning_content = "\n".join(match.strip() for match in reasoning_matches)

    # Remove all reasoning blocks entirely
    cleaned_content = re.sub(pattern, "", content, flags=re.DOTALL)
    # Clean up any extra whitespace
    cleaned_content = re.sub(r"\n\s*\n", "\n", cleaned_content).strip()
    # Also strip markdown wrappers for JSON content
    cleaned_content = strip_json_markdown_wrappers(cleaned_content)

    return reasoning_content if reasoning_content else None, cleaned_content


def process_response_with_reasoning(
    response: Any, reasoning_tags: List[str], is_json_mode: bool = False
) -> Any:
    """
    Process response to separate reasoning from content.

    Args:
        response: The response object from Azure AI Inference
        reasoning_tags: List with start and end tags [start_tag, end_tag]
        is_json_mode: Whether this is JSON mode (kept for backward compatibility, but reasoning is always removed)

    Returns:
        Modified response object with reasoning field added and reasoning removed from content
    """
    if not hasattr(response, "choices") or not response.choices:
        return response

    # Process each choice
    for choice in response.choices:
        if hasattr(choice, "message") and hasattr(choice.message, "content"):
            content = choice.message.content
            if content:
                # Always remove reasoning from content when reasoning_tags are provided
                # This ensures consistent behavior between JSON and non-JSON modes
                reasoning, cleaned_content = parse_reasoning_from_content(
                    content, reasoning_tags
                )

                # Update the message content
                choice.message.content = cleaned_content

                # Add reasoning field to the message if reasoning was found
                if reasoning:
                    choice.message.reasoning = reasoning
                else:
                    choice.message.reasoning = None

    return response


def retry_with_config(
    retry_config: RetryConfig,
    json_validation: bool = False,
    reasoning_tags: Optional[List[str]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator that implements retry logic with optional JSON validation.

    Args:
        retry_config: Configuration for retry behavior
        json_validation: Whether to validate JSON responses
        reasoning_tags: Optional reasoning tags for processing before validation

    Returns:
        Decorated function with retry logic
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(
                1, retry_config.max_retries + 2
            ):  # +2 because range is exclusive and we want max_retries + 1 total attempts
                try:
                    result = func(*args, **kwargs)

                    # Validate JSON if required (on cleaned content after reasoning processing)
                    if (
                        json_validation
                        and hasattr(result, "choices")
                        and result.choices
                    ):
                        content = result.choices[0].message.content
                        if content:
                            # If reasoning tags are provided, validate on cleaned content
                            validation_content = content
                            if reasoning_tags and len(reasoning_tags) == 2:
                                _, validation_content = parse_reasoning_from_content(
                                    content, reasoning_tags
                                )

                            if not validate_json_response(validation_content):
                                raise JSONValidationError(
                                    f"Response content is not valid JSON: {validation_content[:200]}..."
                                )

                    return result

                except Exception as e:
                    last_exception = e

                    # Check if we should retry
                    if not retry_config.should_retry(e, attempt):
                        # If it's a JSON validation error on the last attempt, raise it
                        if isinstance(e, JSONValidationError):
                            raise e
                        # Otherwise, raise the original exception
                        raise e

                    # If this was the last allowed attempt, raise RetryExhaustedError
                    if attempt > retry_config.max_retries:
                        raise RetryExhaustedError(
                            f"All {retry_config.max_retries} retry attempts exhausted. "
                            f"Last error: {str(e)}",
                            last_exception=e,
                        )

                    # Wait before retrying
                    delay = retry_config.get_delay(attempt, e)

                    # Call appropriate retry callback
                    if isinstance(e, JSONValidationError):
                        # For JSON validation retries, use on_json_retry
                        if retry_config.on_json_retry:
                            retry_config.on_json_retry(
                                attempt + 1,
                                retry_config.max_retries + 1,
                                f"Retry {attempt + 1} after JSON validation failed",
                            )
                    else:
                        # For general retries, use on_chat_retry
                        if retry_config.on_chat_retry:
                            retry_config.on_chat_retry(
                                attempt + 1, retry_config.max_retries + 1, e, delay
                            )

                    time.sleep(delay)

            # This should never be reached, but just in case
            raise RetryExhaustedError(
                f"All {retry_config.max_retries} retry attempts exhausted. "
                f"Last error: {str(last_exception)}",
                last_exception=last_exception,
            )

        return wrapper

    return decorator
