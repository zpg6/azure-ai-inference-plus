"""Configuration classes for Azure AI Inference Plus"""

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class RetryConfig:
    """Configuration for retry behavior"""

    max_retries: int = 3
    delay_seconds: float = 1.0
    exponential_backoff: bool = True
    backoff_multiplier: float = 2.0
    max_delay: Optional[float] = 60.0
    retry_on_status_codes: tuple = (429, 500, 502, 503, 504)
    retry_condition: Optional[Callable[[Exception], bool]] = None

    # Callback functions for retry events
    on_chat_retry: Optional[Callable[[int, int, Exception, float], None]] = (
        None  # (attempt, max_retries, exception, delay)
    )
    on_json_retry: Optional[Callable[[int, int, str], None]] = (
        None  # (attempt, max_retries, message)
    )

    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """
        Determine if an exception should trigger a retry.

        Args:
            exception: The exception that occurred
            attempt: Current attempt number (1-based)

        Returns:
            True if retry should be attempted, False otherwise
        """
        if attempt > self.max_retries:
            return False

        # Use custom retry condition if provided
        if self.retry_condition:
            return self.retry_condition(exception)

        # Default retry logic for HTTP errors
        from azure.core.exceptions import HttpResponseError

        if isinstance(exception, HttpResponseError):
            return exception.status_code in self.retry_on_status_codes

        # Retry on JSON validation errors (common with JSON mode)
        from .exceptions import JSONValidationError

        if isinstance(exception, JSONValidationError):
            return True

        # Retry on common transient errors
        transient_errors = (
            ConnectionError,
            TimeoutError,
        )

        # Also retry on Azure ServiceResponseError which includes timeout errors
        from azure.core.exceptions import ServiceResponseError

        if isinstance(exception, ServiceResponseError):
            # Check if it's a timeout error
            if (
                "timeout" in str(exception).lower()
                or "timed out" in str(exception).lower()
            ):
                return True

        return isinstance(exception, transient_errors)

    def get_delay(self, attempt: int, exception: Exception = None) -> float:
        """
        Calculate delay for the given attempt number.

        Args:
            attempt: Current attempt number (1-based)
            exception: The exception that triggered the retry (optional)

        Returns:
            Delay in seconds
        """
        # For JSON validation errors, always use linear delay (no exponential backoff)
        from .exceptions import JSONValidationError

        if isinstance(exception, JSONValidationError):
            return self.delay_seconds

        # For other errors, use the configured backoff strategy
        if not self.exponential_backoff:
            return self.delay_seconds

        delay = self.delay_seconds * (self.backoff_multiplier ** (attempt - 1))

        if self.max_delay:
            delay = min(delay, self.max_delay)

        return delay
