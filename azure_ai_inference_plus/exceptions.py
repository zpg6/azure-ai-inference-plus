"""Custom exception classes for Azure AI Inference Plus"""


class AzureAIInferencePlusError(Exception):
    """Base exception class for Azure AI Inference Plus"""

    pass


class JSONValidationError(AzureAIInferencePlusError):
    """Raised when JSON response validation fails"""

    pass


class RetryExhaustedError(AzureAIInferencePlusError):
    """Raised when all retry attempts have been exhausted"""

    def __init__(self, message: str, last_exception: Exception = None):
        super().__init__(message)
        self.last_exception = last_exception


class ConfigurationError(AzureAIInferencePlusError):
    """Raised when there's a configuration error"""

    pass
