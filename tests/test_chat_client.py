#!/usr/bin/env python3
"""
Tests for ChatCompletionsClient

These tests verify the ChatCompletionsClient functionality.
"""

import os
from unittest.mock import Mock, patch

import pytest

from azure_ai_inference_plus import (
    AzureKeyCredential,
    ChatCompletionsClient,
    ConfigurationError,
    RetryConfig,
    UserMessage,
)


class TestChatCompletionsClient:
    """Test the enhanced ChatCompletionsClient"""

    def test_init_with_params(self):
        """Test client initialization with explicit parameters"""
        endpoint = "https://test.openai.azure.com"
        credential = AzureKeyCredential("test-key")

        client = ChatCompletionsClient(endpoint=endpoint, credential=credential)

        assert client.retry_config is not None
        assert client.retry_config.max_retries == 3  # Default value

    def test_init_with_env_vars(self):
        """Test client initialization with environment variables"""
        with patch.dict(
            os.environ,
            {
                "AZURE_AI_ENDPOINT": "https://test.openai.azure.com",
                "AZURE_AI_API_KEY": "test-key",
            },
        ):
            client = ChatCompletionsClient()
            assert client.retry_config is not None

    def test_init_missing_endpoint(self):
        """Test that missing endpoint raises ConfigurationError"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ConfigurationError, match="Endpoint must be provided"):
                ChatCompletionsClient()

    def test_init_missing_credential(self):
        """Test that missing credential raises ConfigurationError"""
        with patch.dict(
            os.environ, {"AZURE_AI_ENDPOINT": "https://test.openai.azure.com"}
        ):
            with pytest.raises(ConfigurationError, match="Credential must be provided"):
                ChatCompletionsClient()

    def test_custom_retry_config(self):
        """Test client with custom retry configuration"""
        endpoint = "https://test.openai.azure.com"
        credential = AzureKeyCredential("test-key")

        custom_config = RetryConfig(max_retries=5, delay_seconds=2.0)

        client = ChatCompletionsClient(
            endpoint=endpoint, credential=credential, retry_config=custom_config
        )

        assert client.retry_config.max_retries == 5
        assert client.retry_config.delay_seconds == 2.0


if __name__ == "__main__":
    pytest.main([__file__])
