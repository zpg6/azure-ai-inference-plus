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

    @patch('azure_ai_inference_plus.client.AzureChatCompletionsClient.__init__')
    def test_connection_timeout_passed_to_base_client(self, mock_base_init):
        """Test that connection_timeout parameter is passed to base Azure client"""
        mock_base_init.return_value = None  # Mock __init__ to return None
        
        endpoint = "https://test.openai.azure.com"
        credential = AzureKeyCredential("test-key")
        timeout_value = 120.0

        ChatCompletionsClient(
            endpoint=endpoint, 
            credential=credential, 
            connection_timeout=timeout_value
        )

        # Verify the base client was called with connection_timeout in kwargs
        mock_base_init.assert_called_once()
        args, kwargs = mock_base_init.call_args
        
        assert 'connection_timeout' in kwargs
        assert kwargs['connection_timeout'] == timeout_value

    @patch('azure_ai_inference_plus.client.AzureChatCompletionsClient.__init__')
    def test_no_connection_timeout_when_none(self, mock_base_init):
        """Test that connection_timeout is not passed when None"""
        mock_base_init.return_value = None  # Mock __init__ to return None
        
        endpoint = "https://test.openai.azure.com"
        credential = AzureKeyCredential("test-key")

        ChatCompletionsClient(
            endpoint=endpoint, 
            credential=credential, 
            connection_timeout=None
        )

        # Verify connection_timeout is not in kwargs when None
        mock_base_init.assert_called_once()
        args, kwargs = mock_base_init.call_args
        
        assert 'connection_timeout' not in kwargs


if __name__ == "__main__":
    pytest.main([__file__])
