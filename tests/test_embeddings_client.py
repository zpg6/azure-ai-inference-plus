#!/usr/bin/env python3
"""
Tests for EmbeddingsClient

These tests verify the EmbeddingsClient functionality.
"""

import os
from unittest.mock import Mock, patch

import pytest

from azure_ai_inference_plus import AzureKeyCredential, EmbeddingsClient


class TestEmbeddingsClient:
    """Test the enhanced EmbeddingsClient"""

    def test_init_with_params(self):
        """Test embeddings client initialization with explicit parameters"""
        endpoint = "https://test.openai.azure.com"
        credential = AzureKeyCredential("test-key")

        client = EmbeddingsClient(endpoint=endpoint, credential=credential)

        assert client.retry_config is not None
        assert client.retry_config.max_retries == 3  # Default value

    def test_init_with_env_vars(self):
        """Test embeddings client initialization with environment variables"""
        with patch.dict(
            os.environ,
            {
                "AZURE_AI_ENDPOINT": "https://test.openai.azure.com",
                "AZURE_AI_API_KEY": "test-key",
            },
        ):
            client = EmbeddingsClient()
            assert client.retry_config is not None

    @patch('azure_ai_inference_plus.client.AzureEmbeddingsClient.__init__')
    def test_connection_timeout_passed_to_base_client(self, mock_base_init):
        """Test that connection_timeout parameter is passed to base Azure client"""
        mock_base_init.return_value = None  # Mock __init__ to return None
        
        endpoint = "https://test.openai.azure.com"
        credential = AzureKeyCredential("test-key")
        timeout_value = 90.0

        EmbeddingsClient(
            endpoint=endpoint, 
            credential=credential, 
            connection_timeout=timeout_value
        )

        # Verify the base client was called with connection_timeout in kwargs
        mock_base_init.assert_called_once()
        args, kwargs = mock_base_init.call_args
        
        assert 'connection_timeout' in kwargs
        assert kwargs['connection_timeout'] == timeout_value

    @patch('azure_ai_inference_plus.client.AzureEmbeddingsClient.__init__')
    def test_no_connection_timeout_when_none(self, mock_base_init):
        """Test that connection_timeout is not passed when None"""
        mock_base_init.return_value = None  # Mock __init__ to return None
        
        endpoint = "https://test.openai.azure.com"
        credential = AzureKeyCredential("test-key")

        EmbeddingsClient(
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
