#!/usr/bin/env python3
"""
Tests for EmbeddingsClient

These tests verify the EmbeddingsClient functionality.
"""

import os
import pytest
from unittest.mock import patch

from azure_ai_inference_plus import EmbeddingsClient, AzureKeyCredential


class TestEmbeddingsClient:
    """Test the enhanced EmbeddingsClient"""
    
    def test_init_with_params(self):
        """Test embeddings client initialization with explicit parameters"""
        endpoint = "https://test.openai.azure.com"
        credential = AzureKeyCredential("test-key")
        
        client = EmbeddingsClient(
            endpoint=endpoint,
            credential=credential
        )
        
        assert client.retry_config is not None
        assert client.retry_config.max_retries == 3  # Default value
    
    def test_init_with_env_vars(self):
        """Test embeddings client initialization with environment variables"""
        with patch.dict(os.environ, {
            'AZURE_AI_ENDPOINT': 'https://test.openai.azure.com',
            'AZURE_AI_API_KEY': 'test-key'
        }):
            client = EmbeddingsClient()
            assert client.retry_config is not None


if __name__ == "__main__":
    pytest.main([__file__]) 