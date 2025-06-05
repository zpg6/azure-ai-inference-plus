#!/usr/bin/env python3
"""
Tests for parameter filtering

These tests verify the parameter filtering logic in client methods.
"""

import pytest
from unittest.mock import Mock, patch

from azure_ai_inference_plus import (
    ChatCompletionsClient, 
    EmbeddingsClient, 
    UserMessage, 
    JsonSchemaFormat,
    AzureKeyCredential
)


class TestParameterFiltering:
    """Test the improved parameter filtering logic"""
    
    def test_chat_completions_parameter_filtering(self):
        """Test that None parameters are filtered out correctly"""
        endpoint = "https://test.openai.azure.com"
        credential = AzureKeyCredential("test-key")
        
        client = ChatCompletionsClient(
            endpoint=endpoint,
            credential=credential
        )
        
        # Create a proper mock response structure
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = '{"test": "response"}'
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        
        # Mock the parent class complete method to capture the arguments
        with patch.object(client.__class__.__bases__[0], 'complete') as mock_complete:
            mock_complete.return_value = mock_response
            
            # Call with some None parameters
            client.complete(
                messages=[UserMessage("test")],
                model="gpt-4",
                max_tokens=100,
                temperature=None,  # Should be filtered out
                top_p=0.9,
                stop=None,  # Should be filtered out
                stream=False,  # Should NOT be included (only when True)
                response_format="json_object",
                tools=None,  # Should be filtered out
                user="test-user"
            )
            
            # Verify the call was made with filtered parameters
            mock_complete.assert_called_once()
            call_kwargs = mock_complete.call_args[1]
            
            # These should be present
            assert call_kwargs['messages'][0].content == "test"
            assert call_kwargs['model'] == "gpt-4"
            assert call_kwargs['max_tokens'] == 100
            assert call_kwargs['top_p'] == 0.9
            assert call_kwargs['response_format'] == "json_object"
            assert call_kwargs['user'] == "test-user"
            
            # These should be filtered out
            assert 'temperature' not in call_kwargs
            assert 'stop' not in call_kwargs
            assert 'tools' not in call_kwargs
            assert 'stream' not in call_kwargs  # Only added when True
    
    def test_chat_completions_stream_parameter(self):
        """Test that stream parameter is only added when True"""
        endpoint = "https://test.openai.azure.com"
        credential = AzureKeyCredential("test-key")
        
        client = ChatCompletionsClient(
            endpoint=endpoint,
            credential=credential
        )
        
        # Create a proper mock response structure
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = 'test response'
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        
        with patch.object(client.__class__.__bases__[0], 'complete') as mock_complete:
            mock_complete.return_value = mock_response
            
            # Test with stream=True
            client.complete(
                messages=[UserMessage("test")],
                model="gpt-4",
                stream=True
            )
            
            call_kwargs = mock_complete.call_args[1]
            assert call_kwargs['stream'] is True
    
    def test_embeddings_parameter_filtering(self):
        """Test that None parameters are filtered out correctly in EmbeddingsClient"""
        endpoint = "https://test.openai.azure.com"
        credential = AzureKeyCredential("test-key")
        
        client = EmbeddingsClient(
            endpoint=endpoint,
            credential=credential
        )
        
        # Create a proper mock response structure for embeddings
        mock_response = Mock()
        
        with patch.object(client.__class__.__bases__[0], 'embed') as mock_embed:
            mock_embed.return_value = mock_response
            
            # Call with some None parameters
            client.embed(
                input=["test text"],
                model="text-embedding-ada-002",
                encoding_format="float",
                dimensions=None,  # Should be filtered out
                user=None  # Should be filtered out
            )
            
            # Verify the call was made with filtered parameters
            mock_embed.assert_called_once()
            call_kwargs = mock_embed.call_args[1]
            
            # These should be present
            assert call_kwargs['input'] == ["test text"]
            assert call_kwargs['model'] == "text-embedding-ada-002"
            assert call_kwargs['encoding_format'] == "float"
            
            # These should be filtered out
            assert 'dimensions' not in call_kwargs
            assert 'user' not in call_kwargs
    
    def test_response_format_json_validation(self):
        """Test that JSON validation is triggered correctly with new response_format types"""
        endpoint = "https://test.openai.azure.com"
        credential = AzureKeyCredential("test-key")
        
        client = ChatCompletionsClient(
            endpoint=endpoint,
            credential=credential
        )
        
        # Create a proper mock response structure
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = '{"test": "valid json"}'
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        
        # Test with literal "json_object"
        with patch('azure_ai_inference_plus.client.retry_with_config') as mock_retry:
            mock_retry.return_value = lambda func: func
            
            with patch.object(client.__class__.__bases__[0], 'complete') as mock_complete:
                mock_complete.return_value = mock_response
                
                client.complete(
                    messages=[UserMessage("test")],
                    model="gpt-4",
                    response_format="json_object"
                )
                
                # Check that retry_with_config was called with json_validation=True
                mock_retry.assert_called_once()
                call_args = mock_retry.call_args
                assert call_args[1]['json_validation'] is True
        
        # Test with JsonSchemaFormat (should not trigger simple JSON validation)
        json_schema = JsonSchemaFormat(
            name="test_schema",
            schema={"type": "object", "properties": {"result": {"type": "string"}}}
        )
        with patch('azure_ai_inference_plus.client.retry_with_config') as mock_retry:
            mock_retry.return_value = lambda func: func
            
            with patch.object(client.__class__.__bases__[0], 'complete') as mock_complete:
                mock_complete.return_value = mock_response
                
                client.complete(
                    messages=[UserMessage("test")],
                    model="gpt-4",
                    response_format=json_schema
                )
                
                # Check that retry_with_config was called with json_validation=False (JsonSchemaFormat doesn't need our validation)
                mock_retry.assert_called_once()
                call_args = mock_retry.call_args
                assert call_args[1]['json_validation'] is False
        
        # Test with "text" (should not trigger JSON validation)
        with patch('azure_ai_inference_plus.client.retry_with_config') as mock_retry:
            mock_retry.return_value = lambda func: func
            
            with patch.object(client.__class__.__bases__[0], 'complete') as mock_complete:
                mock_complete.return_value = mock_response
                
                client.complete(
                    messages=[UserMessage("test")],
                    model="gpt-4",
                    response_format="text"
                )
                
                # Check that retry_with_config was called with json_validation=False
                mock_retry.assert_called_once()
                call_args = mock_retry.call_args
                assert call_args[1]['json_validation'] is False


if __name__ == "__main__":
    pytest.main([__file__]) 