#!/usr/bin/env python3
"""
Tests for reasoning functionality

These tests verify the reasoning parsing functionality in the ChatCompletionsClient.
"""

from unittest.mock import Mock, patch

import pytest

from azure_ai_inference_plus import (
    AzureKeyCredential,
    ChatCompletionsClient,
    UserMessage,
)


class TestReasoningFunctionality:
    """Test the new reasoning parsing functionality"""

    def test_reasoning_with_json_mode(self):
        """Test reasoning functionality with JSON mode (should remove reasoning)"""
        endpoint = "https://test.openai.azure.com"
        credential = AzureKeyCredential("test-key")

        client = ChatCompletionsClient(endpoint=endpoint, credential=credential)

        # Create mock response with reasoning content
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = (
            '<think>Let me format this as JSON</think>{"result": "success"}'
        )
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        with patch.object(client.__class__.__bases__[0], "complete") as mock_complete:
            mock_complete.return_value = mock_response

            result = client.complete(
                messages=[UserMessage("test")],
                model="gpt-4",
                response_format="json_object",
                reasoning_tags=["<think>", "</think>"],
            )

            # Reasoning should be removed for JSON mode
            assert result.choices[0].message.content == '{"result": "success"}'
            # Reasoning should still be accessible
            assert result.choices[0].message.reasoning == "Let me format this as JSON"

    def test_reasoning_with_non_json_mode(self):
        """Test reasoning functionality with non-JSON mode (should keep reasoning separate)"""
        endpoint = "https://test.openai.azure.com"
        credential = AzureKeyCredential("test-key")

        client = ChatCompletionsClient(endpoint=endpoint, credential=credential)

        # Create mock response with reasoning content
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "<think>Let me think about this</think>The answer is 42."
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        with patch.object(client.__class__.__bases__[0], "complete") as mock_complete:
            mock_complete.return_value = mock_response

            result = client.complete(
                messages=[UserMessage("test")],
                model="gpt-4",
                reasoning_tags=["<think>", "</think>"],
            )

            # Original content should be preserved in non-JSON mode
            assert (
                result.choices[0].message.content
                == "<think>Let me think about this</think>The answer is 42."
            )
            # Reasoning should be accessible separately
            assert result.choices[0].message.reasoning == "Let me think about this"

    def test_no_reasoning_tags(self):
        """Test that normal operation works when no reasoning_tags are provided"""
        endpoint = "https://test.openai.azure.com"
        credential = AzureKeyCredential("test-key")

        client = ChatCompletionsClient(endpoint=endpoint, credential=credential)

        # Create mock response with reasoning-like content
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "<think>This looks like reasoning but no tags configured</think>Regular response."
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        with patch.object(client.__class__.__bases__[0], "complete") as mock_complete:
            mock_complete.return_value = mock_response

            # Mock the process_response_with_reasoning function to verify it's not called
            with patch(
                "azure_ai_inference_plus.client.process_response_with_reasoning"
            ) as mock_process:
                result = client.complete(messages=[UserMessage("test")], model="gpt-4")

                # process_response_with_reasoning should not be called when no reasoning_tags
                mock_process.assert_not_called()

                # Content should be unchanged
                assert (
                    result.choices[0].message.content
                    == "<think>This looks like reasoning but no tags configured</think>Regular response."
                )


if __name__ == "__main__":
    pytest.main([__file__])
