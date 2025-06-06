#!/usr/bin/env python3
"""
Tests for utility functions

These tests verify the utility functions in the azure_ai_inference_plus package.
"""

import pytest


class TestUtils:
    """Test utility functions"""

    def test_build_endpoint_url(self):
        """Test endpoint URL building"""
        from azure_ai_inference_plus.utils import build_endpoint_url

        # Test with HTTPS
        result = build_endpoint_url("https://test.openai.azure.com")
        assert result.startswith("https://")

        # Test without scheme
        result = build_endpoint_url("test.openai.azure.com")
        assert result.startswith("https://")

        # Test Azure AI Foundry endpoint
        result = build_endpoint_url("https://test.models.ai.azure.com")
        assert "/models" in result

    def test_validate_json_response(self):
        """Test JSON response validation"""
        from azure_ai_inference_plus.utils import validate_json_response

        # Valid JSON
        assert validate_json_response('{"key": "value"}') is True
        assert validate_json_response("[]") is True
        assert validate_json_response("null") is True

        # Invalid JSON
        assert validate_json_response("invalid json") is False
        assert validate_json_response('{"key": invalid}') is False
        assert validate_json_response("") is False

    def test_parse_reasoning_from_content(self):
        """Test reasoning parsing utility function"""
        from azure_ai_inference_plus.utils import parse_reasoning_from_content

        # Test content with reasoning tags
        content = "Let me think. <think>This is reasoning content</think> Here is the final answer."
        reasoning, cleaned = parse_reasoning_from_content(
            content, ["<think>", "</think>"], remove_reasoning=True
        )

        assert reasoning == "This is reasoning content"
        assert cleaned == "Let me think.  Here is the final answer."

        # Test content without reasoning tags
        content = "Just a regular response without reasoning."
        reasoning, cleaned = parse_reasoning_from_content(
            content, ["<think>", "</think>"], remove_reasoning=True
        )

        assert reasoning is None
        assert cleaned == "Just a regular response without reasoning."

        # Test with multiple reasoning blocks
        content = "<think>First thought</think> Some text <think>Second thought</think> Final text"
        reasoning, cleaned = parse_reasoning_from_content(
            content, ["<think>", "</think>"], remove_reasoning=True
        )

        assert reasoning == "First thought\nSecond thought"
        assert "Some text" in cleaned
        assert "Final text" in cleaned
        assert "<think>" not in cleaned


if __name__ == "__main__":
    pytest.main([__file__])
