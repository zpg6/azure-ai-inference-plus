#!/usr/bin/env python3
"""
Tests for Callback Integration

These tests verify that callbacks are properly invoked during retry scenarios.
"""

from unittest.mock import Mock, patch

import pytest

from azure_ai_inference_plus import RetryConfig
from azure_ai_inference_plus.exceptions import JSONValidationError
from azure_ai_inference_plus.utils import retry_with_config


class TestCallbackIntegration:
    """Test callback integration with retry mechanism"""

    def test_chat_retry_callback_invoked(self):
        """Test that chat retry callback is invoked on general failures"""
        callback_calls = []

        def test_callback(attempt, max_retries, exception, delay):
            callback_calls.append(
                {
                    "attempt": attempt,
                    "max_retries": max_retries,
                    "exception": exception,
                    "delay": delay,
                }
            )

        config = RetryConfig(
            max_retries=2,
            delay_seconds=0.1,  # Short delay for testing
            on_chat_retry=test_callback,
        )

        # Create a function that fails twice then succeeds
        call_count = 0

        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ConnectionError("Simulated network failure")
            return "success"

        # Apply retry decorator
        retry_function = retry_with_config(config)(failing_function)

        # Mock time.sleep to avoid actual delays
        with patch("time.sleep"):
            result = retry_function()

        assert result == "success"
        assert len(callback_calls) == 2  # Should have 2 retry attempts

        # Check first callback
        first_call = callback_calls[0]
        assert (
            first_call["attempt"] == 2
        )  # attempt numbers are 1-based, callback shows next attempt
        assert first_call["max_retries"] == 3  # max_retries + 1 for total attempts
        assert isinstance(first_call["exception"], ConnectionError)
        assert first_call["delay"] == 0.1

        # Check second callback
        second_call = callback_calls[1]
        assert second_call["attempt"] == 3
        assert second_call["max_retries"] == 3
        assert isinstance(second_call["exception"], ConnectionError)

    def test_json_retry_callback_invoked(self):
        """Test that JSON retry callback is invoked on JSON validation failures"""
        json_callback_calls = []
        chat_callback_calls = []

        def json_callback(attempt, max_retries, message):
            json_callback_calls.append(
                {"attempt": attempt, "max_retries": max_retries, "message": message}
            )

        def chat_callback(attempt, max_retries, exception, delay):
            chat_callback_calls.append(
                {
                    "attempt": attempt,
                    "max_retries": max_retries,
                    "exception": exception,
                    "delay": delay,
                }
            )

        config = RetryConfig(
            max_retries=2,
            delay_seconds=0.1,
            on_chat_retry=chat_callback,
            on_json_retry=json_callback,
        )

        # Create a function that fails with JSON validation error
        call_count = 0

        def json_failing_function():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise JSONValidationError("Invalid JSON response")
            return "success"

        # Apply retry decorator with JSON validation
        retry_function = retry_with_config(config, json_validation=False)(
            json_failing_function
        )

        # Mock time.sleep to avoid actual delays
        with patch("time.sleep"):
            result = retry_function()

        assert result == "success"
        assert len(json_callback_calls) == 2  # Should have 2 JSON retry attempts
        assert (
            len(chat_callback_calls) == 0
        )  # Chat callback should not be called for JSON errors

        # Check JSON callback calls
        first_call = json_callback_calls[0]
        assert first_call["attempt"] == 2
        assert first_call["max_retries"] == 3
        assert "Retry 2 after JSON validation failed" in first_call["message"]

    def test_no_callback_invocation_on_success(self):
        """Test that callbacks are not invoked when function succeeds immediately"""
        chat_callback_calls = []
        json_callback_calls = []

        def chat_callback(attempt, max_retries, exception, delay):
            chat_callback_calls.append(True)

        def json_callback(attempt, max_retries, message):
            json_callback_calls.append(True)

        config = RetryConfig(
            max_retries=3, on_chat_retry=chat_callback, on_json_retry=json_callback
        )

        def successful_function():
            return "immediate success"

        retry_function = retry_with_config(config)(successful_function)
        result = retry_function()

        assert result == "immediate success"
        assert len(chat_callback_calls) == 0
        assert len(json_callback_calls) == 0

    def test_callback_not_invoked_when_none(self):
        """Test that no errors occur when callbacks are None"""
        config = RetryConfig(
            max_retries=1, delay_seconds=0.1, on_chat_retry=None, on_json_retry=None
        )

        call_count = 0

        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Network error")
            return "success"

        retry_function = retry_with_config(config)(failing_function)

        with patch("time.sleep"):
            result = retry_function()

        # Should succeed without any errors despite None callbacks
        assert result == "success"

    def test_callback_exception_does_not_break_retry(self):
        """Test that exceptions in callbacks don't break the retry mechanism"""

        def broken_callback(attempt, max_retries, exception, delay):
            raise ValueError("Callback error")

        config = RetryConfig(
            max_retries=1, delay_seconds=0.1, on_chat_retry=broken_callback
        )

        call_count = 0

        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Network error")
            return "success"

        retry_function = retry_with_config(config)(failing_function)

        with patch("time.sleep"):
            # Even if callback raises an exception, retry should still work
            # Note: In real implementation, you might want to catch callback exceptions
            # This test documents current behavior
            with pytest.raises(ValueError, match="Callback error"):
                retry_function()


class TestCallbackParameters:
    """Test specific callback parameter scenarios"""

    def test_callback_receives_correct_attempt_numbers(self):
        """Test that callback receives correct attempt numbers during retries"""
        attempts_received = []

        def track_attempts(attempt, max_retries, exception, delay):
            attempts_received.append(attempt)

        config = RetryConfig(
            max_retries=3, delay_seconds=0.1, on_chat_retry=track_attempts
        )

        call_count = 0

        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count <= 3:  # Fail 3 times
                raise ConnectionError("Error")
            return "success"

        retry_function = retry_with_config(config)(failing_function)

        with patch("time.sleep"):
            result = retry_function()

        assert result == "success"
        assert attempts_received == [2, 3, 4]  # Next attempt numbers

    def test_callback_receives_correct_delays(self):
        """Test that callback receives correct delay values"""
        delays_received = []

        def track_delays(attempt, max_retries, exception, delay):
            delays_received.append(delay)

        config = RetryConfig(
            max_retries=3,
            delay_seconds=1.0,
            exponential_backoff=True,
            backoff_multiplier=2.0,
            on_chat_retry=track_delays,
        )

        call_count = 0

        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ConnectionError("Error")
            return "success"

        retry_function = retry_with_config(config)(failing_function)

        with patch("time.sleep"):
            result = retry_function()

        assert result == "success"
        assert delays_received == [1.0, 2.0]  # Exponential backoff: 1.0, 2.0


if __name__ == "__main__":
    pytest.main([__file__])
