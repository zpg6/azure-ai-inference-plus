#!/usr/bin/env python3
"""
Tests for RetryConfig

These tests verify the RetryConfig functionality.
"""

import pytest
from azure_ai_inference_plus import RetryConfig


class TestRetryConfig:
    """Test the RetryConfig class"""
    
    def test_default_config(self):
        """Test default retry configuration"""
        config = RetryConfig()
        
        assert config.max_retries == 3
        assert config.delay_seconds == 1.0
        assert config.exponential_backoff is True
        assert config.backoff_multiplier == 2.0
        assert config.max_delay == 60.0
    
    def test_should_retry_http_error(self):
        """Test retry logic for HTTP errors"""
        config = RetryConfig()
        
        # Test with a basic exception (should not retry)
        result = config.should_retry(ValueError("test error"), 1)
        assert isinstance(result, bool)
        assert result is False  # Should not retry ValueError
        
        # Test with attempt exceeding max retries
        result = config.should_retry(Exception("test"), 5)  # Exceeds max_retries=3
        assert result is False
        
        # Test ConnectionError (should retry)
        result = config.should_retry(ConnectionError("connection failed"), 1)
        assert result is True
    
    def test_get_delay_exponential(self):
        """Test exponential backoff delay calculation"""
        config = RetryConfig(
            delay_seconds=1.0,
            exponential_backoff=True,
            backoff_multiplier=2.0,
            max_delay=10.0
        )
        
        # Test with network error (should use exponential backoff)
        network_error = ConnectionError("network failed")
        assert config.get_delay(1, network_error) == 1.0
        assert config.get_delay(2, network_error) == 2.0
        assert config.get_delay(3, network_error) == 4.0
        assert config.get_delay(4, network_error) == 8.0
        assert config.get_delay(5, network_error) == 10.0  # Capped at max_delay
    
    def test_get_delay_linear(self):
        """Test linear delay calculation"""
        config = RetryConfig(
            delay_seconds=2.0,
            exponential_backoff=False
        )
        
        # Test with network error (should still be linear when exponential_backoff=False)
        network_error = ConnectionError("network failed")
        assert config.get_delay(1, network_error) == 2.0
        assert config.get_delay(2, network_error) == 2.0
        assert config.get_delay(3, network_error) == 2.0
    
    def test_get_delay_json_validation_error(self):
        """Test that JSON validation errors always use linear delay regardless of exponential_backoff setting"""
        from azure_ai_inference_plus.exceptions import JSONValidationError
        
        # Test with exponential backoff enabled
        config_exp = RetryConfig(
            delay_seconds=1.0,
            exponential_backoff=True,
            backoff_multiplier=2.0
        )
        
        json_error = JSONValidationError("invalid json")
        
        # JSON validation errors should always use linear delay
        assert config_exp.get_delay(1, json_error) == 1.0
        assert config_exp.get_delay(2, json_error) == 1.0
        assert config_exp.get_delay(3, json_error) == 1.0
        assert config_exp.get_delay(4, json_error) == 1.0
        
        # But network errors should still use exponential backoff
        network_error = ConnectionError("network failed")
        assert config_exp.get_delay(1, network_error) == 1.0
        assert config_exp.get_delay(2, network_error) == 2.0
        assert config_exp.get_delay(3, network_error) == 4.0
    
    def test_callback_initialization(self):
        """Test that callbacks can be set and retrieved"""
        def dummy_chat_callback(attempt, max_retries, exception, delay):
            pass
        
        def dummy_json_callback(attempt, max_retries, message):
            pass
        
        config = RetryConfig(
            on_chat_retry=dummy_chat_callback,
            on_json_retry=dummy_json_callback
        )
        
        assert config.on_chat_retry is dummy_chat_callback
        assert config.on_json_retry is dummy_json_callback
    
    def test_callback_defaults_to_none(self):
        """Test that callbacks default to None"""
        config = RetryConfig()
        
        assert config.on_chat_retry is None
        assert config.on_json_retry is None


class TestRetryCallbacks:
    """Test callback functionality"""
    
    def test_chat_retry_callback_signature(self):
        """Test that chat retry callback receives correct arguments"""
        callback_calls = []
        
        def test_callback(attempt, max_retries, exception, delay):
            callback_calls.append({
                'attempt': attempt,
                'max_retries': max_retries, 
                'exception': exception,
                'delay': delay
            })
        
        config = RetryConfig(on_chat_retry=test_callback)
        
        # Simulate calling the callback
        test_exception = ConnectionError("test error")
        test_delay = 1.5
        
        if config.on_chat_retry:
            config.on_chat_retry(1, 3, test_exception, test_delay)
        
        assert len(callback_calls) == 1
        call = callback_calls[0]
        assert call['attempt'] == 1
        assert call['max_retries'] == 3
        assert call['exception'] is test_exception
        assert call['delay'] == 1.5
    
    def test_json_retry_callback_signature(self):
        """Test that JSON retry callback receives correct arguments"""
        callback_calls = []
        
        def test_callback(attempt, max_retries, message):
            callback_calls.append({
                'attempt': attempt,
                'max_retries': max_retries,
                'message': message
            })
        
        config = RetryConfig(on_json_retry=test_callback)
        
        # Simulate calling the callback
        test_message = "JSON validation failed"
        
        if config.on_json_retry:
            config.on_json_retry(2, 3, test_message)
        
        assert len(callback_calls) == 1
        call = callback_calls[0]
        assert call['attempt'] == 2
        assert call['max_retries'] == 3
        assert call['message'] == test_message
    
    def test_callbacks_are_optional(self):
        """Test that having None callbacks doesn't break anything"""
        config = RetryConfig(
            on_chat_retry=None,
            on_json_retry=None
        )
        
        # These should not raise exceptions
        if config.on_chat_retry:
            config.on_chat_retry(1, 3, Exception("test"), 1.0)
        
        if config.on_json_retry:
            config.on_json_retry(1, 3, "test message")
        
        # No assertions needed - we just want to ensure no exceptions


if __name__ == "__main__":
    pytest.main([__file__]) 