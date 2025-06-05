#!/usr/bin/env python3
"""
Shared test configuration for azure_ai_inference_plus tests

This file contains pytest fixtures and configuration that can be used
across all test files.
"""

import pytest
import os


@pytest.fixture(autouse=True)  
def clean_environment():
    """Clean environment variables before each test to avoid interference"""
    # Store original values
    original_endpoint = os.environ.get('AZURE_AI_ENDPOINT')
    original_key = os.environ.get('AZURE_AI_API_KEY')
    
    # Remove environment variables for clean test state
    if 'AZURE_AI_ENDPOINT' in os.environ:
        del os.environ['AZURE_AI_ENDPOINT']
    if 'AZURE_AI_API_KEY' in os.environ:
        del os.environ['AZURE_AI_API_KEY']
        
    yield
    
    # Restore original values after test
    if original_endpoint is not None:
        os.environ['AZURE_AI_ENDPOINT'] = original_endpoint
    if original_key is not None:
        os.environ['AZURE_AI_API_KEY'] = original_key 