#!/usr/bin/env python3
"""
Basic usage example for Azure AI Inference Plus

This example demonstrates the key features of the enhanced ChatCompletionsClient:
- Automatic retry with exponential backoff
- JSON validation and automatic retries for JSON responses
- Reasoning separation for models like DeepSeek-R1
"""

from dotenv import load_dotenv
from azure_ai_inference_plus import (
    ChatCompletionsClient,
    RetryConfig,
    SystemMessage,
    UserMessage,
    AzureKeyCredential,
)

# Load environment variables from .env file
load_dotenv()


def main():
    """Main example function"""

    # Example 1: Basic usage
    print("=== Example 1: Basic Usage ===")

    try:
        # Create client - uses your environment variables AZURE_AI_ENDPOINT and AZURE_AI_API_KEY
        client = ChatCompletionsClient()

        response = client.complete(
            messages=[
                SystemMessage(content="You are a helpful assistant."),
                UserMessage(content="What is the capital of France?"),
            ],
            max_tokens=100,
            model="Codestral-2501",  # Replace with your model name
        )

        print(f"Response: {response.choices[0].message.content}")
        print(f"Token usage: {response.usage}")

    except Exception as e:
        print(f"Error: {e}")

    # Example 2: JSON mode with reasoning models
    print("\n=== Example 2: JSON Mode with Reasoning Models ===")
    print("📝 Important: For reasoning models (like DeepSeek-R1) in JSON mode:")
    print("   • Use generous max_tokens (1500+) - reasoning + JSON needs space")
    print("   • Provide reasoning_tags to separate thinking from JSON output")

    try:
        response = client.complete(
            messages=[
                SystemMessage(content="You are a helpful assistant that returns JSON."),
                UserMessage(
                    content="Give me information about Paris in JSON format with keys: name, country, population"
                ),
            ],
            max_tokens=2000,  # 🚨 REQUIRED: Generous tokens for reasoning models in JSON mode
            model="DeepSeek-R1",
            response_format="json_object",  # Enables automatic JSON validation & retries
            reasoning_tags=[
                "<think>",
                "</think>",
            ],  # 🚨 REQUIRED: For reasoning separation
        )

        print(f"JSON Response: {response.choices[0].message.content}")

        # Show extracted reasoning if available
        if (
            hasattr(response.choices[0].message, "reasoning")
            and response.choices[0].message.reasoning
        ):
            print(f"Extracted Reasoning: {response.choices[0].message.reasoning}")

    except Exception as e:
        print(f"Error: {e}")

    # Example 3: Custom retry configuration
    print("\n=== Example 3: Custom Retry Configuration ===")

    try:
        # Override default retry behavior
        custom_client = ChatCompletionsClient(
            retry_config=RetryConfig(
                max_retries=3, delay_seconds=1.0, exponential_backoff=True
            )
        )

        response = custom_client.complete(
            messages=[
                SystemMessage(content="You are a helpful assistant."),
                UserMessage(content="Tell me a short joke about programming."),
            ],
            max_tokens=100,
            model="Phi-4",  # Phi-4 isn't the fastest :)
        )

        print(f"Response: {response.choices[0].message.content}")

    except Exception as e:
        print(f"Error: {e}")

    # Example 4: Manual credential setup
    print("\n=== Example 4: Manual Credential Setup ===")

    try:
        # Manual client setup with credentials
        manual_client = ChatCompletionsClient(
            endpoint="https://your-resource.services.ai.azure.com/models",  # Replace with your endpoint
            credential=AzureKeyCredential(
                "your-api-key-here"
            ),  # Replace with your API key
            retry_config=RetryConfig(max_retries=2),
        )

        print(
            "✅ Client created with manual credentials (would work with real endpoint/key)"
        )
        print("✅ Alternative to environment variables for credential management")

    except Exception as e:
        print(
            f"Note: This example shows setup syntax (endpoint/key need to be real): {e}"
        )


if __name__ == "__main__":
    main()
