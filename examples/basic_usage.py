#!/usr/bin/env python3
"""
Basic usage example for Azure AI Inference Plus

This example demonstrates the key features of the enhanced ChatCompletionsClient:
- Automatic retry with exponential backoff
- JSON validation and automatic retries for JSON responses
- Reasoning separation for models like DeepSeek-R1 (both JSON and non-JSON modes)
- Clean content extraction with reasoning accessible separately
"""

from dotenv import load_dotenv

from azure_ai_inference_plus import (
    AzureKeyCredential,
    ChatCompletionsClient,
    RetryConfig,
    SystemMessage,
    UserMessage,
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

    # Example 2: JSON mode with standard models
    print("\n=== Example 2: JSON Mode with Standard Models ===")

    try:
        response = client.complete(
            messages=[
                SystemMessage(content="You are a helpful assistant that returns JSON."),
                UserMessage(
                    content="Give me information about Tokyo in JSON format with keys: name, country, population, famous_for"
                ),
            ],
            max_tokens=500,
            model="gpt-4o",  # Standard model - no reasoning separation needed
            response_format="json_object",  # Enables automatic JSON validation & retries
        )

        print(f"✅ Valid JSON Response: {response.choices[0].message.content}")

        # Demonstrate that it's actually valid JSON by parsing it
        import json

        parsed_json = json.loads(response.choices[0].message.content)
        print(
            f"✅ Successfully parsed as JSON: {type(parsed_json)} with keys: {list(parsed_json.keys())}"
        )

    except Exception as e:
        print(f"Error: {e}")

    # Example 3: JSON mode with reasoning models
    print("\n=== Example 3: JSON Mode with Reasoning Models ===")
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

        print(f"✅ Valid JSON Response: {response.choices[0].message.content}")

        # Demonstrate that it's actually valid JSON by parsing it
        import json

        parsed_json = json.loads(response.choices[0].message.content)
        print(
            f"✅ Successfully parsed as JSON: {type(parsed_json)} with keys: {list(parsed_json.keys())}"
        )

        # Show extracted reasoning if available
        if (
            hasattr(response.choices[0].message, "reasoning")
            and response.choices[0].message.reasoning
        ):
            print(f"Extracted Reasoning: {response.choices[0].message.reasoning}")

    except Exception as e:
        print(f"Error: {e}")

    # Example 4: Custom retry configuration
    print("\n=== Example 4: Custom Retry Configuration ===")

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

    # Example 5: Smart timeout strategy - shorter timeout + retries
    print("\n=== Example 5: Smart Timeout Strategy ===")
    print("💡 Better: 100s timeout + retries vs default 300s timeout")

    try:
        # Recommended approach for handling timeouts
        timeout_client = ChatCompletionsClient(
            connection_timeout=100.0,  # 100s instead of default 300s
            retry_config=RetryConfig(
                max_retries=4,  # 5 total attempts
                delay_seconds=2.0,
                exponential_backoff=True,
            ),
        )

        response = timeout_client.complete(
            messages=[
                SystemMessage(content="You are a helpful assistant."),
                UserMessage(content="Explain quantum computing in detail."),
            ],
            max_tokens=800,
            model="gpt-4o",
        )

        print(f"Response: {response.choices[0].message.content[:100]}...")
        print("✅ Strategy: 5 × 100s attempts = faster failure detection than 1 × 300s")

    except Exception as e:
        print(f"Error: {e}")

    # Example 6: Reasoning separation with DeepSeek in non-JSON mode
    print("\n=== Example 6: DeepSeek Reasoning Separation (Non-JSON Mode) ===")
    print("🎯 Demonstrates: Clean reasoning extraction for better user experience")

    try:
        response = client.complete(
            messages=[
                SystemMessage(
                    content="You are a helpful assistant that thinks step by step."
                ),
                UserMessage(content="Explain why the sky appears blue during the day."),
            ],
            max_tokens=1000,
            model="DeepSeek-R1",
            reasoning_tags=[
                "<think>",
                "</think>",
            ],  # 🚨 Key: Enables reasoning separation
        )

        print(f"   Clean Content: {response.choices[0].message.content}")
        print("\n")

        # Show extracted reasoning if available
        if (
            hasattr(response.choices[0].message, "reasoning")
            and response.choices[0].message.reasoning
        ):
            print(
                f"   Extracted Reasoning: {response.choices[0].message.reasoning[:100]}..."
            )
            print(
                "\n🎯 Perfect! User sees clean answer, reasoning is accessible separately"
            )
        else:
            print("   No reasoning extracted (check model supports reasoning)")

    except Exception as e:
        print(f"Error: {e}")

    # Example 7: Manual credential setup
    print("\n=== Example 7: Manual Credential Setup ===")

    try:
        # Manual client setup with credentials
        ChatCompletionsClient(
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
