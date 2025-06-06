"""
Example: Retry Callbacks

Shows how to use callbacks to monitor retry attempts for logging and debugging.
This example demonstrates callbacks in action with real requests.
"""

from dotenv import load_dotenv

from azure_ai_inference_plus import (
    ChatCompletionsClient,
    RetryConfig,
    SystemMessage,
    UserMessage,
)

# Load environment variables from .env file
load_dotenv()


def on_chat_retry(attempt, max_retries, exception, delay):
    """Called when chat operations retry"""
    print(
        f"🔄 Chat retry {attempt}/{max_retries}: {type(exception).__name__} (waiting {delay:.1f}s)"
    )
    print(f"  └─ Callback triggered! Exception: {exception}")


def on_json_retry(attempt, max_retries, message):
    """Called when JSON validation fails and retries"""
    print(f"📝 JSON retry {attempt}/{max_retries}: {message}")
    print(f"  └─ JSON callback triggered! {message}")


def main():
    print("=== Retry Callbacks Example ===")
    print("This will demonstrate callbacks in action with real requests")

    # Create client with callbacks and more aggressive retry settings
    client = ChatCompletionsClient(
        retry_config=RetryConfig(
            max_retries=3,
            delay_seconds=0.5,  # Shorter delays for demo
            on_chat_retry=on_chat_retry,
            on_json_retry=on_json_retry,
        )
    )

    print("✅ Client created with retry callbacks enabled")

    # Test 1: Normal request (should work without retries)
    print("\n=== Test 1: Normal Request (No Retries Expected) ===")
    try:
        response = client.complete(
            messages=[
                SystemMessage(content="You are a helpful assistant."),
                UserMessage(content="Say 'Hello World' in exactly 2 words."),
            ],
            model="Codestral-2501",
            max_tokens=10,
        )
        print(f"✅ Response: {response.choices[0].message.content}")
        print("   No callbacks triggered (success on first try)")
    except Exception as e:
        print(f"❌ Error: {e}")

    # Test 2: JSON mode with reasoning model (might trigger JSON retries)
    print("\n=== Test 2: JSON Mode (May Trigger JSON Retries) ===")
    print(
        "Using reasoning model with restrictive token limit to potentially trigger JSON validation retries"
    )
    try:
        response = client.complete(
            messages=[
                SystemMessage(content="Return valid JSON only. Be very concise."),
                UserMessage(
                    content="Give me basic info about cats as JSON with keys: type, cute"
                ),
            ],
            model="DeepSeek-R1",  # Reasoning model
            response_format="json_object",
            max_tokens=150,  # Restrictive - might cause incomplete JSON initially
            reasoning_tags=["<think>", "</think>"],
        )
        print(f"✅ JSON Response: {response.choices[0].message.content}")

        # Show reasoning if available
        if (
            hasattr(response.choices[0].message, "reasoning")
            and response.choices[0].message.reasoning
        ):
            print(
                f"💭 Reasoning (first 100 chars): {response.choices[0].message.reasoning[:100]}..."
            )

    except Exception as e:
        print(f"❌ Error: {e}")

    # Test 3: Another JSON attempt with even more restrictive settings
    print("\n=== Test 3: More Aggressive JSON Test ===")
    try:
        response = client.complete(
            messages=[
                SystemMessage(
                    content="You must return valid JSON. Think step by step but be concise."
                ),
                UserMessage(
                    content="JSON object with keys: animal, sound, legs for a dog"
                ),
            ],
            model="DeepSeek-R1",
            response_format="json_object",
            max_tokens=100,  # Very restrictive
            reasoning_tags=["<think>", "</think>"],
        )
        print(f"✅ Final JSON: {response.choices[0].message.content}")
    except Exception as e:
        print(f"❌ Error: {e}")

    print("\n=== Summary ===")
    print("📊 Callbacks provide visibility into:")
    print("   • When retries happen and why")
    print("   • How long delays are between attempts")
    print("   • Which exceptions triggered retries")
    print("   • JSON validation failures and recovery")
    print("\n💡 Integrate callbacks with your logging/monitoring systems!")


if __name__ == "__main__":
    main()
