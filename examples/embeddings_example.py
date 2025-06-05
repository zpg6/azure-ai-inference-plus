#!/usr/bin/env python3
"""
Embeddings example for Azure AI Inference Plus

This example demonstrates how to use the enhanced EmbeddingsClient
with automatic retry features.
"""

from dotenv import load_dotenv
from azure_ai_inference_plus import EmbeddingsClient, RetryConfig, AzureKeyCredential

# Load environment variables from .env file
load_dotenv()


def main():
    """Main example function for embeddings"""

    # Example 1: Basic embeddings
    print("=== Example 1: Basic Embeddings ===")

    try:
        # Create embeddings client - uses environment variables AZURE_AI_ENDPOINT and AZURE_AI_API_KEY
        client = EmbeddingsClient()

        # Example texts to embed
        texts = [
            "The quick brown fox jumps over the lazy dog",
            "Python is a great programming language",
            "Azure AI services provide powerful capabilities",
            "Machine learning models can understand text",
        ]

        print(f"Generating embeddings for {len(texts)} texts...")

        response = client.embed(
            input=texts,
            model="text-embedding-3-large",  # Replace with your embedding model
        )

        print(f"Generated {len(response.data)} embeddings")

        for i, item in enumerate(response.data):
            text = texts[item.index]
            embedding_length = len(item.embedding)
            embedding_preview = item.embedding[:5]  # First 5 dimensions

            print(f"Text {i+1}: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            print(f"  Embedding length: {embedding_length}")
            print(f"  First 5 dimensions: {embedding_preview}")
            print()

    except Exception as e:
        print(f"Error: {e}")

    # Example 2: Custom retry configuration
    print("=== Example 2: Custom Retry Configuration ===")

    try:
        # Override default retry behavior
        custom_client = EmbeddingsClient(
            retry_config=RetryConfig(
                max_retries=2, delay_seconds=1.5, exponential_backoff=True
            )
        )

        response = custom_client.embed(
            input=["Hello world"], model="text-embedding-3-large"
        )

        print(
            f"Single embedding with custom retry: {len(response.data[0].embedding)} dimensions"
        )

    except Exception as e:
        print(f"Error: {e}")

    # Example 3: Manual credential setup
    print("\n=== Example 3: Manual Credential Setup ===")

    try:
        # Manual embeddings client setup
        manual_client = EmbeddingsClient(
            endpoint="https://your-resource.services.ai.azure.com/models",  # Replace with your endpoint
            credential=AzureKeyCredential(
                "your-api-key-here"
            ),  # Replace with your API key
            retry_config=RetryConfig(max_retries=1),
        )

        print(
            "✅ Embeddings client created with manual credentials (would work with real endpoint/key)"
        )
        print("✅ Alternative to environment variables for credential management")

    except Exception as e:
        print(
            f"Note: This example shows setup syntax (endpoint/key need to be real): {e}"
        )


if __name__ == "__main__":
    main()
