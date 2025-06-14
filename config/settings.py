import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    """
    Configuration settings for the RAG application.
    Loads API keys and other parameters from environment variables.
    """
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "rag-index") # Default index name

    # Ensure essential keys are present
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY not found in environment variables.")
    if not PINECONE_ENVIRONMENT:
        raise ValueError("PINECONE_ENVIRONMENT not found in environment variables.")

# Create a singleton settings object
settings = Settings()

if __name__ == "__main__":
    # Example usage and verification
    print("--- Settings Loaded ---")
    print(f"Pinecone Index Name: {settings.PINECONE_INDEX_NAME}")
    print(f"OpenAI API Key (first 5 chars): {settings.OPENAI_API_KEY[:5]}*****")
    print(f"Pinecone API Key (first 5 chars): {settings.PINECONE_API_KEY[:5]}*****")
    print(f"Pinecone Environment: {settings.PINECONE_ENVIRONMENT}")
    print("-----------------------")

