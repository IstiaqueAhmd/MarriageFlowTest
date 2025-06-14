import os
import sys
from typing import List
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain.schema import Document
from pinecone import Pinecone as PineconeClient, PodSpec # Import PineconeClient and PodSpec directly
# This allows imports like 'config.settings' to work correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from config.settings import settings # Import your settings

def initialize_pinecone():
    """
    Initializes the Pinecone client using API key and environment from settings.
    Creates the index if it doesn't already exist.
    """
    print("Initializing Pinecone...")
    # Initialize the Pinecone client
    PineconeClient(api_key=settings.PINECONE_API_KEY, environment=settings.PINECONE_ENVIRONMENT)

    # Check if the index exists, and create it if not
    # This example assumes a serverless index, adjust spec for pod-based
    if settings.PINECONE_INDEX_NAME not in PineconeClient().list_indexes():
        print(f"Creating Pinecone index: {settings.PINECONE_INDEX_NAME}...")
        PineconeClient().create_index(
            name=settings.PINECONE_INDEX_NAME,
            dimension=1536,  # OpenAI's text-embedding-ada-002 model dimension
            metric="cosine", # Cosine similarity is common for embeddings
            spec=PodSpec(environment=settings.PINECONE_ENVIRONMENT) # For Pod-based; use ServerlessSpec() for serverless
        )
        print(f"Pinecone index '{settings.PINECONE_INDEX_NAME}' created.")
    else:
        print(f"Pinecone index '{settings.PINECONE_INDEX_NAME}' already exists.")

def embed_and_store_documents(chunks: List[Document]):
    """
    Generates embeddings for the provided document chunks using OpenAI
    and stores them in the Pinecone vector database.

    Args:
        chunks: A list of LangChain Document objects (chunks).
    """
    if not chunks:
        print("No chunks provided to embed and store. Exiting.")
        return

    print(f"Generating embeddings and storing {len(chunks)} chunks in Pinecone...")

    # Initialize OpenAI embeddings
    # Ensure OPENAI_API_KEY is set in your environment variables
    embeddings = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)

    # Connect to the Pinecone index and add documents
    # from_documents handles embedding and upserting in one step
    try:
        # If your index is new and empty, use from_documents directly
        # If adding to an existing index, ensure you pass the index name and embeddings
        vectorstore = Pinecone.from_documents(
            documents=chunks,
            embedding=embeddings,
            index_name=settings.PINECONE_INDEX_NAME,
        )
        print(f"Successfully embedded and stored {len(chunks)} chunks in Pinecone index '{settings.PINECONE_INDEX_NAME}'.")

    except Exception as e:
        print(f"An error occurred during embedding and storing: {e}")
        # You might want to add more specific error handling here
        # e.g., for API key issues, network errors, etc.

if __name__ == "__main__":
    # This block demonstrates how to use the functions.
    # In a real application, you would call this after load_and_split.py
    
    # For demonstration, let's create some dummy chunks
    from load_and_split import load_documents, split_documents
    
    print("--- Starting Embedding and Storage Process ---")
    
    # 1. Load and split documents (re-using functions from load_and_split.py)
    # Ensure you have files in rag-app/data/source_documents/
    raw_documents = load_documents()
    if raw_documents:
        chunks_to_process = split_documents(raw_documents)
        
        # 2. Initialize Pinecone (creates index if not exists)
        initialize_pinecone()

        # 3. Embed and store the chunks
        embed_and_store_documents(chunks_to_process)
    else:
        print("No documents loaded to embed. Please ensure 'data/source_documents' contains files.")
        
    print("--- Embedding and Storage Process Finished ---")

