import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Define the path to your source documents
SOURCE_DOCUMENTS_PATH = "data\source_documents"

def load_documents() -> List[Document]:
    """
    Loads documents from the specified source directory.
    Supports .txt and .pdf files. You can extend this to include more formats.

    Returns:
        A list of loaded LangChain Document objects.
    """
    documents = []
    print(f"Loading documents from: {SOURCE_DOCUMENTS_PATH}")

    # Use DirectoryLoader to automatically discover and load documents
    # You can add more loaders for different file types as needed
    loader = DirectoryLoader(
        SOURCE_DOCUMENTS_PATH,
        glob="**/*",  # Load all files
        loader_cls=None, # LangChain will try to infer loader based on extension
        # Map common extensions to specific loaders
        loader_kwargs={
            '.txt': {'loader_cls': TextLoader},
            '.pdf': {'loader_cls': PyPDFLoader},
        }
    )
    # The DirectoryLoader can be a bit tricky with specific loader_kwargs.
    # A more robust approach for multiple types might involve iterating and loading.
    
    # Let's use a more explicit approach for robustness
    for root, _, files in os.walk(SOURCE_DOCUMENTS_PATH):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            if file_name.endswith(".txt"):
                print(f"  - Loading text file: {file_name}")
                loader = TextLoader(file_path)
                documents.extend(loader.load())
            elif file_name.endswith(".pdf"):
                print(f"  - Loading PDF file: {file_name}")
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            else:
                print(f"  - Skipping unsupported file: {file_name}")

    if not documents:
        print("No documents found or loaded. Make sure files are in the 'data/source_documents' directory.")
    else:
        print(f"Successfully loaded {len(documents)} raw documents.")
    return documents

def split_documents(documents: List[Document]) -> List[Document]:
    """
    Splits a list of LangChain Document objects into smaller, more manageable chunks.
    This uses a RecursiveCharacterTextSplitter, which attempts to split
    using a list of characters in order to keep sentences and paragraphs together.

    Args:
        documents: A list of LangChain Document objects.

    Returns:
        A list of chunked LangChain Document objects.
    """
    print(f"Splitting {len(documents)} documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,       # The maximum size of each chunk
        chunk_overlap=200,     # The overlap between consecutive chunks
        length_function=len,   # Function to calculate chunk length
        add_start_index=True,  # Adds start index to metadata
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    return chunks

if __name__ == "__main__":
    # This block demonstrates how to use the functions
    # Ensure you have some files in rag-app/data/source_documents/ for testing
    print("--- Starting Document Ingestion Process ---")
    
    # 1. Load documents
    raw_documents = load_documents()

    if raw_documents:
        # 2. Split documents into chunks
        processed_chunks = split_documents(raw_documents)
        
        print("\n--- First 3 Chunks (for demonstration) ---")
        for i, chunk in enumerate(processed_chunks[:3]):
            print(f"Chunk {i+1} (Page {chunk.metadata.get('page', 'N/A')}):")
            print(f"  Content (first 200 chars): {chunk.page_content[:200]}...")
            print(f"  Metadata: {chunk.metadata}")
            print("-" * 30)
    else:
        print("No documents to process. Please add files to 'data/source_documents'.")
    
    print("--- Document Ingestion Process Finished ---")

