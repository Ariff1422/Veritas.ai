import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb

load_dotenv()

# ---- INDEXING (run once to build the vector store) ----

def build_vectorstore(pdf_path: str = "apple_10k.pdf"):
    # 1. Load the PDF
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"Loaded {len(pages)} pages")

    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(pages)
    print(f"Split into {len(chunks)} chunks")

    # 3. Store in ChromaDB using its built-in embeddings
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Delete existing collection if rebuilding
    try:
        client.delete_collection("apple_10k")
    except:
        pass
    
    collection = client.create_collection("apple_10k")
    
    # Add chunks to collection
    collection.add(
        documents=[chunk.page_content for chunk in chunks],
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    
    print(f"Vectorstore built and saved to ./chroma_db")
    return collection

# ---- RETRIEVAL (used at runtime by the RAG node) ----

def get_retriever(query: str, n_results: int = 5) -> list:
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("apple_10k")
    
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    return results["documents"][0]  # list of relevant chunks


if __name__ == "__main__":
    build_vectorstore()
    
    # Test retrieval
    results = get_retriever("What are Apple's main risk factors?")
    print(f"\nRetrieved {len(results)} chunks:")
    for i, chunk in enumerate(results):
        print(f"\n--- Chunk {i+1} ---")
        print(chunk[:300])