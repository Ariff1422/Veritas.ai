import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sec_edgar_downloader import Downloader
import requests
import chromadb
import glob

load_dotenv()

def download_10k(ticker: str) -> str:
    headers = {"User-Agent": "VeritasAI ariffahsan@gmail.com"}
    
    response = requests.get(
        f"https://data.sec.gov/submissions/CIK{get_cik(ticker)}.json",
        headers=headers
    )
    data = response.json()
    
    filings = data["filings"]["recent"]
    cik = data["cik"]
    filing_index = None
    
    for i, form in enumerate(filings["form"]):
        if form == "10-K":
            accession = filings["accessionNumber"][i].replace("-", "")
            filing_index = i
            break
    
    if filing_index is None:
        raise ValueError(f"No 10-K found for {ticker}")
    
    index_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{filings['primaryDocument'][filing_index]}"
    doc_response = requests.get(index_url, headers=headers)
    
    filepath = f"./sec_filings/{ticker}_10k.txt"
    os.makedirs("./sec_filings", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(doc_response.text)
    
    return filepath

def get_cik(ticker: str) -> str:
    headers = {"User-Agent": "VeritasAI ariffahsan@gmail.com"}
    response = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers=headers
    )
    tickers_data = response.json()
    for key, val in tickers_data.items():
        if val["ticker"].upper() == ticker.upper():
            return str(val["cik_str"]).zfill(10)
    raise ValueError(f"CIK not found for ticker {ticker}")

def build_vectorstore(ticker: str = "AAPL"):
    print(f"Downloading 10-K for {ticker}...")
    filing_path = download_10k(ticker)
    print(f"Downloaded: {filing_path}")

    loader = TextLoader(filing_path, encoding="utf-8", autodetect_encoding=True)
    pages = loader.load()
    print(f"Loaded {len(pages)} documents")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(pages)
    
    # Cap at 800 chunks to keep indexing fast
    chunks = chunks[:800]
    print(f"Split into {len(chunks)} chunks (capped at 800)")

    client = chromadb.PersistentClient(path="./chroma_db")
    collection_name = f"{ticker.lower()}_10k"
    try:
        client.delete_collection(collection_name)
    except:
        pass
    
    collection = client.create_collection(collection_name)
    collection.add(
        documents=[chunk.page_content for chunk in chunks],
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    
    print(f"Vectorstore built for {ticker}")
    return collection

def get_retriever(query: str, ticker: str = "AAPL", n_results: int = 5) -> list:
    client = chromadb.PersistentClient(path="./chroma_db")
    collection_name = f"{ticker.lower()}_10k"
    
    try:
        collection = client.get_collection(collection_name)
    except:
        try:
            print(f"No vectorstore found for {ticker}, building one...")
            build_vectorstore(ticker)
            collection = client.get_collection(collection_name)
        except Exception as e:
            print(f"Could not retrieve 10-K for {ticker}: {e}. Skipping RAG.")
            return []
    
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    return results["documents"][0]


if __name__ == "__main__":
    build_vectorstore("AAPL")
    results = get_retriever("What are the main risk factors?", "AAPL")
    print(f"\nRetrieved {len(results)} chunks:")
    for i, chunk in enumerate(results):
        print(f"\n--- Chunk {i+1} ---")
        print(chunk[:300])