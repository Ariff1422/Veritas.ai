# Veritas.ai — Multi-Agent Credit Assessment Tool

An AI-powered financial research assistant that generates structured credit assessment reports using a LangGraph multi-agent pipeline.

## Demo

Enter any stock ticker (e.g. AAPL, MSFT, TSLA) and the pipeline automatically produces a full credit assessment report with financial analysis, risk factors, and a recommendation.

## Architecture

Three agents orchestrated by LangGraph:

1. **Research Agent** — fetches live financial data (yfinance) and recent news (NewsAPI)
2. **RAG Agent** — retrieves relevant excerpts from 10-K filings using ChromaDB vector search
3. **Report Agent** — synthesises all data using Claude Haiku to draft a structured credit assessment

## Tech Stack

- **Orchestration:** LangGraph, LangChain
- **LLM:** Claude Haiku (Anthropic)
- **Vector Database:** ChromaDB
- **Embeddings:** ChromaDB built-in (all-MiniLM-L6-v2)
- **Frontend:** Streamlit
- **Data Sources:** yfinance, NewsAPI, SEC 10-K filings

## Setup

1. Clone the repo and create a virtual environment:

```bash
git clone https://github.com/Ariff1422/Veritas.ai.git
cd Veritas.ai
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Create a `.env` file:

```
ANTHROPIC_API_KEY=your_key
NEWS_API_KEY=your_key
```

3. Add a 10-K PDF named `apple_10k.pdf` to the project root and build the vector store:

```bash
python3 rag.py
```

4. Run the app:

```bash
streamlit run app.py
```

## CI/CD

GitHub Actions runs on every push and PR — installs dependencies and verifies the LangGraph pipeline compiles correctly.
