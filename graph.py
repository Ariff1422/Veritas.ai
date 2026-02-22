import os
import yfinance as yf
import requests
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from typing import TypedDict
from rag import get_retriever

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

# ---- STATE ----

class State(TypedDict):
    ticker: str
    financial_data: str
    news_data: str
    rag_context: str
    report: str

# ---- NODE 1: RESEARCH ----

def research_node(state: State) -> dict:
    ticker = state["ticker"]

    stock = yf.Ticker(ticker)
    info = stock.info
    financial_summary = f"""
    Company: {info.get('longName', 'N/A')}
    Sector: {info.get('sector', 'N/A')}
    Market Cap: {info.get('marketCap', 'N/A')}
    P/E Ratio: {info.get('trailingPE', 'N/A')}
    Revenue: {info.get('totalRevenue', 'N/A')}
    Debt to Equity: {info.get('debtToEquity', 'N/A')}
    Current Ratio: {info.get('currentRatio', 'N/A')}
    """

    url = f"https://newsapi.org/v2/everything?q={ticker}&sortBy=publishedAt&pageSize=5&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    articles = response.json().get("articles", [])
    news_summary = "\n".join([f"- {a['title']}" for a in articles])

    print("✓ Research node complete")
    return {
        "financial_data": financial_summary,
        "news_data": news_summary
    }

# ---- NODE 2: RAG ----

def rag_node(state: State) -> dict:
    ticker = state["ticker"]

    query = f"{ticker} risk factors financial performance debt revenue"
    chunks = get_retriever(query, n_results=5)
    rag_context = "\n\n".join(chunks)

    print("✓ RAG node complete")
    return {"rag_context": rag_context}

# ---- NODE 3: REPORT ----

def report_node(state: State) -> dict:
    prompt = f"""
    You are a senior credit analyst. Using the data below, write a structured credit assessment report.

    ## Live Financial Data
    {state["financial_data"]}

    ## Recent News
    {state["news_data"]}

    ## Relevant 10-K Excerpts
    {state["rag_context"]}

    Write the report with these sections:
    1. Company Overview
    2. Key Financial Metrics & Analysis
    3. Risk Factors
    4. Recent Developments
    5. Credit Assessment Summary & Recommendation
    """

    response = llm.invoke(prompt)
    print("✓ Report node complete")
    return {"report": response.content}

# ---- BUILD GRAPH ----

builder = StateGraph(State)
builder.add_node("research", research_node)
builder.add_node("rag", rag_node)
builder.add_node("report", report_node)

builder.set_entry_point("research")
builder.add_edge("research", "rag")
builder.add_edge("rag", "report")
builder.add_edge("report", END)

graph = builder.compile()

# ---- RUN ----

if __name__ == "__main__":
    result = graph.invoke({
        "ticker": "AAPL",
        "financial_data": "",
        "news_data": "",
        "rag_context": "",
        "report": ""
    })
    print("\n========== CREDIT ASSESSMENT REPORT ==========\n")
    print(result["report"])