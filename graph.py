import yfinance as yf
import requests
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict

load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

class State(TypedDict):
    ticker: str
    financial_data: str
    news_data: str
    rag_context: str
    report: str

def research_node(state: State) -> dict:
    ticker = state["ticker"]
    
    # yfinance data
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
    
    # NewsAPI data
    url = f"https://newsapi.org/v2/everything?q={ticker}&sortBy=publishedAt&pageSize=5&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    articles = response.json().get("articles", [])
    news_summary = "\n".join([f"- {a['title']}" for a in articles])
    
    return {
        "financial_data": financial_summary,
        "news_data": news_summary
    }

# Build the graph (just research node for now)
builder = StateGraph(State)
builder.add_node("research", research_node)
builder.set_entry_point("research")
builder.add_edge("research", END)
graph = builder.compile()

# Test it
if __name__ == "__main__":
    result = graph.invoke({
        "ticker": "AAPL",
        "financial_data": "",
        "news_data": "",
        "rag_context": "",
        "report": ""
    })
    print("=== FINANCIAL DATA ===")
    print(result["financial_data"])
    print("=== NEWS ===")
    print(result["news_data"])