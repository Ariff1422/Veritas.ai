import streamlit as st
from graph import graph

st.set_page_config(page_title="Veritas.ai", layout="wide")

st.title("Veritas.ai — Credit Assessment Agent")
st.caption("Multi-agent AI pipeline for financial credit analysis")

ticker = st.text_input("Enter a stock ticker", placeholder="e.g. AAPL, MSFT, TSLA")

if st.button("Generate Report"):
    if not ticker:
        st.warning("Please enter a ticker symbol")
    else:
        with st.spinner("Running analysis pipeline..."):
            col1, col2, col3 = st.columns(3)
            
            with st.status("Running agents...", expanded=True) as status:
                st.write("🔍 Research Agent — fetching financial data and news...")
                st.write("📚 RAG Agent — retrieving relevant 10-K excerpts...")
                st.write("📝 Report Agent — generating credit assessment...")
                
                result = graph.invoke({
                    "ticker": ticker.upper(),
                    "financial_data": "",
                    "news_data": "",
                    "rag_context": "",
                    "report": ""
                })
                status.update(label="Analysis complete!", state="complete")
        
        st.divider()
        st.markdown(result["report"])
        
        st.download_button(
            label="Download Report",
            data=result["report"],
            file_name=f"{ticker.upper()}_credit_assessment.md",
            mime="text/markdown"
        )