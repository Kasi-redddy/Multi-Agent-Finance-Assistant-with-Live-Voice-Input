import streamlit as st
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import threading
from pydantic import BaseModel
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import numpy as np
import os
import shutil

import whisper
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from transformers import pipeline
import pyttsx3

from audio_recorder_streamlit import audio_recorder

# ---- FastAPI Setup ----
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
class QueryRequest(BaseModel):
    text: str
    session_id: str

# ---- TTS ----
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# ---- Whisper Loader with Fresh Download ----
def load_fresh_whisper_model(model_name="base"):
    cache_dir = os.path.expanduser("~/.cache/whisper")
    model_dir = os.path.join(cache_dir, model_name)
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    return whisper.load_model(model_name)

# ---- Core Components ----
@st.cache_resource
def load_components():
    stt_model = load_fresh_whisper_model("base")
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    llm = pipeline("text-generation", model="distilgpt2")
    index_path = "financial_index"
    if not os.path.exists(index_path):
        docs = ["TSMC earnings beat expectations.", "Samsung missed earnings.", "Asia tech exposure is rising."]
        text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
        docs = text_splitter.create_documents(docs)
        db = FAISS.from_documents(docs, embeddings)
        db.save_local(index_path)
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    return stt_model, embeddings, llm, vectorstore

# ---- Agents ----
class FinancialAgents:
    def __init__(self):
        self.stt, self.embeddings, self.llm, self.vectorstore = load_components()
    def api_agent(self, tickers):
        data = yf.download(tickers, period="1d", group_by='ticker')
        return data.to_dict()
    def scraping_agent(self):
        url = "https://finance.yahoo.com/quote/TSM/"
        try:
            page = requests.get(url, timeout=10)
            soup = BeautifulSoup(page.content, 'html.parser')
            eps = soup.find("td", {"data-test": "EPS_RATIO-value"})
            pe = soup.find("td", {"data-test": "PE_RATIO-value"})
            return {
                "eps": eps.text if eps else "N/A",
                "pe": pe.text if pe else "N/A"
            }
        except Exception as e:
            return {"eps": "N/A", "pe": "N/A", "error": str(e)}
    def retriever_agent(self, query):
        try:
            results = self.vectorstore.similarity_search(query, k=3)
            return [r.page_content for r in results]
        except Exception as e:
            return [f"Retriever error: {e}"]
    def analysis_agent(self, data):
        aum = 100_000_000
        asia_tech_pct_today = 22
        asia_tech_pct_yesterday = 18
        exposure = {
            "Asia Tech": f"{asia_tech_pct_today}%",
            "Asia Tech Yesterday": f"{asia_tech_pct_yesterday}%",
            "US Tech": "35%"
        }
        return {
            "aum": aum,
            "exposure": exposure,
            "change": asia_tech_pct_today - asia_tech_pct_yesterday
        }

    def language_agent(self, context_data):
        # Unpack context_data
        market_data = context_data.get("market_data", {})
        scraped_data = context_data.get("scraped_data", {})
        analysis = context_data.get("analysis", {})
        retrieved = context_data.get("retrieved", [])

        # Format context for the LLM
        asia_tech = analysis.get("exposure", {}).get("Asia Tech", "N/A")
        asia_tech_yesterday = analysis.get("exposure", {}).get("Asia Tech Yesterday", "N/A")
        us_tech = analysis.get("exposure", {}).get("US Tech", "N/A")
        change = analysis.get("change", "N/A")
        eps = scraped_data.get("eps", "N/A")
        pe = scraped_data.get("pe", "N/A")
        tsm_close = market_data.get(("TSM", "Close"), "N/A")
        samsung_close = market_data.get(("005930.KS", "Close"), "N/A")

        summary = (
            f"Asia tech exposure: {asia_tech} (yesterday: {asia_tech_yesterday}). "
            f"US tech exposure: {us_tech}. "
            f"Change in Asia tech exposure: {change} percentage points. "
            f"TSMC close: {tsm_close}. Samsung close: {samsung_close}. "
            f"TSMC EPS: {eps}, P/E: {pe}. "
        )
        if retrieved and isinstance(retrieved, list):
            notes = "; ".join(str(r) for r in retrieved if r)
            if notes:
                summary += "Notes: " + notes

        prompt = (
            "You are a finance assistant. Summarize the following context as a morning market brief "
            "for a portfolio manager. Be concise and highlight Asia tech exposure and earnings surprises.\n"
            f"Context: {summary}\nBrief:"
        )
        result = self.llm(prompt, max_new_tokens=100, do_sample=True)
        return result[0]['generated_text']

# ---- Streamlit UI ----
def streamlit_app():
    st.title("🧑‍💼 Multi-Agent Finance Assistant with Live Voice Input")

    agents = FinancialAgents()

    st.markdown("### 🎙️ Record your question (press to record, press again to stop):")
    audio_bytes = audio_recorder()

    query = ""
    if audio_bytes:
        # Save audio to temp file
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_bytes)
        st.audio("temp_audio.wav")
        try:
            query = agents.stt.transcribe("temp_audio.wav")["text"]
            st.info(f"Transcribed voice input: {query}")
        except Exception as e:
            st.error(f"Transcription failed: {e}")
            query = ""
    # Also allow manual text input if no audio or for editing
    query = st.text_input("Or type your question here:", value=query if query else "What’s our risk exposure in Asia tech stocks today, and highlight any earnings surprises?")

    if st.button("Get Market Brief") and query:
        with st.spinner("Fetching data and generating brief..."):
            market_data = agents.api_agent(["TSM", "005930.KS"])
            scraped_data = agents.scraping_agent()
            retrieved = agents.retriever_agent(query)
            analysis = agents.analysis_agent(market_data)
            context_data = {
                "market_data": market_data,
                "scraped_data": scraped_data,
                "analysis": analysis,
                "retrieved": retrieved,
            }
            response = agents.language_agent(context_data)
            st.subheader("Market Brief")
            st.write(response)
            if st.button("🔊 Play Voice"):
                speak_text(response)

# ---- FastAPI Endpoint (not used by UI) ----
@app.post("/process")
async def process_query(request: QueryRequest, background_tasks: BackgroundTasks):
    agents = FinancialAgents()
    return {"response": f"Received: {request.text}"}

# ---- Main ----
if __name__ == "__main__":
    threading.Thread(target=uvicorn.run, args=(app,), kwargs={"port": 8000, "host": "127.0.0.1"}, daemon=True).start()
    streamlit_app()
