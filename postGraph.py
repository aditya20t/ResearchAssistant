# postGraph.py

import fitz
from io import BytesIO
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from langgraph.graph import StateGraph, END
from typing import TypedDict, Any
from openai import OpenAI
import os

# Define the state for the graph
class PostGraphState(TypedDict):
    pdf_chunks: Any
    pdf_question: str
    last_answer: str

# --- Standalone Utility Function ---
# This function is now called directly from Streamlit ONCE per paper.
# It is NOT a node in the QA graph.
def process_arxiv_pdf(arxiv_id: str) -> dict:
    """Downloads, chunks, and vectorizes a PDF from an arXiv ID."""
    if not arxiv_id:
        return {}

    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
    try:
        response = requests.get(pdf_url)
        response.raise_for_status() # Raise an exception for bad status codes
        pdf_bytes = response.content
    except requests.exceptions.RequestException as e:
        print(f"Error downloading PDF: {e}")
        return {}

    pdf_stream = BytesIO(pdf_bytes)
    doc = fitz.open(stream=pdf_stream, filetype="pdf")

    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_text(full_text)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(chunks)

    return {
        "chunks": chunks,
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix
    }

# --- Graph Node ---
# This is the only node needed for the QA part.
def rag_answer(state: PostGraphState) -> PostGraphState:
    """Finds relevant context and generates an answer using an LLM."""
    print("Executing RAG answer node...")
    
    # Unpack data from state
    question = state.get("pdf_question", "")
    pdf_data = state.get("pdf_chunks", {})
    
    if not all([question, pdf_data]):
        state["last_answer"] = "Error: Missing question or PDF data."
        return state

    vectorizer = pdf_data["vectorizer"]
    tfidf_matrix = pdf_data["tfidf_matrix"]
    chunks = pdf_data["chunks"]

    # Find relevant chunks
    query_vec = vectorizer.transform([question])
    cosine_similarities = (tfidf_matrix @ query_vec.T).toarray().ravel()
    top_idx = np.argsort(cosine_similarities)[::-1][:4] # Get top 4 chunks
    context = "\n\n".join(chunks[i] for i in top_idx)

    # Generate answer with LLM
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=os.getenv("HUGGINGFACE_INFERENCE_KEY"),
    )
    
    try:
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct", # Correct model name
            messages=[
                {"role": "system", "content": "You are a research assistant. Answer the user's question based ONLY on the provided context from the research paper."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ],
            stream=False # For simpler handling
        )
        answer = completion.choices[0].message.content
        state["last_answer"] = answer
    except Exception as e:
        print(f"Error calling LLM: {e}")
        state["last_answer"] = f"Sorry, I encountered an error while generating the answer: {e}"

    return state

# --- Graph Builder ---
def build_postgraph():
    """Builds the simple, single-step QA graph."""
    graph = StateGraph(PostGraphState)
    graph.add_node("rag_answer", rag_answer)
    graph.set_entry_point("rag_answer")
    graph.add_edge("rag_answer", END)
    return graph.compile()