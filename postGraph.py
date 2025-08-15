import fitz
from io import BytesIO
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import numpy as np
from langgraph.graph import StateGraph, END
from typing import TypedDict, Any
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

class PostGraphState(TypedDict):
    arxiv_selected_id: str
    pdf_chunks: Any 
    pdf_question: str
    last_answer: str
    exit_qa: bool
    loop_count: int

def get_arxiv_pdf(state: PostGraphState) -> PostGraphState:
    arxiv_id = state.get("arxiv_selected_id", "").strip()
    if not arxiv_id:
        return state  # skip if no ID

    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
    pdf_bytes = requests.get(pdf_url).content
    pdf_stream = BytesIO(pdf_bytes)
    doc = fitz.open(stream=pdf_stream, filetype="pdf")

    full_text = ""
    for page in doc:
        full_text += page.get_text()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_text(full_text)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(chunks)

    state["pdf_chunks"] = {
        "chunks": chunks,
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix
    }
    return state

def qa_loop(state: PostGraphState) -> PostGraphState:
    st.subheader("Ask questions about the paper")
    question = st.text_input(
        "Your question:",
        key=f"qa_input_{state.get('loop_count', 0)}"
    )

    # Exit button always visible
    exit_pressed = st.button("Exit QA", key=f"exit_qa_{state.get('loop_count', 0)}")
    state["exit_qa"] = exit_pressed

    # Stop Streamlit here if no input and not exiting
    if not question and not exit_pressed:
        st.stop()

    # Only process question if provided
    if question:
        vectorizer = state["pdf_chunks"]["vectorizer"]
        tfidf_matrix = state["pdf_chunks"]["tfidf_matrix"]
        chunks = state["pdf_chunks"]["chunks"]

        query_vec = vectorizer.transform([question])
        cosine_similarities = (tfidf_matrix @ query_vec.T).toarray().ravel()

        top_idx = np.argsort(cosine_similarities)[::-1][:4]
        context = "\n\n".join(chunks[i] for i in top_idx)

        client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=os.getenv("HUGGINGFACE_INFERENCE_KEY"),
        )
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct",
            messages=[
                {"role": "system", "content": "You are a research assistant answering based only on the given paper's content."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ],
        )
        answer = completion.choices[0].message.content
        st.write("**Answer:**", answer)
        state["last_answer"] = answer

    state['loop_count'] = state.get('loop_count', 0) + 1
    return state

def counter(state: PostGraphState) -> str:
    """
    Only exit QA if user pressed Exit button.
    Otherwise, stay in QA loop.
    """
    return 'false' if state.get("exit_qa", False) else 'true'

def build_postgraph() -> StateGraph:
    graph = StateGraph(PostGraphState)

    graph.add_node('get_arxiv_pdf', get_arxiv_pdf)
    graph.add_node('qa_loop', qa_loop)

    graph.set_entry_point('get_arxiv_pdf')
    graph.add_edge('get_arxiv_pdf', 'qa_loop')

    graph.add_conditional_edges(
        'qa_loop',
        counter,
        {
            'true': 'qa_loop',   # stay in QA
            'false': END         # exit graph
        }
    )

    return graph.compile()