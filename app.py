# app.py

import streamlit as st
from preGraph import build_pregraph, GraphState
from postGraph import build_postgraph, PostGraphState, process_arxiv_pdf

st.set_page_config(page_title="Arxiv Research Assistant", page_icon="📖", layout="wide")
st.title("Arxiv Research Assistant 📖")

# --- Sidebar for API Key Handling & Footer ---
st.sidebar.header("Configuration")

# Ensure session_state variable exists
if "huggingface_api_key" not in st.session_state:
    st.session_state.huggingface_api_key = None

# Show API key form if not provided
if st.session_state.huggingface_api_key:
    st.sidebar.success("✅ API Key is configured for this session.")
else:
    st.sidebar.warning("This app requires a Hugging Face Inference API key to function.")
    api_key = st.sidebar.text_input(
        "Hugging Face Inference Key", 
        type="password", 
        help="Get your key from the Hugging Face website's settings page.",
        autocomplete="off"
    )
    
    if st.sidebar.button("Submit Key"):
        if api_key.strip():
            st.session_state.huggingface_api_key = api_key.strip()
            st.sidebar.success("API Key accepted!")
            st.rerun()
        else:
            st.sidebar.error("Please enter a valid API key.")

# Allow user to clear API key
if st.sidebar.button("Clear API Key"):
    st.session_state.huggingface_api_key = None
    st.rerun()

# --- Footer in Sidebar ---
footer = """
<style>
.footer a {
    color: #1a73e8;
    text-decoration: none;
    font-weight: bold;
}
.footer a:hover {
    color: #d93025;
    text-decoration: underline;
}
.footer {
    margin-top: 20px;
    padding: 10px;
    text-align: center;
    font-family: 'Arial', sans-serif;
    font-size: 14px;
    color: #5f6368;
}
</style>
<div class="footer">
    <p>Developed with <span style="color:#e25555;">&#10084;</span> by 
    <a href="https://www.linkedin.com/in/aditya20t/" target="_blank">Aditya Tomar</a></p>
</div>
"""
st.sidebar.markdown(footer, unsafe_allow_html=True)

# --- Main App Logic ---
if not st.session_state.huggingface_api_key:
    st.info("Please add your Hugging Face API Key in the sidebar to begin.")
else:
    # Initialize app flow state
    if "step" not in st.session_state:
        st.session_state.step = 1
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # -----------------------------
    # Step 1: Search for papers
    # -----------------------------
    if st.session_state.step == 1:
        st.header("1. Find Research Papers")
        with st.form("search_form"):
            query = st.text_input(
                "Enter research topic", 
                autocomplete='off', 
                placeholder="e.g., RAG, machine learning"
            )
            k = st.slider("Number of papers to fetch", 1, 10, 3)
            submitted = st.form_submit_button("Search Papers")

            if submitted:
                with st.spinner("Finding papers on arXiv..."):
                    pregraph = build_pregraph()
                    initial_state = GraphState(user_query=query, arxiv_max_results=k)
                    result_state = pregraph.invoke(initial_state)
                    st.session_state.papers = result_state.get("arxiv_response", [])
                    st.session_state.step = 2
                    st.rerun()

    # -----------------------------
    # Step 2: Select a paper
    # -----------------------------
    if st.session_state.step == 2 and "papers" in st.session_state:
        st.header("2. Select a Paper to Analyze")
        papers = st.session_state.papers
        if not papers:
            st.warning("No papers found. Please try a different search query.")
            st.session_state.step = 1
            st.rerun()

        paper_map = {f"{p['title']} ({p['authors'][0]} et al.)": p for p in papers}
        selected_title = st.radio("Choose a paper:", options=list(paper_map.keys()))

        if st.button("Analyze this Paper"):
            with st.spinner("Processing PDF... This may take a moment."):
                selected_paper = paper_map[selected_title]
                st.session_state.selected_paper = selected_paper
                pdf_chunks = process_arxiv_pdf(selected_paper["arxiv_id"])
                if not pdf_chunks:
                    st.error("Failed to download or process the PDF. Please try another paper.")
                else:
                    st.session_state.pdf_chunks = pdf_chunks
                    st.session_state.step = 3
                    st.rerun()

    # -----------------------------
    # Step 3: Chat with the paper
    # -----------------------------
    if st.session_state.step == 3:
        paper = st.session_state.selected_paper
        st.header(f"3. Ask Questions About: {paper['title']}")

        with st.expander("Paper Details", expanded=False):
            st.write(f"**Authors:** {', '.join(paper['authors'])}")
            st.write(f"**Published:** {paper['published_date']}")
            st.write(f"**Link:** [Read on arXiv]({paper['link']})")
            st.write(f"**Abstract:** {paper['summary']}")

        # Chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # User input
        if prompt := st.chat_input("What would you like to know?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    postGraph = build_postgraph()
                    graph_state = PostGraphState(
                        pdf_chunks=st.session_state.pdf_chunks,
                        pdf_question=prompt,
                        last_answer=""
                    )
                    result_state = postGraph.invoke(graph_state)
                    answer = result_state.get("last_answer", "Sorry, I couldn't find an answer.")
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

        # Reset to search
        if st.button("Search for a Different Paper"):
            keys_to_clear = ["papers", "selected_paper", "pdf_chunks", "messages"]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.step = 1
            st.rerun()