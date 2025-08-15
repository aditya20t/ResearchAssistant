# app.py

import streamlit as st
import os
from preGraph import build_pregraph, GraphState
from postGraph import build_postgraph, PostGraphState, process_arxiv_pdf

st.title("Arxiv Search Assistant 📖")

# --- Sidebar for API Key Handling ---
st.sidebar.header("Configuration")

# Check if the key is already in session state or environment variables.
if 'huggingface_api_key' in st.session_state:
    os.environ['HUGGINGFACE_INFERENCE_KEY'] = st.session_state.huggingface_api_key
    api_key_provided = True
elif 'HUGGINGFACE_INFERENCE_KEY' in os.environ:
    st.session_state.huggingface_api_key = os.environ['HUGGINGFACE_INFERENCE_KEY']
    api_key_provided = True
else:
    api_key_provided = False

# Display API key status and input form in the sidebar
if api_key_provided:
    st.sidebar.success("✅ API Key is configured.")
else:
    st.sidebar.warning("This app requires a Hugging Face Inference API key to function.")
    api_key = st.sidebar.text_input(
        "Hugging Face Inference Key", 
        type="password", 
        help="Get your key from the Hugging Face website's settings page."
    )
    
    if st.sidebar.button("Submit Key"):
        if api_key:
            st.session_state.huggingface_api_key = api_key
            os.environ['HUGGINGFACE_INFERENCE_KEY'] = api_key
            st.sidebar.success("API Key accepted!")
            st.rerun()
        else:
            st.sidebar.error("Please enter a valid API key.")

# --- Main App Logic ---
# The main app will only run if the API key has been provided.
if not api_key_provided:
    st.info("Please add your Hugging Face API Key in the sidebar to begin.")
else:
    # Initialize session state variables for app flow
    if "step" not in st.session_state:
        st.session_state.step = 1
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # -----------------------------
    # Step 1: User inputs for search
    # -----------------------------
    if st.session_state.step == 1:
        st.header("1. Find Research Papers")
        with st.form("search_form"):
            query = st.text_input("Enter research topic")
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
                # Process the PDF *once* and store the chunks in session state
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
            st.write(f"**Summary:** {paper['summary']}")

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input from user
        if prompt := st.chat_input("What would you like to know?"):
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Get assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    postGraph = build_postgraph()
                    
                    # Prepare the state for the graph
                    graph_state = PostGraphState(
                        pdf_chunks=st.session_state.pdf_chunks,
                        pdf_question=prompt,
                        last_answer=""
                    )
                    
                    # Invoke the graph
                    result_state = postGraph.invoke(graph_state)
                    answer = result_state.get("last_answer", "Sorry, I couldn't find an answer.")
                    
                    st.markdown(answer)
                    # Add assistant response to history
                    st.session_state.messages.append({"role": "assistant", "content": answer})

        if st.button("Search for a Different Paper"):
            # Clear session state for QA
            keys_to_clear = ["papers", "selected_paper", "pdf_chunks", "messages"]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.step = 1
            st.rerun()
