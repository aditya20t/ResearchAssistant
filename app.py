import streamlit as st
from preGraph import build_pregraph, GraphState
from postGraph import build_postgraph, PostGraphState

st.title("Arxiv Search Assistant")

# -----------------------------
# Step 1: User inputs for search
# -----------------------------
query = st.text_input("Enter research topic:", "bias and stereotype published in 2023")
k = st.slider("Number of papers to fetch", 1, 10, 3)

# Initialize pregraph state once
if "pregraph_state" not in st.session_state:
    st.session_state.pregraph_state = None

if st.button("Search Papers"):
    pregraph = build_pregraph()
    initial_state = GraphState(
        user_query=query,
        arxiv_search_query="",
        arxiv_response=[],
        arxiv_max_results=k,
        arxiv_selected_id="",  # optional placeholder
        pdf_chunks=None,
        pdf_question="",
        last_answer="",
        exit_qa=False,
        loop_count=0
    )
    # Run pregraph to get arxiv results
    result_state = pregraph.invoke(initial_state)
    st.session_state.pregraph_state = result_state
    st.session_state.papers = result_state["arxiv_response"]

# -----------------------------
# Step 2: Select paper after search
# -----------------------------
if "papers" in st.session_state and st.session_state["papers"]:
    st.subheader("Select one paper to analyze")

    paper_map = {
        f"{p['title']} ({', '.join(p['authors'])})": p
        for p in st.session_state["papers"]
    }

    selected_title = st.radio(
        "Choose a paper",
        options=list(paper_map.keys()),
        key="selected_paper"
    )

    paper = paper_map[selected_title]
    with st.expander("View details"):
        st.write(f"**Authors:** {', '.join(paper['authors'])}")
        st.write(f"**Published Date:** {paper['published_date']}")
        st.write(f"**Link:** [Read on arXiv]({paper['link']})")
        st.write("**Summary:**")
        st.write(paper["summary"])

    selected_id = paper["arxiv_id"]

    # Initialize postgraph state
    if "postgraph_state" not in st.session_state:
        st.session_state.postgraph_state = PostGraphState(
            arxiv_selected_id=selected_id,
            pdf_chunks=None,
            pdf_question="",
            last_answer="",
            exit_qa=False,
            loop_count=0
        )

    postGraph = build_postgraph()

    # -----------------------------
    # Step 3: Process paper & QA loop
    # -----------------------------
    # Run postgraph starting from get_arxiv_pdf
    result_state = postGraph.invoke(st.session_state.postgraph_state, start="get_arxiv_pdf")
    st.session_state.postgraph_state = result_state

    # If exit QA pressed
    if result_state.get("exit_qa"):
        st.session_state.pop("papers", None)
        st.session_state.pop("postgraph_state", None)
        st.experimental_rerun()