from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any
from arxiv_search import search_arxiv
import os
from openai import OpenAI
import streamlit as st
import numpy as np

class GraphState(TypedDict):
    user_query: str
    arxiv_search_query: str
    arxiv_max_results: int
    arxiv_response: List[Dict[str, Any]]
    arxiv_selected_id: str
    pdf_chunks: Any 
    pdf_question: str
    last_answer: str
    exit_qa: bool
    loop_count: int

def user_query(state: GraphState) -> GraphState:
    return state

def build_arxiv_query(state: GraphState) -> GraphState:
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=st.session_state.huggingface_api_key,
    )

    completion = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct",
        messages=[
                {   "role": "system",
                    "content": "You are a helpful assistant that converts user queries into search queries for the arxiv API."
                },
                {
                    "role": "user",
                    "content": f"""
                    Convert the user query into a search query for arxiv API. Arxiv API query is of the form:
                    http://export.arxiv.org/api/query?search_query=<search_query>&start=0&max_results=<max_results>
                    where <search_query> is the query to search. Return only the search query part.
                    Don't return anything else apart from the search query.

                    You can search specific fields using a prefix. The available field prefixes are:
                    ti for Title
                    au for Author
                    abs for Abstract
                    co for Comment
                    jr for Journal Reference
                    cat for Subject Category
                    rn for Report Number
                    id for Id
                    all for all of the above fields.

                    Additionally, the API provides one date filter, submittedDate, that allows you to select data within a given date range of when the paper was submitted. The expected format is [YYYYMMDDTTTT+TO+YYYYMMDDTTTT] where the TTTT is the time provided in 24 hour format to the minute, in GMT.

                    Here are some examples of how to construct a query:

                    User query: we wanted to find all articles by the author Adrian Del Maestro.
                    Search query: au:del_maestro

                    User query: I'm looking for papers on "language models" in the computer science category.
                    Search query: ti:"language models"+AND+cat:cs

                    User query: Find papers by Del Maestro submitted between Jan 1, 2023 6:00 AM and Jan 1, 2024 6:00 AM GMT.
                    Search query: au:del_maestro+AND+submittedDate:[202301010600+TO+202401010600]
                                            
                    User query: {state['user_query']}
                    Search query:
                    """
                }
        ],
    )

    state['arxiv_search_query'] = completion.choices[0].message.content.strip()
    return state

def search_on_arxiv(state):
    query = state['arxiv_search_query']
    max_results = state['arxiv_max_results']
    response = search_arxiv(query, max_results)
    # Assuming response is a list of dictionaries with arxiv data
    state['arxiv_response'] = response
    return state


# Build the state graph
def build_pregraph():
    graph = StateGraph(GraphState)

    graph.add_node('user_query', user_query)
    graph.add_node('build_arxiv_query', build_arxiv_query)
    graph.add_node('search_on_arxiv', search_on_arxiv)

    graph.set_entry_point('user_query')
    graph.add_edge('user_query', 'build_arxiv_query')
    graph.add_edge('build_arxiv_query', 'search_on_arxiv')
    graph.add_edge('search_on_arxiv', END)

    return graph.compile()

        