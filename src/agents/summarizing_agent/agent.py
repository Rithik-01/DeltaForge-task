import os
from typing import List, TypedDict,Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from src.llm.client import ask_groq
from src.retrival.search import RAGSearch

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END

class SummarizerState(TypedDict):
    user_input: str                     # user query or topic to summarize
    retrieved_context: Optional[str]    # text retrieved from vector DB
    summary: Optional[str]     

def retrieve_step(state: SummarizerState):
    query = state["user_input"]
    rag_search=RAGSearch()
    context=rag_search.search(query)
    state["retrieved_context"] = context
    return state
         


    
