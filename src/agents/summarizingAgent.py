import os
from typing import List, TypedDict,Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from llm.client import ask_groq
from retrival import RAGSearch

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END

class SummarizerState(TypedDict):
    user_input: str                     
    retrieved_context: Optional[str]    
    summary: Optional[str]     



# def validate(state:SummarizerState):



         


    
