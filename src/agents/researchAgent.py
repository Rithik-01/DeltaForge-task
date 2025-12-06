from typing import TypedDict, Sequence,Optional,List
from src.llm import ask_groq,ask_gemini
from src.retrival import RAGSearch
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition

class AgentState(TypedDict):
    user_input: str
    rewritten_query: Optional[str]
    retrieved_docs: List[str]
    validated_docs:List[str]
    explanation:str
    

def retrieve_step(state: AgentState):
    if state["rewritten_query"] is None:
        query = state["user_input"]
    else:
        query = state["rewritten_query"]
    rag_search=RAGSearch()
    context=rag_search.search(query,top_k=10)
    state["retrieved_docs"] = context
    # print(f"[INFO] retrived the content for the query :{query}")
    return state

def validate(state:AgentState):
    
    query = state["user_input"]
    context=state["retrieved_docs"]
    prompt=f"""Your are an validator Expert. User question: {query} 
            and the retrived context :{context}
            now validate for any llm can answer the question using this context.
            NOTE:You should return Only : Yes or No """
    result = ask_groq(prompt=prompt)

    print(f"[INFO] context is validated :{result}")

    if "Yes" in result:
        print("---DECISION: DOCS RELEVANT---")
        return "next_step" #this should be a node name
    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        return "rewrite" #this should be a node name

def rewrite(state:AgentState):
    if state["rewritten_query"] is None:
        query = state["user_input"]
    else:
        query = state["rewritten_query"]
    prompt=f"""your an question rewriter understand semantic intent or meaning. 
                rephrases the user's question to be a standalone question optimized for retrieval from RAG application.
                    Here is the initial user question: {query} 
                    return only the improved question
                    """
    result = ask_gemini(prompt=prompt)
    # print(f"[INFO] query is rewritten :{result}")
    state["rewritten_query"]=result.strip()
    return state

def explain(state:AgentState):
    query=state["user_input"]
    content=state["retrieved_docs"]

    prompt=f"""
    You are a Retrieval-Augmented Generation (RAG) assistant. 
    Your job is to answer user questions strictly based on the provided retrieved context. 
    Follow these rules:

    1. If the retrieved context contains relevant information, use ONLY that information to answer.
    2. Do NOT use outside knowledge unless the context is insufficient.
    3. If the context is insufficient or unrelated, say:
        “I could not find relevant information in the documents. Please provide more details.”
    4. When answering:
        - Be accurate and concise.
        - Cite the context by referencing the chunk or section name when relevant.
        - Maintain the meaning exactly as in the retrieved content (no hallucinations).
    5. Never fabricate facts that are not supported by the retrieved documents.
    6. If the user asks something outside the context, politely decline:
        “This question cannot be answered from the available documents.”

        The retrieved context will be provided below as:
        DOCUMENT:
        {content}

        User QUESTION:
        {query}

    """
    result = ask_gemini(prompt=prompt)
    print(f"[INFO] result from llm:{result}")
    state["explanation"]=result

    return state

def agent():

    workflow=StateGraph(AgentState)
    workflow.add_node("retriver",retrieve_step)
    workflow.add_node("rewrite_query",rewrite)
    workflow.add_node("llm",explain)

    workflow.add_edge(START,"retriver")
    workflow.add_edge("rewrite_query","retriver")
    workflow.add_conditional_edges("retriver",validate,{"next_step":"llm","rewrite":"rewrite_query"})
    workflow.add_edge("llm",END)
    app=workflow.compile()

    return app

if __name__=="__main__":

    workflow=StateGraph(AgentState)
    workflow.add_node("retriver",retrieve_step)
    workflow.add_node("rewrite_query",rewrite)
    workflow.add_node("llm",explain)

    workflow.add_edge(START,"retriver")
    workflow.add_edge("rewrite_query","retriver")
    workflow.add_conditional_edges("retriver",validate,{"next_step":"llm","rewrite":"rewrite_query"})
    workflow.add_edge("llm",END)
    app=workflow.compile()

    initial_state = {
    "user_input": "What is meant by ‘electric field lines’?",
    "rewritten_query": None,
    "retrieved_docs": [],
    "validated_docs": []
    }
    result = app.invoke(initial_state)

    # print("Rewritten Query:", result.get("rewritten_query"))
    # print("Retrieved Docs:", result.get("retrieved_docs"))
    print("Explanantion:", result.get("explanation"))

    graph_png = app.get_graph(xray=True).draw_mermaid_png()

    # Save PNG file locally
    output_path = "workflow_graph.png"
    with open(output_path, "wb") as f:
        f.write(graph_png)

