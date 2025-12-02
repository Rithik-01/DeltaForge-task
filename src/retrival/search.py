from src.retrival.vectorStore import FaissVectorStore

class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2"):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        self.vectorstore.load()
        print("[INFO] VDB is loaded")

    def search(self, query: str, top_k: int = 5) -> str:
        results = self.vectorstore.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found."
        return context
       
if __name__=="__main__":
    search=RAGSearch()
    result=search.search("how are te cast in this story?")
    print(result)

