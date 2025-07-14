import os
from langchain_chroma import Chroma
from langchain import vectorstores
from langchain_ollama import OllamaEmbeddings

embedder = OllamaEmbeddings(model="llama3.1:8b")
VECTOR_PATH = "./vectorstore/chroma_index" 

def search_vectorstore(query: str) -> str:
    """Search the knowledge base for relevant information."""
    try:
        if not os.path.exists(VECTOR_PATH):
            return "No knowledge base found. Use --ingest to add documents first."
        
        vs = Chroma(persist_directory=VECTOR_PATH, embedding_function=embedder)
        docs = vs.similarity_search(query, k=3)
        if docs:
            results = "\n---\n".join([doc.page_content[:500] + "..." if len(doc.page_content) > 500 
                                     else doc.page_content for doc in docs])
            return f"📚 Knowledge base results for '{query}':\n{results}"
        else:
            return f"No relevant information found for '{query}' in knowledge base."
    except Exception as e:
        return f"Error searching knowledge base: {str(e)}"