from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate
from typing import Any
from langchain import hub
from tools import ping, nmap, searchDB, find
import os
import hashlib

import config

config.debug #this is the flag to turn debug on or off

embedder = OllamaEmbeddings(model="llama3.1:8b")
VECTOR_PATH = "./vectorstore/chroma_index"

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

memory = ConversationBufferWindowMemory(
    k=5,
    memory_key="chat_history",
    return_messages=True
)

class CleanStreamingHandler(BaseCallbackHandler):
    def __init__(self):
        self.text = ""
        self.in_tool_execution = False
        
    def on_tool_start(self, serialized, input_str, **kwargs):
        self.in_tool_execution = True
        
    def on_tool_end(self, output, **kwargs):
        self.in_tool_execution = False
        
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        if not self.in_tool_execution and not any(keyword in token.lower() for keyword in 
                                                 ['thought:', 'action:', 'action input:', 'observation:']):
            print(token, end="", flush=True)
            self.text += token

def get_or_create_vectorstore():
    """Get existing vectorstore or create new one if it doesn't exist"""
    if os.path.exists(VECTOR_PATH):
        return Chroma(persist_directory=VECTOR_PATH, embedding_function=embedder)
    else:
        os.makedirs(VECTOR_PATH, exist_ok=True)
        return Chroma(persist_directory=VECTOR_PATH, embedding_function=embedder)

def generate_doc_id(text, source):
    """Generate a unique ID for a document based on content and source"""
    content_hash = hashlib.md5(f"{text}{source}".encode()).hexdigest()
    return f"{source}_{content_hash}"

def add_to_vectorstore(text, source):
    """Add new documents to existing vectorstore without overwriting"""
    try:
        # Get existing vectorstore
        vs = get_or_create_vectorstore()
        
        # Create document with unique ID
        doc_id = generate_doc_id(text, source)
        
        try:
            # can use vs.update here 
            existing_docs = vs.get(where={"source": source})
            if existing_docs and existing_docs['ids']:
                #for myself, i passed same ids because i wanna keep the ID same just update the text.
                vs.update_documents(existing_docs["ids"], text)
                print(f"Documents from source '{source}' already exist. Updated with most recent data ingested.")
        except:
           
            pass
        
        # Split text into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        doc = Document(page_content=text, metadata={"source": source, "doc_id": doc_id})
        chunks = splitter.split_documents([doc])
        
        # Add documents to existing vectorstore
        vs.add_documents(chunks)
        
        print(f"Successfully added {len(chunks)} chunks from source: {source}")
        return True
        
    except Exception as e:
        print(f"Error adding to vectorstore: {str(e)}")
        return False

def remove_from_vectorstore(source):
    """Remove documents from vectorstore by source"""
    try:
        vs = get_or_create_vectorstore()
        
        # Get all documents from the source
        docs_to_remove = vs.get(where={"source": source})
        
        if docs_to_remove and docs_to_remove['ids']:
            # Delete the documents
            vs.delete(ids=docs_to_remove['ids'])
            # Note: persist() is not needed in newer Chroma versions
            print(f"Removed {len(docs_to_remove['ids'])} documents from source: {source}")
            return True
        else:
            print(f"No documents found for source: {source}")
            return False
            
    except Exception as e:
        print(f"Error removing from vectorstore: {str(e)}")
        return False

def list_vectorstore_sources():
    """List all sources in the vectorstore"""
    try:
        vs = get_or_create_vectorstore()
        all_docs = vs.get()
        
        if all_docs and all_docs['metadatas']:
            sources = set()
            for metadata in all_docs['metadatas']:
                if 'source' in metadata:
                    sources.add(metadata['source'])
            
            print("Sources in vectorstore:")
            for source in sorted(sources):
                count = sum(1 for meta in all_docs['metadatas'] if meta.get('source') == source)
                print(f"  - {source}: {count} chunks")
            
            return list(sources)
        else:
            print("No documents in vectorstore")
            return []
            
    except Exception as e:
        print(f"Error listing sources: {str(e)}")
        return []

def clear_vectorstore():
    """Clear all documents from vectorstore"""
    try:
        vs = get_or_create_vectorstore()
        # Delete all documents
        all_docs = vs.get()
        if all_docs and all_docs['ids']:
            vs.delete(ids=all_docs['ids'])
            # Note: persist() is not needed in newer Chroma versions
            print(f"Cleared {len(all_docs['ids'])} documents from vectorstore")
            return True
        else:
            print("Vectorstore is already empty")
            return True
    except Exception as e:
        print(f"Error clearing vectorstore: {str(e)}")
        return False

# Updated tools list
tools = [
    Tool(
        name="find file",
        description="Run command in the internal computers commandline, to find and locate files on the computer",
        func=find.find_file
    ),
    Tool(
        name="nmap_scan",
        description="Run nmap network scan on a target IP, hostname, or network range (e.g., 192.168.1.0/24)",
        func=nmap.run_nmap
    ),
    Tool(
        name="ping_host",
        description="Ping a target host to check connectivity. Use IP address or hostname.",
        func=ping.run_ping
    ),
    Tool(
        name="search_knowledge",
        description="Search the ingested knowledge base for information on a topic",
        func=searchDB.search_vectorstore
    )
]

# Your existing prompt and other functions remain the same
agent_prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
    template="""You are E.V.A an AI assistant for IT and cybersecurity tasks.

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}

Thought: {agent_scratchpad}

"""
)

def query_with_agent(question, stream=False):
    """Query using agent with tools and memory - streaming disabled for cleaner output"""
    llm = OllamaLLM(model="llama3.1:8b", temperature=0.1)
    
    agent = create_react_agent(llm, tools, agent_prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        memory=memory,
        tools=tools,
        verbose=True,  
        max_iterations=3,
        handle_parsing_errors=True
    )
    
    try:
        chat_history = ""
        if hasattr(memory, 'chat_memory') and memory.chat_memory.messages:
            history_messages = memory.chat_memory.messages[-4:]
            chat_history = "\n".join([f"{msg.type}: {msg.content}" for msg in history_messages])
        
        result = agent_executor.invoke({
            "input": question,
            "chat_history": chat_history
        })
        
        memory.save_context({"input": question}, {"output": result["output"]})
        
        output = result.get("output", "Sorry, I couldn't process that request.")
        return {"result": output, "full_result": result}
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        memory.save_context({"input": question}, {"output": error_msg})
        return {"result": error_msg, "full_result": None}

def query_vectorstore_ollama(question, stream=True):
    """Original RAG function - kept for simple knowledge base queries"""
    try:
        vs = get_or_create_vectorstore()
        retriever = vs.as_retriever(search_kwargs={"k": 100})  # Get more relevant docs
        
        # Debug: Check retrieval
        docs = retriever.get_relevant_documents(question)
        if config.debug:
            print(f"[DEBUG] Retrieved {len(docs)} relevant documents")
        for i, doc in enumerate(docs):
            if config.debug:
                print(f"[DEBUG] Doc {i+1}: {doc.page_content[:100]}... (Source: {doc.metadata.get('source', 'unknown')})")
        
        if not docs:
            return {"result": "I couldn't find any relevant information in my knowledge base.", "source_documents": []}
        
        if stream:
            streaming_handler = CleanStreamingHandler()
            llm = OllamaLLM(model="llama3.1:8b", callbacks=[streaming_handler])
            qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
            result = qa.invoke(question, config={"callbacks": [streaming_handler]})
            return {"result": streaming_handler.text, "source_documents": result.get("source_documents", [])}
        else:
            llm = OllamaLLM(model="llama3.1:8b")
            qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
            return qa.invoke(question)
    except Exception as e:
        return {"result": f"Knowledge base error: {str(e)}", "source_documents": []}

# Compatibility function for existing code
def get_vectorstore(text, source):
    return add_to_vectorstore(text, source)

def debug_vectorstore():
    """Debug function to inspect vectorstore contents"""
    try:
        vs = get_or_create_vectorstore()
        all_docs = vs.get()
        
        if not all_docs or not all_docs['metadatas']:
            print("❌ Vectorstore is empty!")
            return
        
        print(f"✅ Found {len(all_docs['metadatas'])} documents in vectorstore")
        
        sources = {}
        for i, metadata in enumerate(all_docs['metadatas']):
            source = metadata.get('source', 'unknown')
            if source not in sources:
                sources[source] = []
            sources[source].append({
                'content': all_docs['documents'][i][:200] + "..." if len(all_docs['documents'][i]) > 200 else all_docs['documents'][i],
                'metadata': metadata
            })
        
        # Display sources and sample content
        for source, docs in sources.items():
            print(f"\n📄 Source: {source} ({len(docs)} chunks)")
            print(f"   Sample content: {docs[0]['content']}")
            
    except Exception as e:
        print(f"❌ Error inspecting vectorstore: {e}")

def create_chat_agent():
    """Create a conversational agent that can chat AND use tools"""
    llm = OllamaLLM(model="llama3.1:8b", temperature=0.7)
    
    # Enhanced prompt that handles both chat and tool usage
    chat_prompt = PromptTemplate(
        input_variables=["input", "agent_scratchpad", "tools", "tool_names", "chat_history"],
        template="""You are E.V.A, an AI assistant for IT and cybersecurity tasks. You are helpful, conversational, and knowledgeable.

Chat History:
{chat_history}

You have access to the following tools:
{tools}

For general conversation, respond naturally and helpfully. 
For technical tasks, use the appropriate tools when needed.

Use the following format when tools are needed:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
Thought: I now know the final answer
Final Answer: the final answer to the original input question

For general chat, respond directly without using the tool format.

Current input: {input}

{agent_scratchpad}"""
    )
    
    agent = create_react_agent(llm, tools, chat_prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        memory=memory,
        tools=tools,
        verbose=True,
        max_iterations=3,
        handle_parsing_errors=True
    )
    
    return agent_executor

def query_jarvis(question, stream=False):
    """FIXED: Main query function - now properly handles both chat and database queries"""
    
    # First, check if vectorstore has any data
    try:
        vs = get_or_create_vectorstore()
        all_docs = vs.get()
        vectorstore_has_data = bool(all_docs and all_docs['metadatas'])
        
        if config.debug:
            if vectorstore_has_data:
                print(f"[DEBUG] Vectorstore has {len(all_docs['metadatas'])} documents")
                sources = set(meta.get('source', 'unknown') for meta in all_docs['metadatas'])
                print(f"[DEBUG] Sources: {list(sources)}")
            else:
                print("[DEBUG] Vectorstore is empty!")
                
    except Exception as e:
        if config.debug:
            print(f"[DEBUG] Error checking vectorstore: {e}")
        vectorstore_has_data = False
    
    # Determine query type
    tool_keywords = ['scan', 'nmap', 'ping', 'network', 'connections', 'ports', 'find file', 'find']
    knowledge_keywords = ['what is', 'explain', 'how to', 'tell me about', 'information about']
    
    needs_tools = any(keyword in question.lower() for keyword in tool_keywords)
    needs_knowledge = any(keyword in question.lower() for keyword in knowledge_keywords)
    
    if needs_tools:
        # Use agent for tool-based queries
        return query_with_agent(question, stream=False)
    
    elif needs_knowledge and vectorstore_has_data:
        # Use RAG for knowledge queries when vectorstore has data
        if config.debug:
            print("[DEBUG] Using RAG for knowledge query")
        return query_vectorstore_ollama(question, stream=stream)
    
    elif vectorstore_has_data:
        # Try RAG first, fall back to chat if no relevant docs found
        if config.debug:
            print("[DEBUG] Trying RAG first, will fall back to chat if needed")
        
        rag_result = query_vectorstore_ollama(question, stream=stream)
        
        # If RAG didn't find relevant info, fall back to conversational mode
        if (isinstance(rag_result, dict) and 
            "couldn't find any relevant information" in rag_result.get("result", "")):
            
            if config.debug:
                print("[DEBUG] RAG found no relevant info, falling back to chat")
            
            # Use simple LLM for chat
            llm = OllamaLLM(model="llama3.1:8b", temperature=0.7)
            
            # Get chat history for context
            chat_history = ""
            if hasattr(memory, 'chat_memory') and memory.chat_memory.messages:
                history_messages = memory.chat_memory.messages[-4:]
                chat_history = "\n".join([f"{msg.type}: {msg.content}" for msg in history_messages])
            
            # Create chat prompt
            chat_prompt = f"""You are E.V.A, a helpful AI assistant for IT and cybersecurity tasks. 
            
Previous conversation: {chat_history} Current question: {question} Please respond in a helpful, conversational manner."""
            
            try:
                if stream:
                    streaming_handler = CleanStreamingHandler()
                    llm_with_stream = OllamaLLM(model="llama3.1:8b", temperature=0.7, callbacks=[streaming_handler])
                    response = llm_with_stream.invoke(chat_prompt)
                    result = {"result": streaming_handler.text}
                else:
                    response = llm.invoke(chat_prompt)
                    result = {"result": response}
                
                # Save to memory
                memory.save_context({"input": question}, {"output": result["result"]})
                return result
                
            except Exception as e:
                error_msg = f"Chat error: {str(e)}"
                memory.save_context({"input": question}, {"output": error_msg})
                return {"result": error_msg}
        
        else:
            # RAG found relevant information, save to memory and return
            memory.save_context({"input": question}, {"output": rag_result.get("result", "")})
            return rag_result
    
    else:
        # No vectorstore data, use conversational mode
        if config.debug:
            print("[DEBUG] No vectorstore data, using conversational mode")
        
        # Simple chat mode when no knowledge base is available
        llm = OllamaLLM(model="llama3.1:8b", temperature=0.7)
        
        # Get chat history for context
        chat_history = ""
        if hasattr(memory, 'chat_memory') and memory.chat_memory.messages:
            history_messages = memory.chat_memory.messages[-4:]
            chat_history = "\n".join([f"{msg.type}: {msg.content}" for msg in history_messages])
        
        # Create chat prompt
        chat_prompt = f"""You are E.V.A, a helpful AI assistant for IT and cybersecurity tasks. Previous conversation:{chat_history}
            Current question: {question} 
            Please respond in a helpful, conversational manner. If you need access to specific documents or knowledge base, let the user know they can provide them."""
        
        try:
            if stream:
                streaming_handler = CleanStreamingHandler()
                llm_with_stream = OllamaLLM(model="llama3.1:8b", temperature=0.7, callbacks=[streaming_handler])
                response = llm_with_stream.invoke(chat_prompt)
                result = {"result": streaming_handler.text}
            else:
                response = llm.invoke(chat_prompt)
                result = {"result": response}
            
            # Save to memory
            memory.save_context({"input": question}, {"output": result["result"]})
            return result
            
        except Exception as e:
            error_msg = f"Chat error: {str(e)}"
            memory.save_context({"input": question}, {"output": error_msg})
            return {"result": error_msg}

# Helper function to test the system
def test_system():
    """Test function to verify everything works"""
    print("=== Testing E.V.A System ===\n")
    
    # Test 1: Check vectorstore
    print("1. Testing vectorstore...")
    debug_vectorstore()
    
    # Test 2: Simple chat
    print("\n2. Testing simple chat...")
    result = query_jarvis("Hello, how are you?")
    print(f"Response: {result['result']}")
    
    # Test 3: Knowledge query (if vectorstore has data)
    print("\n3. Testing knowledge query...")
    result = query_jarvis("What information do you have available?")
    print(f"Response: {result['result']}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_system()