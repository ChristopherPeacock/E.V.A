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

embedder = OllamaEmbeddings(model="llama3.1:8b")
VECTOR_PATH = "./vectorstore/chroma_index"

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

memory = ConversationBufferWindowMemory(
    k=5,  # Reduced to 5 for cleaner context
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
        # Only stream the final response, not tool execution details
        if not self.in_tool_execution and not any(keyword in token.lower() for keyword in 
                                                 ['thought:', 'action:', 'action input:', 'observation:']):
            print(token, end="", flush=True)
            self.text += token

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


def get_vectorstore(text, source):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    doc = Document(page_content=text, metadata={"source": source})
    chunks = splitter.split_documents([doc])
    vs = Chroma.from_documents(chunks, embedder, persist_directory=VECTOR_PATH)
    
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
            history_messages = memory.chat_memory.messages[-4:]  # Last 4 messages
            chat_history = "\n".join([f"{msg.type}: {msg.content}" for msg in history_messages])
        
        result = agent_executor.invoke({
            "input": question,
            "chat_history": chat_history
        })
        
        memory.save_context({"input": question}, {"output": result["output"]})
        
        # Extract just the final answer
        output = result.get("output", "Sorry, I couldn't process that request.")
        return {"result": output, "full_result": result}
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        memory.save_context({"input": question}, {"output": error_msg})
        return {"result": error_msg, "full_result": None}

def query_vectorstore_ollama(question, stream=True):
    """Original RAG function - kept for simple knowledge base queries"""
    try:
        vs = Chroma(persist_directory=VECTOR_PATH, embedding_function=embedder)
        retriever = vs.as_retriever()
        
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

def query_jarvis(question, stream=False):
    """Main query function - detects if tools are needed"""
    tool_keywords = ['scan', 'nmap', 'ping', 'network', 'connections', 'ports', 'find file', 'find']
    
    if any(keyword in question.lower() for keyword in tool_keywords):
        # Use agent for tool-based queries
        return query_with_agent(question, stream=False)
    else:
        # Use simple RAG for general questions
        return query_vectorstore_ollama(question, stream=stream)