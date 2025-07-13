# My Own Ai CLI Assistant - Fun AI project to learn from

> _“ Hello Chris. What are we doing today?.”_

---

![Alt text](assets/Screenshot%202025-07-13%20182611.png)


## Features

- **Web Scraper**: Ingests clean, readable content from any URL.
- **Local Knowledge Base**: Embeds documents using LangChain + OllamaEmbeddings + ChromaDB.
- **Smart Query Engine**: RAG & agent-based querying with tool invocation.
- **Integrated Tools**:
  - `nmap` scanner
  - `ping` connectivity check
  - `find` file locator
  - semantic knowledge search
- **Memory-Enhanced Agent**: React-based LangChain agent with conversational memory.
-  **Interactive Mode**: Stay in the terminal and interact in real-time.

---

##  Stack

- `Python 3.10+`
- [LangChain](https://github.com/langchain-ai/langchain)
- [Ollama](https://ollama.com) (local LLM runner)
- [llama3.1:8b] (current running this model, great for technical assistance)
- `Chroma` (vector DB)
- `BeautifulSoup` (scraping)
- `.env` for config
- Modular tool handler using LangChain’s `Tool` wrapper

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## Ensure ollama is installed and running locally:
    ollama run llama3
    
## 2. Create a .env file from the example.env
    name= (this is where the LLM will refrence you)

### Use Cases
    Take advanatge of the RAG by injesting documention via URL

```bash
python main.py --ingest "https://example.com"
```

    Ask it a direct question 

```bash
python main.py --ask "example question"
```

## Interactve mode
Interact with the LLM

```bash
python main.py
```

Then type away

```bash 
> scan my network
> ping google.com
> find /etc/passwd
```

## How It Works
- RAG Pipeline for standard Q&A

- ReAct Agent invokes tools when needed

- Tool Usage triggered via natural language

- Memory System stores context (last 5 messages)

- Streaming Handler gives live feedback from LLM

- Embeddings generated via OllamaEmbeddings("llama3.1:8b")

## Roadmap

[] PDF / Markdown / TXT ingestion

[] GUI wrapper (Gradio / TUI) for desktop companion option

[] Voice command support

[] Docker packaging

## Developer Notes
Personalize your assistant:

Swap llama3.1:8b with any local Ollama-compatible LLM

Drop in new Tool functions via LangChain's API

