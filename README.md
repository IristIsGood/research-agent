# Autonomous Research Agent: Multi-Step LangGraph Explorer 🔍
An autonomous AI agent capable of performing multi-step web research, data synthesis, and self-evaluating report generation. Built using **LangGraph**, it features a state-managed loop that re-triggers research if the initial findings are insufficient.

## 🎯 Problem & Solution
* **Problem:** Standard LLM research often suffers from "knowledge cutoff" or shallow results because the model only performs a single search and stops, regardless of the quality of the findings.
* **Solution:** A **Stateful Research Loop**. By modeling the research process as a directed graph, the agent can "read" its own summary and decide if it needs to return to the search layer for more data, ensuring high-quality, comprehensive reports.

## 🏗️ Technical Architecture & Agentic Logic
This project implements a **Directed Acyclic Graph (DAG)** with conditional loops to manage the research lifecycle:

1. **The Search Node:** Utilizes the **Tavily Search API** to fetch real-time web data based on the user's query. It tracks `retries` to manage the lifecycle of the loop.
2. **The Read Node:** Acts as a data-processing layer, parsing raw JSON results into a structured text block for the LLM. This mimics a **Retrieval-Augmented Generation (RAG)** context window.
3. **The Summarize Node:** Leverages `gpt-4o-mini` to synthesize the gathered data into a factual, 2-paragraph report.
4. **Conditional Branching (`should_retry`):** The graph's "brain." It inspects the `summary` length and `retries` count. If the output is too thin, it automatically re-routes the flow back to the **Search Node** with the existing state.



## ✨ Key Features
* **Cyclic State Machine:** Uses `StateGraph` to maintain a persistent "notebook" (`ResearchState`) that evolves as the agent works.
* **Autonomous Self-Correction:** Implements logic to detect short or incomplete answers and self-correct via re-searching.
* **Real-Time Web Integration:** Uses Tavily for high-relevance, AI-optimized search results.
* **Infinite Loop Protection:** Hard-capped at 3 search rounds to balance thoroughness with API cost-efficiency.

## 🛠️ Tech Stack
* **Orchestration:** LangGraph (StateGraph)
* **Intelligence:** OpenAI (`gpt-4o-mini`)
* **Search Engine:** Tavily Search API
* **Language:** Python 3.10+ (utilizing `TypedDict` for state safety)

## 🚀 Getting Started

### 1. Installation
```bash
pip install langgraph langchain-openai langchain-community tavily-python python-dotenv
```

### 2. Configuration
Create a `.env` file with your API keys:
```env
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
```

### 3. Usage
```python
python research_agent.py
```
*The agent will stream its progress through the nodes and output a final summary with the total number of search rounds performed.*

## 🔑 Key Technical Decisions
* **Choice of State Design:** Used a `TypedDict` for the `ResearchState`. This ensures that every node in the graph has a strictly defined contract for what it can read and write, preventing "silent failures" during state merges.
* **Model Selection:** Opted for `gpt-4o-mini` with `temperature=0`. In agentic loops, **consistency and speed** are more valuable than creative variance, making a low-temperature, high-efficiency model the ideal choice.
* **Decoupled Processing:** Separated the **Read** and **Summarize** nodes. This architectural choice makes it easy to swap the "Reading" logic (e.g., adding a reranker or a scraper) without breaking the LLM's "Writing" logic.

## 🛡️ License
MIT

## 👤 Developer
**Irist** – Exploring the frontier of Autonomous Agents.
