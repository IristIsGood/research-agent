# What you'll learn: LangGraph's state machine (nodes + edges), tool use, agent loops, conditional branching.
# 你将学到：LangGraph 的状态机（节点 + 边）、工具调用、Agent循环、条件分支

# Stack: Python, LangGraph, LangChain, Tavily Search API (free tier), OpenAI.
# 技术栈：Python、LangGraph、LangChain、Tavily 搜索 API（免费版）、OpenAI

# How to start:
# 如何开始：

# 1. pip install langgraph langchain-community tavily-python
# 1. 安装依赖库

# 2. Define a graph with 3 nodes: search → read → summarize.
# 2. 定义一个包含3个节点的图：搜索 → 读取 → 总结

# 3. Use StateGraph with a shared state dict that carries search results between nodes.
# 3. 使用 StateGraph，并通过共享 state 字典在节点之间传递数据

# 4. Add a conditional edge: if the summary is too short → search again.
# 4. 添加条件边：如果总结太短 → 重新搜索


# Resume value: "Built an autonomous research agent using LangGraph that performs multi-step web search and report generation."
# 简历亮点："使用 LangGraph 构建了一个自动化研究 Agent，可进行多步搜索与报告生成"

# ── THEORY: We import the tools we need.
# 理论：导入我们需要的工具
# os + dotenv: load secret keys from our .env file safely
# os + dotenv：安全地从 .env 文件加载密钥
# TypedDict: lets us define a typed Python dict (our STATE)
# TypedDict：定义带类型的字典（我们的状态）
# StateGraph: the core LangGraph class that manages nodes/edges
# StateGraph：LangGraph 核心类，用于管理节点和边
# TavilySearchResults: a pre-built tool that calls the Tavily web search API
# TavilySearchResults：封装好的 Tavily 搜索工具
# ChatOpenAI: LangChain's wrapper around OpenAI's GPT models ──
# ChatOpenAI：LangChain 对 OpenAI 模型的封装 ──

import os
from dotenv import load_dotenv
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI

# Load .env file so os.environ can find your API keys
# 加载 .env 文件，让程序可以读取 API key
load_dotenv()

# ── THEORY: We instantiate our tools once here at the top.
# 理论：在顶部初始化工具（避免重复创建）
# max_results=3 means Tavily returns 3 web results per search.
# max_results=3 表示每次返回3条搜索结果
# model="gpt-4o-mini" is cheap and fast — perfect for learning ──
# gpt-4o-mini 成本低、速度快，适合学习 ──
search_tool = TavilySearchResults(max_results=3)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ── THEORY: TypedDict defines the shape of our shared state.
# 理论：TypedDict 定义 state（状态）的结构
# Think of this as a blueprint for the "notebook" passed between nodes.
# 可以理解为节点之间传递的“共享笔记本”
# query:    the question the user asks (set once, never changed)
# query：用户问题（初始化后不变）
# results:  raw search results from Tavily (list of dicts)
# results：搜索结果（字典列表）
# summary:  the final written report from GPT
# summary：GPT 生成的最终总结
# retries:  how many times we've re-searched (prevents infinite loops) ──
# retries：重试次数（防止死循环） ──

class ResearchState(TypedDict):
    query:   str
    results: List[dict]
    summary: str
    retries: int

# ── THEORY: A NODE is just a Python function.
# 理论：节点本质就是一个函数
# It receives the full state dict and returns a PARTIAL update dict.
# 输入完整 state，返回部分更新
# LangGraph merges your returned dict into the state automatically.
# LangGraph 会自动合并返回值到 state
# This node's job: call the Tavily search API and store results ──
# 当前节点：调用搜索 API 并存储结果 ──

def search_node(state: ResearchState) -> dict:
    # state["query"] was set by the user at graph invocation time
    # query 在 graph.invoke 时传入
    raw_results = search_tool.invoke(state["query"])
    
    # raw_results is a list like:
    # 返回结果格式如下：
    # [{"url": "...", "content": "..."}, ...]
    # We store it in state so the next node can use it
    # 存入 state 供后续节点使用
    
    return {
        "results": raw_results,
        # Increment retries so we can detect infinite loops later
        # 增加重试计数（用于防止无限循环）
        "retries": state.get("retries", 0) + 1
    }

# ── THEORY: This node processes/formats the raw search data.
# 理论：该节点用于处理/格式化搜索结果
# In a real agent this might extract key sentences, deduplicate,
# 实际项目中可能会做：提取重点、去重、排序
# or rank by relevance. Here we keep it simple
# 这里我们简化处理
# ──

def read_node(state: ResearchState) -> dict:
    # Each result is a dict with "content" and "url" keys
    # 每条结果包含 content 和 url
    passages = []
    for r in state["results"]:
        url     = r.get("url",     "unknown source")
        content = r.get("content", "")
        passages.append(f"SOURCE: {url}\n{content}")
    
    # Join into one text block for GPT
    # 拼接成一个文本块供 GPT 使用
    joined = "\n\n---\n\n".join(passages)
    
    # Store formatted text back
    # 将格式化后的文本重新存入 state
    return {"results": [{"formatted": joined}]}

# ── THEORY: This node calls GPT to write a report.
# 理论：该节点调用 GPT 生成总结
# ──

def summarise_node(state: ResearchState) -> dict:
    formatted = state["results"][0]["formatted"]
    
    # Build prompt
    # 构造提示词
    prompt = f"""You are a research assistant. 
Based on these web search results, write a clear 2-paragraph summary 
that answers the query: "{state['query']}"

SEARCH RESULTS:
{formatted}

Write a factual, informative summary:"""

    # Call LLM
    # 调用大模型
    response = llm.invoke(prompt)
    
    return {"summary": response.content}

# ── THEORY: Conditional edge controls flow
# 理论：条件边决定流程走向
# ──

def should_retry(state: ResearchState) -> str:
    summary = state.get("summary", "")
    retries = state.get("retries", 0)
    
    # If too short and retry limit not reached
    # 如果总结太短且未超过最大重试次数
    if len(summary) < 100 and retries < 3:
        return "search_node"
    
    # Otherwise stop
    # 否则结束
    return END

# ── THEORY: Build the graph
# 理论：构建图结构
# ──

graph_builder = StateGraph(ResearchState)

# Register nodes
# 注册节点
graph_builder.add_node("search_node",    search_node)
graph_builder.add_node("read_node",       read_node)
graph_builder.add_node("summarise_node", summarise_node)

# Set entry point
# 设置入口节点
graph_builder.set_entry_point("search_node")

# Add edges
# 添加边（执行顺序）
graph_builder.add_edge("search_node",    "read_node")
graph_builder.add_edge("read_node",       "summarise_node")

# Add conditional edge
# 添加条件跳转
graph_builder.add_conditional_edges(
    "summarise_node",
    should_retry,
)

# Compile graph
# 编译图（变成可运行对象）
graph = graph_builder.compile()

# ── THEORY: Start execution
# 理论：启动执行
# ──

if __name__ == "__main__":
    initial_state = {
        "query":   "What are the latest breakthroughs in fusion energy?",
        "results": [],
        "summary": "",
        "retries": 0
    }
    
    # Run graph
    # 执行 graph
    final_state = graph.invoke(initial_state)
    
    # Output result
    # 输出结果
    print("=" * 60)
    print("RESEARCH SUMMARY")
    print("=" * 60)
    print(final_state["summary"])
    print(f"\nCompleted in {final_state['retries']} search round(s)")