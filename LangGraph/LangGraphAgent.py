import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False

# ─── 加载环境变量 ─────────────────────────────────────────────────
load_dotenv()

# ─── LLM ─────────────────────────────────────────────────────────
llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL_ID", "deepseek-v3"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    temperature=0.7,
)

# ─── Tavily 客户端（可选，未配置时跳过搜索）────────────────────────
tavily_client = None
if TAVILY_AVAILABLE:
    api_key = os.getenv("TAVILY_API_KEY")
    if api_key:
        tavily_client = TavilyClient(api_key=api_key)


# ─── 状态定义 ─────────────────────────────────────────────────────
class SearchState(TypedDict):
    messages: Annotated[list[BaseMessage], "追加新消息"]
    user_query: str        # 用户原始问题
    search_query: str      # 优化后的搜索词
    search_results: str    # Tavily 搜索结果
    final_answer: str      # 最终答案
    step: str             # 当前步骤


# ─── 节点函数 ─────────────────────────────────────────────────────
def understand_query_node(state: SearchState) -> dict:
    """步骤1：理解用户查询并生成搜索关键词"""
    user_message = state["messages"][-1].content

    understand_prompt = f"""分析用户的查询："{user_message}"

请完成两个任务：
1. 简洁总结用户想要了解什么
2. 生成最适合搜索引擎的关键词（中英文均可，要精准）

格式：
理解：[用户需求总结]
搜索词：[最佳搜索关键词]"""

    response = llm.invoke([SystemMessage(content=understand_prompt)])
    response_text = response.content

    search_query = user_message
    if "搜索词：" in response_text:
        search_query = response_text.split("搜索词：")[1].strip()

    return {
        "user_query": user_message,   # 保留用户原始问题
        "search_query": search_query,
        "step": "understood",
        "messages": [AIMessage(content=f"我将为您搜索：{search_query}")],
    }


def tavily_search_node(state: SearchState) -> dict:
    """步骤2：使用 Tavily API 进行真实搜索"""
    search_query = state["search_query"]
    print(f"🔍 正在搜索: {search_query}")

    if tavily_client is None:
        return {
            "search_results": "",
            "step": "search_failed",
            "messages": [AIMessage(content="⚠️ Tavily API 未配置，跳过搜索，直接基于知识回答。")],
        }

    try:
        response = tavily_client.search(
            query=search_query,
            search_depth="basic",
            max_results=5,
            include_answer=True,
        )

        results = response.get("results", [])
        answer_raw = response.get("answer", "")

        lines = []
        for i, r in enumerate(results, 1):
            title   = r.get("title", "无标题")
            url     = r.get("url", "")
            snippet = r.get("content", "")
            lines.append(f"[{i}] {title}\n   {snippet}\n   🔗 {url}")

        search_results = "\n\n".join(lines)
        if answer_raw:
            search_results = f"📌 AI 摘要：{answer_raw}\n\n{search_results}"

        return {
            "search_results": search_results,
            "step": "searched",
            "messages": [AIMessage(content="✅ 搜索完成！正在整理答案...")],
        }

    except Exception as e:
        return {
            "search_results": "",
            "step": "search_failed",
            "messages": [AIMessage(content=f"❌ 搜索失败：{e}，将基于知识回答。")],
        }


def generate_answer_node(state: SearchState) -> dict:
    """步骤3：基于搜索结果生成最终答案"""
    if state["step"] == "search_failed":
        prompt = f"""请直接回答用户的问题，无需道歉或引导。

用户问题：{state['user_query']}"""
        response = llm.invoke([SystemMessage(content=prompt)])
    else:
        prompt = f"""基于以下搜索结果，为用户提供完整、准确、有用的答案：

用户问题：{state['user_query']}

搜索结果：
{state['search_results']}

要求：综合搜索结果给出完整回答，引用来源，适当补充。"""
        response = llm.invoke([SystemMessage(content=prompt)])

    return {
        "final_answer": response.content,
        "step": "completed",
        "messages": [AIMessage(content=response.content)],
    }


# ─── 构建并编译图 ─────────────────────────────────────────────────
def create_search_assistant():
    workflow = StateGraph(SearchState)

    workflow.add_node("understand", understand_query_node)
    workflow.add_node("search", tavily_search_node)
    workflow.add_node("answer", generate_answer_node)

    workflow.add_edge(START, "understand")
    workflow.add_edge("understand", "search")
    workflow.add_edge("search", "answer")
    workflow.add_edge("answer", END)

    return workflow.compile(checkpointer=InMemorySaver())


# ─── 主入口 ──────────────────────────────────────────────────────
if __name__ == "__main__":
    app = create_search_assistant()

    print("=" * 60)
    print("🔍 智能搜索助手（LangGraph + Tavily）")
    print("输入问题后回车，输入 exit 退出")
    print("=" * 60)

    thread = {"configurable": {"thread_id": "user-session-1"}}

    while True:
        user_input = input("\n🧑 您：").strip()
        if not user_input or user_input.lower() in ("exit", "quit", "q"):
            print("再见！")
            break

        print("\n🤖 助手：", end="", flush=True)

        for event in app.stream(
            {"messages": [HumanMessage(content=user_input)]},
            thread,
        ):
            for node_data in event.values():
                if "final_answer" in node_data:
                    print(node_data["final_answer"])
