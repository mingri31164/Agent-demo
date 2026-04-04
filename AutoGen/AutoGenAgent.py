import os

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.teams import RoundRobinGroupChat


def create_model_client():
    """创建并配置模型客户端（适配 DashScope / DeepSeek 等兼容 OpenAI API 的后端）"""
    from dotenv import load_dotenv
    from autogen_core.models import ModelInfo
    load_dotenv()

    return OpenAIChatCompletionClient(
        model=os.getenv("LLM_MODEL_ID", "deepseek-v3"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        model_info=ModelInfo(
            vision=False,
            function_calling=True,
            json_output=False,
            structured_output=False,
            family="unknown",
            multi_modality=False,
        ),
    )


def create_product_manager(model_client):
    """创建产品经理智能体"""
    system_message = """你是一位经验丰富的产品经理，专门负责软件产品的需求分析和项目规划。

你的核心职责包括：
1. **需求分析**：深入理解用户需求，识别核心功能和边界条件
2. **技术规划**：基于需求制定清晰的技术实现路径
3. **风险评估**：识别潜在的技术风险和用户体验问题
4. **协调沟通**：与工程师和其他团队成员进行有效沟通

当接到开发任务时，请按以下结构进行分析：
1. 需求理解与分析
2. 功能模块划分
3. 技术选型建议
4. 实现优先级排序
5. 验收标准定义

请简洁明了地回应，并在分析完成后说"请工程师开始实现"。"""

    return AssistantAgent(
        name="ProductManager",
        model_client=model_client,
        system_message=system_message,
    )


def create_engineer(model_client):
    """创建软件工程师智能体"""
    system_message = """你是一位资深的软件工程师，擅长 Python 开发和 Web 应用构建。

你的技术专长包括：
1. **Python 编程**：熟练掌握 Python 语法和最佳实践
2. **Web 开发**：精通 Streamlit、Flask、Django 等框架
3. **API 集成**：有丰富的第三方 API 集成经验
4. **错误处理**：注重代码的健壮性和异常处理

当收到开发任务时，请：
1. 仔细分析技术需求
2. 选择合适的技术方案
3. 编写完整的代码实现
4. 添加必要的注释和说明
5. 考虑边界情况和异常处理

请提供完整的可运行代码，并在完成后说"请代码审查员检查"。"""

    return AssistantAgent(
        name="Engineer",
        model_client=model_client,
        system_message=system_message,
    )


def create_code_reviewer(model_client):
    """创建代码审查员智能体"""
    system_message = """你是一位经验丰富的代码审查专家，专注于代码质量和最佳实践。

你的审查重点包括：
1. **代码质量**：检查代码的可读性、可维护性和性能
2. **安全性**：识别潜在的安全漏洞和风险点
3. **最佳实践**：确保代码遵循行业标准和最佳实践
4. **错误处理**：验证异常处理的完整性和合理性

审查流程：
1. 仔细阅读和理解代码逻辑
2. 检查代码规范和最佳实践
3. 识别潜在问题和改进点
4. 提供具体的修改建议
5. 评估代码的整体质量

请提供具体的审查意见，完成后说"代码审查完成，请用户代理测试"。"""

    return AssistantAgent(
        name="CodeReviewer",
        model_client=model_client,
        system_message=system_message,
    )


def create_user_proxy():
    """创建用户代理智能体"""
    return UserProxyAgent(
        name="UserProxy",
        description="""用户代理，负责以下职责：
1. 代表用户提出开发需求
2. 执行最终的代码实现
3. 验证功能是否符合预期
4. 提供用户反馈和建议

完成测试后请回复 TERMINATE。""",
    )


async def create_team_chat(model_client):
    """创建团队聊天实例"""
    product_manager = create_product_manager(model_client)
    engineer = create_engineer(model_client)
    code_reviewer = create_code_reviewer(model_client)
    user_proxy = create_user_proxy()

    return RoundRobinGroupChat(
        participants=[
            product_manager,
            engineer,
            code_reviewer,
            user_proxy
        ],
        max_turns=3,
    )


if __name__ == '__main__':
    import asyncio

    model_client = create_model_client()
    team_chat = asyncio.run(create_team_chat(model_client))

    # ─── 分角色配色方案 ────────────────────────────────────────
    _ROLE_COLORS = {
        "ProductManager": "\033[96m",   # 青色
        "Engineer":       "\033[92m",   # 绿色
        "CodeReviewer":   "\033[93m",   # 黄色
        "UserProxy":      "\033[95m",   # 紫色
        "system":         "\033[90m",   # 灰色
    }
    _BOLD   = "\033[1m"
    _RESET  = "\033[0m"
    _SEP    = f"\n{'─' * 60}\n"

    async def main():
        task = "开发一个简单的计算器 Web 应用，支持加减乘除运算。"
        print(f"\n{_BOLD}📋 任务: {_RESET}{task}")
        print(_SEP)

        stream = team_chat.run_stream(task=task)
        # 已打印的消息 ID，避免重复输出
        printed_ids: set = set()
        # 当前正在流式打印的消息缓存
        active_source = ""
        active_color = ""
        active_prefix = ""

        async for event in stream:
            messages = getattr(event, "messages", None)
            if messages is None:
                continue
            for msg in messages:
                msg_id = getattr(msg, "id", None) or ""
                if msg_id in printed_ids:
                    continue
                printed_ids.add(msg_id)

                source = getattr(msg, "source", None) or "unknown"
                content = getattr(msg, "content", None) or ""
                if not content:
                    continue

                # 流式打印：逐字符/逐词输出（模拟打字机效果）
                active_color = _ROLE_COLORS.get(source, "")
                active_prefix = {
                    "ProductManager": "📌",
                    "Engineer":       "🔧",
                    "CodeReviewer":   "🔍",
                    "UserProxy":      "👤",
                    "user":           "🧑",
                    "system":         "⚙️",
                }.get(source, "💬")

                # 换行后打印角色头
                print(f"{_SEP}{_BOLD}{active_color}{active_prefix} [{source}]{_RESET}\n", end="")
                print(f"{active_color}", end="")
                # 模拟流式输出，每 15 字符 flush 一次
                flush_every = 15
                for i, ch in enumerate(content[:3000]):
                    print(ch, end="", flush=True)
                    if (i + 1) % flush_every == 0:
                        import time
                        time.sleep(0.01)
                print(f"{_RESET}")
                print(f"{_SEP}", end="", flush=True)
                # 大段完成后稍作停顿，区分不同 Agent
                import time
                time.sleep(0.3)

    asyncio.run(main())
