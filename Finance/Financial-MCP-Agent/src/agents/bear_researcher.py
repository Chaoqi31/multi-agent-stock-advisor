"""Bear researcher node.

Argues the bearish case from the four analyst reports, rebutting the bull's latest points.
Part of a bounded bull/bear debate that sharpens a balanced view before the report is written.
This is research debate only - it never decides or places a trade.
"""

import os
import time

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

from src.utils.state_definition import AgentState
from src.utils.logging_config import setup_logger, ERROR_ICON, SUCCESS_ICON, WAIT_ICON
from src.utils.execution_logger import get_execution_logger
from src.agents.debate_common import build_analyses_block, new_debate_state

load_dotenv(override=True)
logger = setup_logger(__name__)


async def bear_researcher(state: AgentState) -> AgentState:
    """Produce the bearish argument for this debate turn."""
    logger.info(f"{WAIT_ICON} BearResearcher: Building the bearish case.")
    execution_logger = get_execution_logger()
    agent_name = "bear_researcher"
    start = time.time()

    current_data = state.get("data", {})
    current_messages = state.get("messages", [])
    current_metadata = state.get("metadata", {})
    execution_logger.log_agent_start(agent_name, {"stock_code": current_data.get("stock_code")})

    debate = dict(current_data.get("research_debate_state") or new_debate_state())
    company = current_data.get("company_name", "Unknown")
    ticker = current_data.get("stock_code", "Unknown")
    memory_context = current_data.get("memory_context", "")
    bull_last = debate.get("bull_history", "").strip()

    try:
        api_key = os.getenv("OPENAI_COMPATIBLE_API_KEY")
        base_url = os.getenv("OPENAI_COMPATIBLE_BASE_URL")
        model_name = os.getenv("OPENAI_COMPATIBLE_MODEL")
        if not all([api_key, base_url, model_name]):
            raise RuntimeError("Missing OpenAI-compatible environment variables.")

        llm = ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url,
                         temperature=0.5, max_tokens=2000)

        prompt = f"""你是一名"空方"(看空)研究员，正在与"多方"研究员就 {company}（{ticker}）展开投资辩论。

请基于以下四份分析师报告，提出有说服力的**看空/风险论点**：
{build_analyses_block(current_data)}
"""
        if memory_context:
            prompt += f"\n历史上类似情形与当时结论（供参考，避免重复犯错）：\n{memory_context}\n"
        if bull_last:
            prompt += f"\n多方此前的观点（请有针对性地反驳）：\n{bull_last[:1500]}\n"
        prompt += """
要求：
1. 用具体数据和事实指出风险点（估值过高、成长放缓、竞争、监管、财务质量、宏观逆风等）。
2. 针对多方的乐观论点逐条反驳。
3. 观点鲜明、简洁有力，控制在 400 字以内。

请用简体中文输出。"""

        response = await llm.ainvoke([HumanMessage(content=prompt)])
        argument = response.content if isinstance(response, AIMessage) else str(response.content)

        debate["count"] = int(debate.get("count", 0)) + 1
        debate["bear_history"] = (debate.get("bear_history", "") + "\n" + argument).strip()
        debate["history"] = (debate.get("history", "") + f"\n\n【空方·第{debate['count']}轮】\n{argument}").strip()
        debate["current_response"] = f"Bear: {argument}"
        current_data["research_debate_state"] = debate
        current_metadata["bear_researcher_executed"] = True

        execution_logger.log_agent_complete(
            agent_name, {"turn": debate["count"], "argument_length": len(argument)},
            time.time() - start, True)
        logger.info(f"{SUCCESS_ICON} BearResearcher: Turn {debate['count']} done.")
    except Exception as exc:  # noqa: BLE001
        logger.error(f"{ERROR_ICON} BearResearcher error: {exc}", exc_info=True)
        debate["count"] = int(debate.get("count", 0)) + 1
        debate["current_response"] = "Bear: (本轮看空论点生成失败)"
        current_data["research_debate_state"] = debate
        execution_logger.log_agent_complete(agent_name, current_data, time.time() - start, False, str(exc))

    return {"data": current_data, "messages": current_messages, "metadata": current_metadata}
