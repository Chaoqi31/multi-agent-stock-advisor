"""Bull researcher node.

Argues the bullish investment case from the four analyst reports, rebutting the bear's latest
points. Part of a bounded bull/bear debate that sharpens a balanced view before the report is
written. This is research debate only - it never decides or places a trade.
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


async def bull_researcher(state: AgentState) -> AgentState:
    """Produce the bullish argument for this debate turn."""
    logger.info(f"{WAIT_ICON} BullResearcher: Building the bullish case.")
    execution_logger = get_execution_logger()
    agent_name = "bull_researcher"
    start = time.time()

    current_data = state.get("data", {})
    current_messages = state.get("messages", [])
    current_metadata = state.get("metadata", {})
    execution_logger.log_agent_start(agent_name, {"stock_code": current_data.get("stock_code")})

    debate = dict(current_data.get("research_debate_state") or new_debate_state())
    company = current_data.get("company_name", "Unknown")
    ticker = current_data.get("stock_code", "Unknown")
    memory_context = current_data.get("memory_context", "")
    bear_last = debate.get("bear_history", "").strip()

    try:
        api_key = os.getenv("OPENAI_COMPATIBLE_API_KEY")
        base_url = os.getenv("OPENAI_COMPATIBLE_BASE_URL")
        model_name = os.getenv("OPENAI_COMPATIBLE_MODEL")
        if not all([api_key, base_url, model_name]):
            raise RuntimeError("Missing OpenAI-compatible environment variables.")

        llm = ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url,
                         temperature=0.5, max_tokens=2000)

        prompt = f"""你是一名"多方"(看多)研究员，正在与"空方"研究员就 {company}（{ticker}）展开投资辩论。

请基于以下四份分析师报告，提出有说服力的**看多论点**：
{build_analyses_block(current_data)}
"""
        if memory_context:
            prompt += f"\n历史上类似情形与当时结论（供参考，避免重复犯错）：\n{memory_context}\n"
        if bear_last:
            prompt += f"\n空方此前的观点（请有针对性地反驳）：\n{bear_last[:1500]}\n"
        prompt += """
要求：
1. 用具体数据和事实支撑看多逻辑（成长性、护城河、催化剂、估值修复空间等）。
2. 针对空方的担忧逐条反驳。
3. 观点鲜明、简洁有力，控制在 400 字以内。

请用简体中文输出。"""

        response = await llm.ainvoke([HumanMessage(content=prompt)])
        argument = response.content if isinstance(response, AIMessage) else str(response.content)

        debate["count"] = int(debate.get("count", 0)) + 1
        debate["bull_history"] = (debate.get("bull_history", "") + "\n" + argument).strip()
        debate["history"] = (debate.get("history", "") + f"\n\n【多方·第{debate['count']}轮】\n{argument}").strip()
        debate["current_response"] = f"Bull: {argument}"
        current_data["research_debate_state"] = debate
        current_metadata["bull_researcher_executed"] = True

        execution_logger.log_agent_complete(
            agent_name, {"turn": debate["count"], "argument_length": len(argument)},
            time.time() - start, True)
        logger.info(f"{SUCCESS_ICON} BullResearcher: Turn {debate['count']} done.")
    except Exception as exc:  # noqa: BLE001
        logger.error(f"{ERROR_ICON} BullResearcher error: {exc}", exc_info=True)
        debate["count"] = int(debate.get("count", 0)) + 1
        debate["current_response"] = "Bull: (本轮看多论点生成失败)"
        current_data["research_debate_state"] = debate
        execution_logger.log_agent_complete(agent_name, current_data, time.time() - start, False, str(exc))

    return {"data": current_data, "messages": current_messages, "metadata": current_metadata}
