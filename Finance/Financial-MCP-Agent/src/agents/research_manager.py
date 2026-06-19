"""Research manager node.

Reads the full bull/bear debate transcript and synthesizes a balanced research conclusion that
feeds the final report. As an advisor, it produces a measured analytical view (lean, conviction,
key agreements and disagreements) - not a trade instruction.
"""

import os
import time

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

from src.utils.state_definition import AgentState
from src.utils.logging_config import setup_logger, ERROR_ICON, SUCCESS_ICON, WAIT_ICON
from src.utils.execution_logger import get_execution_logger
from src.agents.debate_common import new_debate_state

load_dotenv(override=True)
logger = setup_logger(__name__)


async def research_manager(state: AgentState) -> AgentState:
    """Synthesize the debate into a balanced conclusion stored as data['debate_conclusion']."""
    logger.info(f"{WAIT_ICON} ResearchManager: Synthesizing the debate.")
    execution_logger = get_execution_logger()
    agent_name = "research_manager"
    start = time.time()

    current_data = state.get("data", {})
    current_messages = state.get("messages", [])
    current_metadata = state.get("metadata", {})
    execution_logger.log_agent_start(agent_name, {"stock_code": current_data.get("stock_code")})

    debate = dict(current_data.get("research_debate_state") or new_debate_state())
    company = current_data.get("company_name", "Unknown")
    ticker = current_data.get("stock_code", "Unknown")
    transcript = debate.get("history", "").strip() or "（本次未产生有效辩论内容）"
    memory_context = current_data.get("memory_context", "")

    conclusion = ""
    try:
        api_key = os.getenv("OPENAI_COMPATIBLE_API_KEY")
        base_url = os.getenv("OPENAI_COMPATIBLE_BASE_URL")
        model_name = os.getenv("OPENAI_COMPATIBLE_MODEL")
        if not all([api_key, base_url, model_name]):
            raise RuntimeError("Missing OpenAI-compatible environment variables.")

        llm = ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url,
                         temperature=0.3, max_tokens=2500)

        prompt = f"""你是研究主管，需要复盘 {company}（{ticker}）的多空辩论并给出**均衡的研究结论**。

多空辩论记录：
{transcript}
"""
        if memory_context:
            prompt += f"\n历史类似情形与当时结论（供校准）：\n{memory_context}\n"
        prompt += """
请输出一段结构化的研究结论，包含：
1. **辩论焦点**：多空双方的核心分歧与共识。
2. **倾向判断**：综合证据后更偏向哪一方，并说明理由与把握程度（高/中/低）。
3. **关键观察指标**：后续需要跟踪验证的 2-3 个指标或催化剂。

注意：这是面向投资者的研究观点，不是交易指令，不要给出买卖手数或下单建议。请用简体中文输出。"""

        response = await llm.ainvoke([HumanMessage(content=prompt)])
        conclusion = response.content if isinstance(response, AIMessage) else str(response.content)

        current_data["debate_conclusion"] = conclusion
        current_metadata["research_manager_executed"] = True
        current_metadata["debate_rounds"] = debate.get("count", 0)

        execution_logger.log_agent_complete(
            agent_name, {"conclusion_length": len(conclusion), "debate_turns": debate.get("count", 0)},
            time.time() - start, True)
        logger.info(f"{SUCCESS_ICON} ResearchManager: Conclusion ready.")
    except Exception as exc:  # noqa: BLE001
        logger.error(f"{ERROR_ICON} ResearchManager error: {exc}", exc_info=True)
        current_data["debate_conclusion"] = f"（辩论综合失败：{exc}）"
        execution_logger.log_agent_complete(agent_name, current_data, time.time() - start, False, str(exc))

    return {"data": current_data, "messages": current_messages, "metadata": current_metadata}
