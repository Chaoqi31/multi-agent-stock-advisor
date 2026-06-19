"""Risk-disclosure node.

Compiles the material risk factors and standard disclaimers for the report from the analyst
findings and the debate conclusion. This is advisory risk *disclosure* (what could go wrong and
caveats a reader must know) - not position sizing or trade risk management.
"""

import os
import time

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

from src.utils.state_definition import AgentState
from src.utils.logging_config import setup_logger, ERROR_ICON, SUCCESS_ICON, WAIT_ICON
from src.utils.execution_logger import get_execution_logger
from src.agents.debate_common import build_analyses_block

load_dotenv(override=True)
logger = setup_logger(__name__)

_DISCLAIMER = (
    "本报告由 AI 系统基于公开数据自动生成，仅供研究与学习参考，"
    "不构成任何投资建议或要约。投资有风险，决策需结合自身风险承受能力和独立判断。"
)


async def risk_disclosure_agent(state: AgentState) -> AgentState:
    """Produce a risk-factor + disclaimer block stored as data['risk_disclosure']."""
    logger.info(f"{WAIT_ICON} RiskDisclosure: Compiling risk factors and disclaimers.")
    execution_logger = get_execution_logger()
    agent_name = "risk_disclosure_agent"
    start = time.time()

    current_data = state.get("data", {})
    current_messages = state.get("messages", [])
    current_metadata = state.get("metadata", {})
    execution_logger.log_agent_start(agent_name, {"stock_code": current_data.get("stock_code")})

    company = current_data.get("company_name", "Unknown")
    ticker = current_data.get("stock_code", "Unknown")
    debate_conclusion = current_data.get("debate_conclusion", "")

    risk_text = ""
    try:
        api_key = os.getenv("OPENAI_COMPATIBLE_API_KEY")
        base_url = os.getenv("OPENAI_COMPATIBLE_BASE_URL")
        model_name = os.getenv("OPENAI_COMPATIBLE_MODEL")
        if not all([api_key, base_url, model_name]):
            raise RuntimeError("Missing OpenAI-compatible environment variables.")

        llm = ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url,
                         temperature=0.2, max_tokens=1500)

        prompt = f"""你是合规与风险披露专员，需要为 {company}（{ticker}）的研究报告整理**风险披露**部分。

可参考的分析与结论：
{build_analyses_block(current_data, limit=800)}

辩论结论：
{debate_conclusion[:1200]}

请输出：
1. **主要风险因素**：列出 4-6 条该标的的具体风险（如基本面、估值、行业竞争、监管政策、流动性、宏观与市场风险、数据时效性等），每条一句话说明。
2. **数据与方法局限**：简述本报告数据来源与方法的局限性（如公开数据时效、模型判断的不确定性）。

只做风险披露，不要给出买卖建议或仓位。请用简体中文输出，使用简洁的要点列表。"""

        response = await llm.ainvoke([HumanMessage(content=prompt)])
        body = response.content if isinstance(response, AIMessage) else str(response.content)
        risk_text = f"{body}\n\n**免责声明：** {_DISCLAIMER}"

        current_data["risk_disclosure"] = risk_text
        current_metadata["risk_disclosure_executed"] = True

        execution_logger.log_agent_complete(
            agent_name, {"risk_disclosure_length": len(risk_text)}, time.time() - start, True)
        logger.info(f"{SUCCESS_ICON} RiskDisclosure: Done.")
    except Exception as exc:  # noqa: BLE001
        logger.error(f"{ERROR_ICON} RiskDisclosure error: {exc}", exc_info=True)
        current_data["risk_disclosure"] = f"**免责声明：** {_DISCLAIMER}"
        execution_logger.log_agent_complete(agent_name, current_data, time.time() - start, False, str(exc))

    return {"data": current_data, "messages": current_messages, "metadata": current_metadata}
