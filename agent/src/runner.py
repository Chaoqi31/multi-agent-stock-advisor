"""Reusable async driver for the analysis workflow.

Keeps the orchestration (state setup, checkpointing, memory store) in one place so both a UI and
scripts can run an analysis and observe per-node progress. Streams node-level updates via the
optional `on_node` callback.
"""

import os
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Tuple

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from src.graph.build_graph import build_workflow
from src.utils.state_definition import AgentState
from src.utils.stock_identifier import extract_stock_info, normalize_stock_code
from src.utils.execution_logger import initialize_execution_logger, finalize_execution_logger
from src.memory.advisory_memory import AdvisoryMemory

_WEEKDAYS_CN = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]

# Ordered node labels for progress display.
NODE_LABELS = {
    "retrieve_memory": "🧠 记忆检索",
    "fundamental_analyst": "📊 基本面分析",
    "technical_analyst": "📈 技术分析",
    "value_analyst": "💰 估值分析",
    "news_analyst": "📰 新闻分析",
    "debate_start": "🔀 汇合",
    "bull_researcher": "🐂 多方论证",
    "bear_researcher": "🐻 空方论证",
    "research_manager": "⚖️ 研究主管综合",
    "risk_disclosure": "⚠️ 风险披露",
    "summarizer": "📝 生成报告",
}


def _checkpoint_path() -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt_dir = os.path.join(root, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    return os.path.join(ckpt_dir, "checkpoints.sqlite")


def build_initial_state(query: str) -> Tuple[AgentState, str]:
    """Mirror the CLI's state setup; return the initial state and a derived thread_id."""
    company, code = extract_stock_info(query) if query else (None, None)
    code = normalize_stock_code(code) if code else None
    now = datetime.now()
    date_cn = now.strftime("%Y年%m月%d日")
    date_en = now.strftime("%Y-%m-%d")
    data: Dict[str, Any] = {
        "query": query,
        "current_date": date_en,
        "current_date_cn": date_cn,
        "current_time": now.strftime("%H:%M:%S"),
        "current_weekday_cn": _WEEKDAYS_CN[now.weekday()],
        "current_time_info": f"{date_cn} ({date_en}) {_WEEKDAYS_CN[now.weekday()]} {now.strftime('%H:%M:%S')}",
        "analysis_timestamp": now.isoformat(),
    }
    if company:
        data["company_name"] = company
    if code:
        data["stock_code"] = code
    thread_id = f"{code or 'NA'}_{now.strftime('%Y%m%d_%H%M%S')}"
    return AgentState(messages=[], data=data, metadata={}), thread_id


async def run_analysis(
    query: str,
    max_debate_rounds: int = 1,
    on_node: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Run the full pipeline, streaming per-node updates to on_node(node, update).

    Returns (thread_id, final_data) where final_data is the final state's `data` dict.
    """
    os.environ["MAX_DEBATE_ROUNDS"] = str(max_debate_rounds)
    initialize_execution_logger()
    state, thread_id = build_initial_state(query)
    workflow = build_workflow()

    final_data: Dict[str, Any] = {}
    async with AsyncSqliteSaver.from_conn_string(_checkpoint_path()) as checkpointer:
        app = workflow.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 60}
        async for chunk in app.astream(state, config=config, stream_mode="updates"):
            for node, update in chunk.items():
                if on_node:
                    on_node(node, update or {})
        snapshot = await app.aget_state(config)
        final_data = (snapshot.values or {}).get("data", {})

    try:
        if final_data.get("final_report"):
            AdvisoryMemory().store(
                thread_id=thread_id,
                ticker=final_data.get("stock_code", ""),
                situation=final_data.get("final_report", "")[:2000],
                recommendation=final_data.get("debate_conclusion", "")[:1000],
                date=final_data.get("current_date", ""),
                report_path=final_data.get("report_path", ""),
            )
    except Exception:  # noqa: BLE001 - memory is best-effort
        pass

    finalize_execution_logger(success=bool(final_data.get("final_report")))
    return thread_id, final_data
