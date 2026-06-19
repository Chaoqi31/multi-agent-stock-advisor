"""Memory retriever node.

Runs before the parallel analysts and writes `memory_context` into the shared state so that
downstream reasoning (debate + summary) can be calibrated against similar past situations and
their realized outcomes. Read-only with respect to the other state keys, which keeps it safe to
place ahead of the parallel fan-out.
"""

import os
import time

from src.utils.state_definition import AgentState
from src.memory.advisory_memory import AdvisoryMemory
from src.utils.logging_config import setup_logger, SUCCESS_ICON, WAIT_ICON
from src.utils.execution_logger import get_execution_logger

logger = setup_logger(__name__)


async def memory_retriever(state: AgentState) -> AgentState:
    """Retrieve similar past situations for the queried ticker and inject them into state."""
    logger.info(f"{WAIT_ICON} MemoryRetriever: Looking up similar past situations.")
    execution_logger = get_execution_logger()
    agent_name = "memory_retriever"
    start = time.time()

    current_data = state.get("data", {})
    current_messages = state.get("messages", [])
    current_metadata = state.get("metadata", {})

    ticker = current_data.get("stock_code", "")
    query = current_data.get("query", "")
    execution_logger.log_agent_start(agent_name, {"ticker": ticker, "query": query})

    memory_context = ""
    try:
        memory = AdvisoryMemory()
        top_k = int(os.getenv("MEMORY_TOP_K", "3"))
        memory_context = memory.retrieve(ticker, query, k=top_k)
    except Exception as exc:  # noqa: BLE001 - memory is best-effort
        logger.warning(f"MemoryRetriever: retrieval skipped ({exc}).")

    current_data["memory_context"] = memory_context
    current_metadata["memory_retriever_executed"] = True

    if memory_context:
        logger.info(f"{SUCCESS_ICON} MemoryRetriever: Injected {len(memory_context)} chars of history.")
    else:
        logger.info(f"{SUCCESS_ICON} MemoryRetriever: No prior history for {ticker or 'this query'}.")

    execution_logger.log_agent_complete(
        agent_name, {"memory_context_length": len(memory_context)}, time.time() - start, True
    )
    return {"data": current_data, "messages": current_messages, "metadata": current_metadata}
