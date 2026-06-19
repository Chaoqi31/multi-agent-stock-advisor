"""Builds the multi-agent analysis workflow graph.

Topology:

    start_node -> retrieve_memory
                -> [fundamental, technical, value, news]   (parallel analysts)
                -> debate_start                            (fan-in join)
                -> bull_researcher <-> bear_researcher     (bounded debate loop)
                -> research_manager
                -> risk_disclosure
                -> summarizer -> END
"""

from langgraph.graph import StateGraph, END

from src.utils.state_definition import AgentState
from src.agents.memory_retriever import memory_retriever
from src.agents.fundamental_agent import fundamental_agent
from src.agents.technical_agent import technical_agent
from src.agents.value_agent import value_agent
from src.agents.news_agent import news_agent
from src.agents.bull_researcher import bull_researcher
from src.agents.bear_researcher import bear_researcher
from src.agents.research_manager import research_manager
from src.agents.risk_disclosure_agent import risk_disclosure_agent
from src.agents.summary_agent import summary_agent
from src.graph.conditional_logic import should_continue_debate

_ANALYSTS = {
    "fundamental_analyst": fundamental_agent,
    "technical_analyst": technical_agent,
    "value_analyst": value_agent,
    "news_analyst": news_agent,
}


def build_workflow() -> StateGraph:
    """Construct and return the (uncompiled) StateGraph for the analysis pipeline."""
    workflow = StateGraph(AgentState)

    # Nodes
    workflow.add_node("start_node", lambda state: state)
    workflow.add_node("retrieve_memory", memory_retriever)
    for name, fn in _ANALYSTS.items():
        workflow.add_node(name, fn)
    workflow.add_node("debate_start", lambda state: state)  # fan-in join for the analysts
    workflow.add_node("bull_researcher", bull_researcher)
    workflow.add_node("bear_researcher", bear_researcher)
    workflow.add_node("research_manager", research_manager)
    workflow.add_node("risk_disclosure", risk_disclosure_agent)
    workflow.add_node("summarizer", summary_agent)

    # Edges
    workflow.set_entry_point("start_node")
    workflow.add_edge("start_node", "retrieve_memory")
    for name in _ANALYSTS:
        workflow.add_edge("retrieve_memory", name)   # parallel fan-out
        workflow.add_edge(name, "debate_start")       # fan-in join
    workflow.add_edge("debate_start", "bull_researcher")

    # Bounded bull/bear debate loop
    workflow.add_edge("bull_researcher", "bear_researcher")
    workflow.add_conditional_edges(
        "bear_researcher",
        should_continue_debate,
        {"bull_researcher": "bull_researcher", "research_manager": "research_manager"},
    )

    workflow.add_edge("research_manager", "risk_disclosure")
    workflow.add_edge("risk_disclosure", "summarizer")
    workflow.add_edge("summarizer", END)
    return workflow
