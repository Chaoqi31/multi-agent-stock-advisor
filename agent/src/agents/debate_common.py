"""Shared helpers for the research debate nodes (bull / bear / research manager)."""

from typing import Any, Dict


def new_debate_state() -> Dict[str, Any]:
    """Initial research-debate state stored under data['research_debate_state']."""
    return {
        "count": 0,            # total turns taken so far (bull + bear)
        "bull_history": "",    # accumulated bull arguments
        "bear_history": "",    # accumulated bear arguments
        "history": "",         # interleaved transcript
        "current_response": "",  # last speaker's argument, prefixed "Bull:"/"Bear:"
    }


def build_analyses_block(data: Dict[str, Any], limit: int = 1200) -> str:
    """Compact, token-bounded block of the four analyst reports for debate prompts."""
    def clip(value: str) -> str:
        return (value or "(无)")[:limit]

    return (
        "【基本面分析】\n" + clip(data.get("fundamental_analysis", "(无)")) + "\n\n"
        "【技术分析】\n" + clip(data.get("technical_analysis", "(无)")) + "\n\n"
        "【估值分析】\n" + clip(data.get("value_analysis", "(无)")) + "\n\n"
        "【新闻分析】\n" + clip(data.get("news_analysis", "(无)"))
    )
