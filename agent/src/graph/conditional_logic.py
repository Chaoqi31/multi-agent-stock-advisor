"""Conditional routing for the research-debate loop.

The debate alternates bull -> bear -> bull -> ... Each turn increments
data['research_debate_state']['count']. After the bear's turn we either loop back to the bull
for another round or hand off to the research manager, bounded by MAX_DEBATE_ROUNDS so the loop
always terminates (one round == one bull turn + one bear turn).
"""

import os


def max_debate_rounds() -> int:
    try:
        return max(1, int(os.getenv("MAX_DEBATE_ROUNDS", "1")))
    except (TypeError, ValueError):
        return 1


def should_continue_debate(state) -> str:
    """Return the next node after a bear turn: loop to 'bull_researcher' or end at 'research_manager'."""
    debate = (state.get("data", {}) or {}).get("research_debate_state", {}) or {}
    count = int(debate.get("count", 0))
    if count >= 2 * max_debate_rounds():
        return "research_manager"
    return "bull_researcher"
