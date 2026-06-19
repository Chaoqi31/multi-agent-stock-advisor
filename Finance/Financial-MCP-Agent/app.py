"""Streamlit dashboard for the US-equity advisor agent.

Run with:  uv run streamlit run app.py
"""

import asyncio
import os
import sys

# Ensure `src` is importable regardless of how streamlit launches this file.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)

from src.runner import run_analysis, NODE_LABELS
from src.memory.advisory_memory import AdvisoryMemory

st.set_page_config(page_title="US Equity Advisor Agent", page_icon="📈", layout="wide")

EXAMPLES = ["分析 Apple (AAPL)", "分析英伟达 (NVDA)", "看看 Tesla (TSLA) 怎么样", "Microsoft (MSFT) 值得投资吗"]


def _render_progress(placeholder, done_nodes):
    lines = []
    for node, label in NODE_LABELS.items():
        lines.append(f"{'✅' if node in done_nodes else '⬜️'} {label}")
    placeholder.markdown("  \n".join(lines))


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ 设置")
    rounds = st.slider("多空辩论轮数", 1, 3, 1, help="每轮 = 一次多方 + 一次空方")
    st.caption(f"记忆: {'开启' if os.getenv('ENABLE_MEMORY', 'true').lower() != 'false' else '关闭'} · "
               f"模型: {os.getenv('OPENAI_COMPATIBLE_MODEL', '未配置')}")

    st.divider()
    st.subheader("🧠 历史分析")
    recent = AdvisoryMemory().list_recent(15)
    if recent:
        for item in recent:
            outcome = item.get("outcome") or "—"
            st.markdown(f"**{item.get('ticker','?')}** · {item.get('date','')}  \n"
                        f"结果: {outcome[:60]}")
    else:
        st.caption("暂无历史记录")

# ----------------------------------------------------------------------------- header
st.title("📈 美股投资顾问 Agent")
st.caption("多 Agent 研究系统：并行分析 → 多空辩论 → 风险披露 → 综合报告。仅出研究报告，不做交易决策。")

query = st.text_input("输入分析需求", value="", placeholder=EXAMPLES[0])
cols = st.columns(len(EXAMPLES))
for col, ex in zip(cols, EXAMPLES):
    if col.button(ex, use_container_width=True):
        query = ex
run_clicked = st.button("🚀 开始分析", type="primary", use_container_width=True)

# ----------------------------------------------------------------------------- run
if run_clicked and query.strip():
    done_nodes = set()
    st.subheader("⏳ 分析进度")
    progress_box = st.empty()
    _render_progress(progress_box, done_nodes)

    def on_node(node, _update):
        done_nodes.add(node)
        _render_progress(progress_box, done_nodes)

    try:
        with st.spinner("多 Agent 分析进行中…（约 1–5 分钟，取决于行情接口）"):
            thread_id, data = asyncio.run(run_analysis(query.strip(), rounds, on_node))
        st.session_state["result"] = {"thread_id": thread_id, "data": data}
    except Exception as exc:  # noqa: BLE001
        st.error(f"分析失败：{exc}")

# ----------------------------------------------------------------------------- results
result = st.session_state.get("result")
if result:
    data = result["data"]
    thread_id = result["thread_id"]
    report = data.get("final_report", "")
    st.success(f"✅ 完成 · thread_id: `{thread_id}`")

    tab_report, tab_debate, tab_memory = st.tabs(["📄 分析报告", "⚔️ 多空辩论", "🧠 记忆与反思"])

    with tab_report:
        if report:
            ticker = data.get("stock_code", "report")
            st.download_button("⬇️ 下载报告 (.md)", report, file_name=f"report_{ticker}.md")
            st.markdown(report)
        else:
            st.warning("本次未生成报告，请查看日志。")

    with tab_debate:
        debate = data.get("research_debate_state", {}) or {}
        left, right = st.columns(2)
        with left:
            st.markdown("#### 🐂 多方")
            st.markdown(debate.get("bull_history") or "（无）")
        with right:
            st.markdown("#### 🐻 空方")
            st.markdown(debate.get("bear_history") or "（无）")
        st.divider()
        st.markdown("#### ⚖️ 研究主管结论")
        st.markdown(data.get("debate_conclusion") or "（无）")

    with tab_memory:
        st.markdown("#### 本次注入的历史参考")
        st.markdown(data.get("memory_context") or "（无历史记录）")
        st.divider()
        st.markdown("#### 回填实际结果 (reflection)")
        with st.form("reflect_form"):
            tid = st.text_input("thread_id", value=thread_id)
            outcome = st.text_input("实际结果", placeholder="例如：AAPL +6% over 5 trading days")
            if st.form_submit_button("提交回填"):
                ok = AdvisoryMemory().update_with_outcome(tid, outcome or "（未填写）")
                st.success("已回填") if ok else st.error("未找到该 thread_id")
