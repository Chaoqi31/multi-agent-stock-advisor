# US Equity Advisor Agent

A multi-agent system that produces in-depth research reports on US-listed equities. Four
specialist analysts (fundamentals, technicals, valuation, news) gather market data through an
MCP server and reason over it with an LLM; a summarizer consolidates their findings into a
single structured report.

This is an **analysis advisor** — it generates research only and never places trades.

## Quickstart

```bash
# 1. Configure the LLM API
cd Finance/Financial-MCP-Agent
cp .env.example .env          # set OPENAI_COMPATIBLE_API_KEY / _BASE_URL / _MODEL

# 2. Install dependencies (two self-contained uv projects)
cd ../us_stock_mcp_server && uv sync
cd ../Financial-MCP-Agent && uv sync

# 3. Run an analysis
uv run python src/main.py --command "分析 Apple (AAPL)"
```

Only an OpenAI-compatible API key is required to run locally. The optional Qwen-LoRA news
scoring models are disabled by default (no GPU needed).


