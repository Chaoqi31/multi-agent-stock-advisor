"""MCP server configuration - settings for connecting to the US-market MCP server.

The server directory is resolved relative to this file so the project stays
portable across machines. Override with the MCP_SERVER_DIR environment variable
if the server lives elsewhere.
"""

import os
from pathlib import Path

# This file lives at <repo>/Finance/Financial-MCP-Agent/src/tools/mcp_config.py
# parents[3] -> <repo>/Finance, which also contains the MCP server package.
_DEFAULT_SERVER_DIR = Path(__file__).resolve().parents[3] / "us_stock_mcp_server"
_SERVER_DIR = os.getenv("MCP_SERVER_DIR", str(_DEFAULT_SERVER_DIR))

SERVER_CONFIGS = {
    "us_stock_mcp_v2": {
        "command": "uv",
        "args": [
            "run",
            "--directory",
            _SERVER_DIR,
            "python",
            "mcp_server.py",
        ],
        "transport": "stdio",
    }
}
