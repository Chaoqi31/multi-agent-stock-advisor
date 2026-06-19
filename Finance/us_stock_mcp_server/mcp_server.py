# Main MCP server file
import logging
from datetime import datetime

from mcp.server.fastmcp import FastMCP

# Import the interface and the concrete implementation
from src.data_source_interface import FinancialDataSource
from src.us_stock_data_source import USStockDataSource
from src.utils import setup_logging

# Import the registration functions for each module's tools
from src.tools.stock_market import register_stock_market_tools
from src.tools.financial_reports import register_financial_report_tools
from src.tools.indices import register_index_tools
from src.tools.market_overview import register_market_overview_tools
from src.tools.macroeconomic import register_macroeconomic_tools
from src.tools.date_utils import register_date_utils_tools
from src.tools.analysis import register_analysis_tools
from src.tools.news_crawler import register_news_crawler_tools

# --- Logging Setup ---
# Call the setup function from utils
# You can control the default level here (e.g., logging.DEBUG for more verbose logs)
setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Dependency Injection ---
# Instantiate the data source - easy to swap later if needed
active_data_source: FinancialDataSource = USStockDataSource()

# --- Get current date for system prompt ---
current_date = datetime.now().strftime("%Y-%m-%d")

# --- FastMCP App Initialization ---
app = FastMCP(
#     server_name="us_stock_data_provider",
#     description=f"""Today is {current_date}. Provides tools for analyzing US stock market data. This service offers objective data analysis; users must make their own investment decisions. The analysis is based on public market information, does not constitute investment advice, and is for reference only.

# ⚠️ Important notes:
# 1. The latest trading day is not necessarily today; it must be obtained from get_latest_trading_date()
# 2. Always use the get_latest_trading_date() tool to obtain the actual most recent trading day; do not rely on date knowledge from training data
# 3. When analyzing "latest" or "recent" market conditions, you must first call the get_market_analysis_timeframe() tool to determine the actual analysis time range
# 4. Any date-related analysis must be based on the actual data returned by the tools; do not use outdated or assumed dates
# 5. Added a news crawler feature that can search for company- and industry-related news to support investment decisions
# """,
    # Specify dependencies for installation if needed (e.g., when using `mcp install`)
    # dependencies=["pandas", "requests", "beautifulsoup4"]
)

# --- Register each module's tools ---
register_stock_market_tools(app, active_data_source)
register_financial_report_tools(app, active_data_source)
register_index_tools(app, active_data_source)
register_market_overview_tools(app, active_data_source)
register_macroeconomic_tools(app, active_data_source)
register_date_utils_tools(app, active_data_source)
register_analysis_tools(app, active_data_source)
register_news_crawler_tools(app, active_data_source)

# --- Main Execution Block ---
if __name__ == "__main__":
    logger.info(
        f"Starting US Stock MCP Server via stdio... Today is {current_date}")
    # Run the server using stdio transport, suitable for MCP Hosts like Claude Desktop
    app.run(transport='stdio')
