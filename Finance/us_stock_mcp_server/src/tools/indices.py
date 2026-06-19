"""
Index-related tools for the MCP server.
Contains tools for retrieving index constituents.
"""
import logging
from typing import Optional

from mcp.server.fastmcp import FastMCP
from src.data_source_interface import FinancialDataSource
from src.tools.base import call_index_constituent_tool

logger = logging.getLogger(__name__)


def register_index_tools(app: FastMCP, active_data_source: FinancialDataSource):
    """
    Register the index-related tools with the MCP app.

    Args:
        app: The FastMCP app instance
        active_data_source: The active financial data source
    """

    @app.tool()
    def get_stock_industry(code: Optional[str] = None, date: Optional[str] = None) -> str:
        """
        Get industry-classification data for a specific stock, or for all stocks on a specific date.

        Args:
            code: Optional US stock ticker (e.g. 'AAPL'). If None, retrieves data for the sample stocks
            date: Optional date, formatted as 'YYYY-MM-DD'. If None, uses the latest available date

        Returns:
            A Markdown table with industry-classification data, or an error message
        """
        log_msg = f"Tool 'get_stock_industry' called for code={code or 'all'}, date={date or 'latest'}"
        logger.info(log_msg)
        try:
            # Date validation can be added here if needed
            df = active_data_source.get_stock_industry(code=code, date=date)
            logger.info(
                f"Successfully retrieved industry data for {code or 'all'}, {date or 'latest'}.")
            from src.formatting.markdown_formatter import format_df_to_markdown
            return format_df_to_markdown(df)

        except Exception as e:
            logger.exception(
                f"Exception processing get_stock_industry: {e}")
            return f"Error: An unexpected error occurred: {e}"

    @app.tool()
    def get_dow30_stocks(date: Optional[str] = None) -> str:
        """
        Get the constituents of the Dow Jones Industrial Average for a given date.

        Args:
            date: Optional date, formatted as 'YYYY-MM-DD'. If None, uses the latest available date

        Returns:
            A Markdown table with the Dow Jones Industrial Average constituents, or an error message
        """
        return call_index_constituent_tool(
            "get_dow30_stocks",
            active_data_source.get_dow30_stocks,
            "Dow Jones Industrial Average",
            date
        )

    @app.tool()
    def get_sp500_stocks(date: Optional[str] = None) -> str:
        """
        Get sample S&P 500 constituents for a given date.

        Args:
            date: Optional date, formatted as 'YYYY-MM-DD'. If None, uses the latest available date

        Returns:
            A Markdown table with the sample S&P 500 constituents, or an error message
        """
        return call_index_constituent_tool(
            "get_sp500_stocks",
            active_data_source.get_sp500_stocks,
            "S&P 500 sample",
            date
        )

    @app.tool()
    def get_nasdaq100_stocks(date: Optional[str] = None) -> str:
        """
        Get sample Nasdaq-100 constituents for a given date.

        Args:
            date: Optional date, formatted as 'YYYY-MM-DD'. If None, uses the latest available date

        Returns:
            A Markdown table with the sample Nasdaq-100 constituents, or an error message
        """
        return call_index_constituent_tool(
            "get_nasdaq100_stocks",
            active_data_source.get_nasdaq100_stocks,
            "Nasdaq-100 sample",
            date
        )
