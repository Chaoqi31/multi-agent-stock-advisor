"""
Financial-report tools for the MCP server.
"""
import logging
from typing import List, Optional

from mcp.server.fastmcp import FastMCP
from src.data_source_interface import FinancialDataSource
from src.tools.base import call_financial_data_tool

logger = logging.getLogger(__name__)


def safe_financial_report_fetch(
    func_name: str,
    data_source_func,
    report_type: str,
    code: str,
    start_date: str = None,
    end_date: str = None,
    year: str = None,
    quarter: int = None
) -> str:
    """
    Safe financial-report data-fetch function that uniformly handles all exceptions and error cases.

    Args:
        func_name: Function name, used for logging
        data_source_func: Data-source function
        report_type: Description of the report type
        code: Stock ticker
        start_date: Start date (optional)
        end_date: End date (optional)
        year: Year (optional)
        quarter: Quarter (optional)

    Returns:
        Markdown-formatted data table, or an error message
    """
    try:
        # Call a different data-source function depending on the argument types
        if year and quarter:
            df = data_source_func(code=code, year=year, quarter=quarter)
        elif start_date and end_date:
            df = data_source_func(code=code, start_date=start_date, end_date=end_date)
        else:
            raise ValueError("Invalid parameters provided")
        
        logger.info(f"Successfully retrieved {report_type} data for {code}")
        from src.formatting.markdown_formatter import format_df_to_markdown
        return format_df_to_markdown(df)
        
    except Exception as e:
        logger.exception(f"Exception processing {func_name} for {code}: {e}")
        return f"Error: An unexpected error occurred: {e}"


def register_financial_report_tools(app: FastMCP, active_data_source: FinancialDataSource):
    """
    Register the financial-report tools with the MCP app.

    Args:
        app: The FastMCP app instance
        active_data_source: The active financial data source
    """

    @app.tool()
    def get_profit_data(code: str, year: str, quarter: int) -> str:
        """
        Get a stock's quarterly profitability data (e.g. ROE, net profit margin).

        Args:
            code: US stock ticker (e.g. 'AAPL')
            year: 4-digit year (e.g. '2023')
            quarter: Quarter (1, 2, 3, or 4)

        Returns:
            A Markdown table with profitability data, or an error message
        """
        return call_financial_data_tool(
            "get_profit_data",
            active_data_source.get_profit_data,
            "profitability",
            code, year, quarter
        )

    @app.tool()
    def get_operation_data(code: str, year: str, quarter: int) -> str:
        """
        Get a stock's quarterly operating-efficiency data (e.g. turnover ratios).

        Args:
            code: US stock ticker (e.g. 'AAPL')
            year: 4-digit year (e.g. '2023')
            quarter: Quarter (1, 2, 3, or 4)

        Returns:
            A Markdown table with operating-efficiency data, or an error message
        """
        return call_financial_data_tool(
            "get_operation_data",
            active_data_source.get_operation_data,
            "operating efficiency",
            code, year, quarter
        )

    @app.tool()
    def get_growth_data(code: str, year: str, quarter: int) -> str:
        """
        Get a stock's quarterly growth data (e.g. year-over-year growth rates).

        Args:
            code: US stock ticker (e.g. 'AAPL')
            year: 4-digit year (e.g. '2023')
            quarter: Quarter (1, 2, 3, or 4)

        Returns:
            A Markdown table with growth data, or an error message
        """
        return call_financial_data_tool(
            "get_growth_data",
            active_data_source.get_growth_data,
            "growth",
            code, year, quarter
        )

    @app.tool()
    def get_balance_data(code: str, year: str, quarter: int) -> str:
        """
        Get a stock's quarterly balance-sheet / solvency data (e.g. current ratio, debt-to-asset ratio).

        Args:
            code: US stock ticker (e.g. 'AAPL')
            year: 4-digit year (e.g. '2023')
            quarter: Quarter (1, 2, 3, or 4)

        Returns:
            A Markdown table with balance-sheet data, or an error message
        """
        return call_financial_data_tool(
            "get_balance_data",
            active_data_source.get_balance_data,
            "balance sheet",
            code, year, quarter
        )

    @app.tool()
    def get_cash_flow_data(code: str, year: str, quarter: int) -> str:
        """
        Get a stock's quarterly cash-flow data (e.g. CFO-to-revenue ratio).

        Args:
            code: US stock ticker (e.g. 'AAPL')
            year: 4-digit year (e.g. '2023')
            quarter: Quarter (1, 2, 3, or 4)

        Returns:
            A Markdown table with cash-flow data, or an error message
        """
        return call_financial_data_tool(
            "get_cash_flow_data",
            active_data_source.get_cash_flow_data,
            "cash flow",
            code, year, quarter
        )

    @app.tool()
    def get_dupont_data(code: str, year: str, quarter: int) -> str:
        """
        Get a stock's quarterly DuPont-analysis data (ROE decomposition).

        Args:
            code: US stock ticker (e.g. 'AAPL')
            year: 4-digit year (e.g. '2023')
            quarter: Quarter (1, 2, 3, or 4)

        Returns:
            A Markdown table with DuPont-analysis data, or an error message
        """
        return call_financial_data_tool(
            "get_dupont_data",
            active_data_source.get_dupont_data,
            "DuPont analysis",
            code, year, quarter
        )

    @app.tool()
    def get_performance_express_report(code: str, start_date: str, end_date: str) -> str:
        """
        Get a stock's performance express-report data within a given date range.
        Note: companies are only required to publish these reports under certain circumstances.

        Args:
            code: US stock ticker (e.g. 'AAPL')
            start_date: Start date (report publication date), formatted as 'YYYY-MM-DD'
            end_date: End date (report publication date), formatted as 'YYYY-MM-DD'

        Returns:
            A Markdown table with performance express-report data, or an error message
        """
        return safe_financial_report_fetch(
            "get_performance_express_report",
            active_data_source.get_performance_express_report,
            "performance express report",
            code,
            start_date=start_date,
            end_date=end_date
        )

    @app.tool()
    def get_forecast_report(code: str, start_date: str, end_date: str) -> str:
        """
        Get a stock's earnings-forecast data within a given date range.
        Note: companies are only required to publish these reports under certain circumstances.

        Args:
            code: US stock ticker (e.g. 'AAPL')
            start_date: Start date (report publication date), formatted as 'YYYY-MM-DD'
            end_date: End date (report publication date), formatted as 'YYYY-MM-DD'

        Returns:
            A Markdown table with earnings-forecast data, or an error message
        """
        return safe_financial_report_fetch(
            "get_forecast_report",
            active_data_source.get_forecast_report,
            "earnings forecast",
            code,
            start_date=start_date,
            end_date=end_date
        )

