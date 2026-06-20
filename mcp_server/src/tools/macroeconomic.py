"""
Macroeconomic-data tools for the MCP server.
Contains tools for retrieving interest-rate and money-supply data, among others.
"""
import logging
from typing import Optional

from mcp.server.fastmcp import FastMCP
from src.data_source_interface import FinancialDataSource
from src.tools.base import call_macro_data_tool

logger = logging.getLogger(__name__)


def register_macroeconomic_tools(app: FastMCP, active_data_source: FinancialDataSource):
    """
    Register the macroeconomic-data tools with the MCP app.

    Args:
        app: The FastMCP app instance
        active_data_source: The active financial data source
    """

    @app.tool()
    def get_deposit_rate_data(start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
        """
        Get benchmark deposit-rate data (demand and time deposits) within a given date range.

        Args:
            start_date: Optional start date, formatted as 'YYYY-MM-DD'
            end_date: Optional end date, formatted as 'YYYY-MM-DD'

        Returns:
            A Markdown table with deposit-rate data, or an error message
        """
        return call_macro_data_tool(
            "get_deposit_rate_data",
            active_data_source.get_deposit_rate_data,
            "deposit rate",
            start_date, end_date
        )

    @app.tool()
    def get_loan_rate_data(start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
        """
        Get benchmark loan-rate data (lending rates) within a given date range.

        Args:
            start_date: Optional start date, formatted as 'YYYY-MM-DD'
            end_date: Optional end date, formatted as 'YYYY-MM-DD'

        Returns:
            A Markdown table with loan-rate data, or an error message
        """
        return call_macro_data_tool(
            "get_loan_rate_data",
            active_data_source.get_loan_rate_data,
            "loan rate",
            start_date, end_date
        )

    @app.tool()
    def get_required_reserve_ratio_data(start_date: Optional[str] = None, end_date: Optional[str] = None, year_type: str = '0') -> str:
        """
        Get required-reserve-ratio data within a given date range.

        Args:
            start_date: Optional start date, formatted as 'YYYY-MM-DD'
            end_date: Optional end date, formatted as 'YYYY-MM-DD'
            year_type: Optional year type used for date filtering. '0' for the announcement date (default), '1' for the effective date

        Returns:
            A Markdown table with required-reserve-ratio data, or an error message
        """
        # Basic validation of year_type
        if year_type not in ['0', '1']:
            logger.warning(f"Invalid year_type requested: {year_type}")
            return "Error: Invalid year_type '{year_type}'. Valid options are '0' (announcement date) or '1' (effective date)."

        return call_macro_data_tool(
            "get_required_reserve_ratio_data",
            active_data_source.get_required_reserve_ratio_data,
            "required reserve ratio",
            start_date, end_date,
            year_type=year_type
        )

    @app.tool()
    def get_money_supply_data_month(start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
        """
        Get monthly money-supply data (M0, M1, M2) within a given date range.

        Args:
            start_date: Optional start date, formatted as 'YYYY-MM'
            end_date: Optional end date, formatted as 'YYYY-MM'

        Returns:
            A Markdown table with monthly money-supply data, or an error message
        """
        # Specific validation of the YYYY-MM format can be added here if needed
        return call_macro_data_tool(
            "get_money_supply_data_month",
            active_data_source.get_money_supply_data_month,
            "monthly money supply",
            start_date, end_date
        )

    @app.tool()
    def get_money_supply_data_year(start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
        """
        Get annual money-supply data (M0, M1, M2 year-end balances) within a given date range.

        Args:
            start_date: Optional start year, formatted as 'YYYY'
            end_date: Optional end year, formatted as 'YYYY'

        Returns:
            A Markdown table with annual money-supply data, or an error message
        """
        # Specific validation of the YYYY format can be added here if needed
        return call_macro_data_tool(
            "get_money_supply_data_year",
            active_data_source.get_money_supply_data_year,
            "annual money supply",
            start_date, end_date
        )

    # @app.tool()
    # def get_shibor_data(start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
    #     """
    #     Get SHIBOR (Shanghai Interbank Offered Rate) data within a given date range.

    #     Args:
    #         start_date: Optional start date, formatted as 'YYYY-MM-DD'
    #         end_date: Optional end date, formatted as 'YYYY-MM-DD'

    #     Returns:
    #         A Markdown table with SHIBOR data, or an error message
    #     """
    #     return call_macro_data_tool(
    #         "get_shibor_data",
    #         active_data_source.get_shibor_data,
    #         "SHIBOR",
    #         start_date, end_date
    #     )
