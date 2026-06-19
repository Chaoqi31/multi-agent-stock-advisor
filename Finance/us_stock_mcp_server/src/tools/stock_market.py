"""
Stock-market data tools for the MCP server.
"""
import logging
from typing import List, Optional, Callable, Any

from mcp.server.fastmcp import FastMCP
from src.data_source_interface import FinancialDataSource, NoDataFoundError, LoginError, DataSourceError
from src.formatting.markdown_formatter import format_df_to_markdown

logger = logging.getLogger(__name__)


def safe_data_fetch(
    func_name: str,
    data_source_func: Callable,
    *args,
    **kwargs
) -> str:
    """
    Safe data-fetch function that uniformly handles all exceptions and error cases.

    Args:
        func_name: Function name, used for logging
        data_source_func: Data-source function
        *args: Positional arguments passed to the data-source function
        **kwargs: Keyword arguments passed to the data-source function

    Returns:
        Markdown-formatted data table, or an error message
    """
    try:
        # Call the data-source function
        df = data_source_func(*args, **kwargs)

        # Format the result
        logger.info(f"Successfully retrieved data for {func_name}, formatting to Markdown.")
        return format_df_to_markdown(df)
        
    except NoDataFoundError as e:
        logger.warning(f"NoDataFoundError for {func_name}: {e}")
        return f"Error: {e}"
    except LoginError as e:
        logger.error(f"LoginError for {func_name}: {e}")
        return f"Error: Could not connect to data source. {e}"
    except DataSourceError as e:
        logger.error(f"DataSourceError for {func_name}: {e}")
        return f"Error: An error occurred while fetching data. {e}"
    except ValueError as e:
        logger.warning(f"ValueError processing request for {func_name}: {e}")
        return f"Error: Invalid input parameter. {e}"
    except Exception as e:
        logger.exception(f"Unexpected Exception processing {func_name}: {e}")
        return f"Error: An unexpected error occurred: {e}"


def register_stock_market_tools(app: FastMCP, active_data_source: FinancialDataSource):
    """
    Register the stock-market data tools with the MCP app.

    Args:
        app: The FastMCP app instance
        active_data_source: The active financial data source
    """

    @app.tool()
    def get_historical_k_data(
        code: str,
        start_date: str,
        end_date: str,
        frequency: str = "d",
        adjust_flag: str = "3",
        fields: Optional[List[str]] = None,
    ) -> str:
        """
        Get historical K-line (OHLCV) data for a US stock.

        Args:
            code: US stock ticker (e.g. 'AAPL', 'MSFT', 'BRK-B')
            start_date: Start date, formatted as 'YYYY-MM-DD'
            end_date: End date, formatted as 'YYYY-MM-DD'
            frequency: Data frequency. Valid options:
                         'd': daily
                         'w': weekly
                         'm': monthly
                         '5': 5-minute
                         '15': 15-minute
                         '30': 30-minute
                         '60': 60-minute
                       Defaults to 'd'
            adjust_flag: Price/volume adjustment flag. Kept for compatibility with the original interface:
                           '1': forward-adjusted
                           '2': back-adjusted
                           '3': not adjusted
                         Defaults to '3'
            fields: Optional list of specific data fields (must be valid fields in the returned table)
                    If None or empty, default fields are used (e.g. date, code, open, high, low, close, volume, amount, pctChg)

        Returns:
            A Markdown-formatted string containing the K-line data table, or an error message
            The table may be truncated if the result set is too large
        """
        logger.info(
            f"Tool 'get_historical_k_data' called for {code} ({start_date}-{end_date}, freq={frequency}, adj={adjust_flag}, fields={fields})")

        # Validate frequency and adjustment flag
        valid_freqs = ['d', 'w', 'm', '5', '15', '30', '60']
        valid_adjusts = ['1', '2', '3']
        if frequency not in valid_freqs:
            logger.warning(f"Invalid frequency requested: {frequency}")
            return f"Error: Invalid frequency '{frequency}'. Valid options are: {valid_freqs}"
        if adjust_flag not in valid_adjusts:
            logger.warning(f"Invalid adjust_flag requested: {adjust_flag}")
            return f"Error: Invalid adjust_flag '{adjust_flag}'. Valid options are: {valid_adjusts}"

        # Use the generic function to handle data fetching
        return safe_data_fetch(
            "get_historical_k_data",
            active_data_source.get_historical_k_data,
            code=code,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            adjust_flag=adjust_flag,
            fields=fields,
        )

    @app.tool()
    def get_stock_basic_info(code: str, fields: Optional[List[str]] = None) -> str:
        """
        Get basic information for a given US stock.

        Args:
            code: US stock ticker (e.g. 'AAPL', 'MSFT', 'BRK-B')
            fields: Optional list used to select specific columns from the available basic information
                    (e.g. ['code', 'code_name', 'industry', 'listingDate'])
                    If None or empty, returns all available basic-information columns

        Returns:
            A Markdown-formatted string containing the basic stock information table, or an error message
        """
        logger.info(
            f"Tool 'get_stock_basic_info' called for {code} (fields={fields})")

        # Use the generic function to handle data fetching
        return safe_data_fetch(
            "get_stock_basic_info",
            active_data_source.get_stock_basic_info,
            code=code,
            fields=fields,
        )

    @app.tool()
    def get_dividend_data(code: str, year: str, year_type: str = "report") -> str:
        """
        Get dividend information for a given stock ticker and year.

        Args:
            code: US stock ticker (e.g. 'AAPL', 'MSFT', 'BRK-B')
            year: Query year (e.g. '2023')
            year_type: Year type. Kept for compatibility with the original interface:
                         'report': year of the proposal announcement
                         'operate': year of the ex-dividend/ex-rights event
                       Defaults to 'report'

        Returns:
            A Markdown-formatted string containing the dividend data table, or an error message
        """
        logger.info(
            f"Tool 'get_dividend_data' called for {code}, year={year}, year_type={year_type}")

        # Basic validation
        if year_type not in ['report', 'operate']:
            logger.warning(f"Invalid year_type requested: {year_type}")
            return f"Error: Invalid year_type '{year_type}'. Valid options are: 'report', 'operate'"
        if not year.isdigit() or len(year) != 4:
            logger.warning(f"Invalid year format requested: {year}")
            return f"Error: Invalid year '{year}'. Please provide a 4-digit year."

        # Use the generic function to handle data fetching
        return safe_data_fetch(
            "get_dividend_data",
            active_data_source.get_dividend_data,
            code=code,
            year=year,
            year_type=year_type,
        )

    @app.tool()
    def get_adjust_factor_data(code: str, start_date: str, end_date: str) -> str:
        """
        Get adjustment-factor data for a given stock ticker and date range.
        Retrieves US stock split/adjustment events within the given date range. Useful for understanding adjusted prices.

        Args:
            code: US stock ticker (e.g. 'AAPL', 'MSFT', 'BRK-B')
            start_date: Start date, formatted as 'YYYY-MM-DD'
            end_date: End date, formatted as 'YYYY-MM-DD'

        Returns:
            A Markdown-formatted string containing the adjustment-factor data table, or an error message
        """
        logger.info(
            f"Tool 'get_adjust_factor_data' called for {code} ({start_date} to {end_date})")

        # Use the generic function to handle data fetching
        return safe_data_fetch(
            "get_adjust_factor_data",
            active_data_source.get_adjust_factor_data,
            code=code,
            start_date=start_date,
            end_date=end_date,
        )
