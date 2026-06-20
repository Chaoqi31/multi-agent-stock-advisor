"""
Base utility module for MCP tools.
Contains shared helper functions for calling the data source.
"""
import logging
from typing import Callable, Optional
import pandas as pd

from src.formatting.markdown_formatter import format_df_to_markdown
from src.data_source_interface import NoDataFoundError, LoginError, DataSourceError

logger = logging.getLogger(__name__)


def safe_data_source_call(
    tool_name: str,
    data_source_method: Callable,
    data_type_name: str,
    **kwargs
) -> str:
    """
    Generic safe data-source call function that uniformly handles all exceptions and error cases.

    Args:
        tool_name: Tool name, used for logging
        data_source_method: The data-source method to call
        data_type_name: Name of the data type (used for logging)
        **kwargs: Keyword arguments passed to the data-source method

    Returns:
        Result string in Markdown format, or an error message
    """
    try:
        # Call the data-source method
        df = data_source_method(**kwargs)
        logger.info(f"Successfully retrieved {data_type_name} data.")
        return format_df_to_markdown(df)
        
    except NoDataFoundError as e:
        logger.warning(f"NoDataFoundError: {e}")
        return f"Error: {e}"
    except LoginError as e:
        logger.error(f"LoginError: {e}")
        return f"Error: Could not connect to data source. {e}"
    except DataSourceError as e:
        logger.error(f"DataSourceError: {e}")
        return f"Error: An error occurred while fetching data. {e}"
    except ValueError as e:
        logger.warning(f"ValueError: {e}")
        return f"Error: Invalid input parameter. {e}"
    except Exception as e:
        logger.exception(f"Unexpected Exception processing {tool_name}: {e}")
        return f"Error: An unexpected error occurred: {e}"


def call_financial_data_tool(
    tool_name: str,
    # Pass the bound method, e.g. active_data_source.get_profit_data
    data_source_method: Callable,
    data_type_name: str,
    code: str,
    year: str,
    quarter: int
) -> str:
    """
    Helper function to reduce duplicate code across financial-data tools.

    Args:
        tool_name: Tool name, used for logging
        data_source_method: The method to call on the data source
        data_type_name: Financial data type (used for logging)
        code: Stock ticker
        year: Query year
        quarter: Query quarter

    Returns:
        Markdown-formatted string with the result, or an error message
    """
    logger.info(f"Tool '{tool_name}' called for {code}, {year}Q{quarter}")
    try:
        # Basic validation
        if not year.isdigit() or len(year) != 4:
            logger.warning(f"Invalid year format requested: {year}")
            return f"Error: Invalid year '{year}'. Please provide a 4-digit year."
        if not 1 <= quarter <= 4:
            logger.warning(f"Invalid quarter requested: {quarter}")
            return f"Error: Invalid quarter '{quarter}'. Must be between 1 and 4."

        # Call the corresponding method on the instantiated active_data_source
        df = data_source_method(code=code, year=year, quarter=quarter)
        logger.info(
            f"Successfully retrieved {data_type_name} data for {code}, {year}Q{quarter}.")
        # Use a smaller limit for financial tables?
        return format_df_to_markdown(df)

    except NoDataFoundError as e:
        logger.warning(f"NoDataFoundError for {code}, {year}Q{quarter}: {e}")
        return f"Error: {e}"
    except LoginError as e:
        logger.error(f"LoginError for {code}: {e}")
        return f"Error: Could not connect to data source. {e}"
    except DataSourceError as e:
        logger.error(f"DataSourceError for {code}: {e}")
        return f"Error: An error occurred while fetching data. {e}"
    except ValueError as e:
        logger.warning(f"ValueError processing request for {code}: {e}")
        return f"Error: Invalid input parameter. {e}"
    except Exception as e:
        logger.exception(
            f"Unexpected Exception processing {tool_name} for {code}: {e}")
        return f"Error: An unexpected error occurred: {e}"


def call_macro_data_tool(
    tool_name: str,
    data_source_method: Callable,
    data_type_name: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    **kwargs  # For extra arguments such as year_type
) -> str:
    """
    Helper function for macroeconomic-data tools.

    Args:
        tool_name: Tool name, used for logging
        data_source_method: The method to call on the data source
        data_type_name: Data type (used for logging)
        start_date: Optional start date
        end_date: Optional end date
        **kwargs: Extra keyword arguments passed to data_source_method

    Returns:
        Markdown-formatted string with the result, or an error message
    """
    date_range_log = f"from {start_date or 'default'} to {end_date or 'default'}"
    kwargs_log = f", extra_args={kwargs}" if kwargs else ""
    logger.info(f"Tool '{tool_name}' called {date_range_log}{kwargs_log}")
    
    return safe_data_source_call(
        tool_name,
        data_source_method,
        data_type_name,
        start_date=start_date,
        end_date=end_date,
        **kwargs
    )


def call_index_constituent_tool(
    tool_name: str,
    data_source_method: Callable,
    index_name: str,
    date: Optional[str] = None
) -> str:
    """
    Helper function for index-constituent tools.

    Args:
        tool_name: Tool name, used for logging
        data_source_method: The method to call on the data source
        index_name: Index name (used for logging)
        date: Optional query date

    Returns:
        Markdown-formatted string with the result, or an error message
    """
    log_msg = f"Tool '{tool_name}' called for date={date or 'latest'}"
    logger.info(log_msg)
    
    return safe_data_source_call(
        tool_name,
        data_source_method,
        index_name,
        date=date
    )
