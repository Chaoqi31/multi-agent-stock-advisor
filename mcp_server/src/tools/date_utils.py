"""
Date utilities for the MCP server.
Contains tools for getting the current date and the latest trading day.
"""
import logging
from datetime import datetime, timedelta
import calendar

from mcp.server.fastmcp import FastMCP
from src.data_source_interface import FinancialDataSource

logger = logging.getLogger(__name__)


def register_date_utils_tools(app: FastMCP, active_data_source: FinancialDataSource):
    """
    Register the date tools with the MCP app.

    Args:
        app: The FastMCP app instance
        active_data_source: The active financial data source
    """

    # @app.tool()
    # def get_current_date() -> str:
    #     """
    #     Get the current date, which can be used to query the latest data.

    #     Returns:
    #         The current date, formatted as 'YYYY-MM-DD'.
    #     """
    #     logger.info("Tool 'get_current_date' called")
    #     current_date = datetime.now().strftime("%Y-%m-%d")
    #     logger.info(f"Returning current date: {current_date}")
    #     return current_date

    @app.tool()
    def get_latest_trading_date() -> str:
        """
        Get the most recent trading date. If today is a trading day, returns today's date; otherwise returns the most recent trading day.

        Returns:
            The most recent trading date, formatted as 'YYYY-MM-DD'.
        """
        logger.info("Tool 'get_latest_trading_date' called")
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            # Get the trading calendar for roughly a week around the current date
            start_date = (datetime.now().replace(day=1)).strftime("%Y-%m-%d")
            end_date = (datetime.now().replace(day=28)).strftime("%Y-%m-%d")

            df = active_data_source.get_trade_dates(
                start_date=start_date, end_date=end_date)

            # Filter out the most recent trading days
            valid_trading_days = df[df['is_trading_day']
                                    == '1']['calendar_date'].tolist()

            # Find the largest date less than or equal to today
            latest_trading_date = None
            for date in valid_trading_days:
                if date <= today and (latest_trading_date is None or date > latest_trading_date):
                    latest_trading_date = date

            if latest_trading_date:
                logger.info(
                    f"Latest trading date found: {latest_trading_date}")
                return latest_trading_date
            else:
                logger.warning(
                    "No trading dates found before today, returning today's date")
                return today

        except Exception as e:
            logger.exception(f"Error determining latest trading date: {e}")
            return datetime.now().strftime("%Y-%m-%d")

    @app.tool()
    def get_market_analysis_timeframe(period: str = "recent") -> str:
        """
        Get a time range suitable for market analysis, based on the current real date rather than training data.
        This tool should be called first when performing market or broad-index analysis, to ensure the latest actual data is used.

        Args:
            period: Time-range type. Allowed values:
                   "recent": the last 1-2 months (default)
                   "quarter": the last quarter
                   "half_year": the last half year
                   "year": the last year

        Returns:
            A detailed description string for the analysis time range, formatted as "YYYY年M月-YYYY年M月".
        """
        logger.info(
            f"Tool 'get_market_analysis_timeframe' called with period={period}")

        now = datetime.now()
        end_date = now

        # Determine the start date based on the requested period
        if period == "recent":
            # The last 1-2 months
            if now.day < 15:
                # If it is early in the month, look at the previous two months
                if now.month == 1:
                    start_date = datetime(now.year - 1, 11, 1)  # November of the previous year
                    middle_date = datetime(now.year - 1, 12, 1)  # December of the previous year
                elif now.month == 2:
                    start_date = datetime(now.year, 1, 1)  # January of this year
                    middle_date = start_date
                else:
                    start_date = datetime(now.year, now.month - 2, 1)  # Two months ago
                    middle_date = datetime(now.year, now.month - 1, 1)  # Last month
            else:
                # If it is mid- or end-of-month, look from last month to now
                if now.month == 1:
                    start_date = datetime(now.year - 1, 12, 1)  # December of the previous year
                    middle_date = start_date
                else:
                    start_date = datetime(now.year, now.month - 1, 1)  # Last month
                    middle_date = start_date

        elif period == "quarter":
            # The last quarter (about 3 months)
            if now.month <= 3:
                start_date = datetime(now.year - 1, now.month + 9, 1)
            else:
                start_date = datetime(now.year, now.month - 3, 1)
            middle_date = start_date

        elif period == "half_year":
            # The last half year
            if now.month <= 6:
                start_date = datetime(now.year - 1, now.month + 6, 1)
            else:
                start_date = datetime(now.year, now.month - 6, 1)
            middle_date = datetime(start_date.year, start_date.month + 3, 1) if start_date.month <= 9 else \
                datetime(start_date.year + 1, start_date.month - 9, 1)

        elif period == "year":
            # The last year
            start_date = datetime(now.year - 1, now.month, 1)
            middle_date = datetime(start_date.year, start_date.month + 6, 1) if start_date.month <= 6 else \
                datetime(start_date.year + 1, start_date.month - 6, 1)
        else:
            # Default to the last 1 month
            if now.month == 1:
                start_date = datetime(now.year - 1, 12, 1)
            else:
                start_date = datetime(now.year, now.month - 1, 1)
            middle_date = start_date

        # Format into a user-friendly display
        def get_month_end_day(year, month):
            return calendar.monthrange(year, month)[1]

        # Ensure the end date does not exceed the current date
        end_day = min(get_month_end_day(
            end_date.year, end_date.month), end_date.day)
        end_display_date = f"{end_date.year}年{end_date.month}月"
        end_iso_date = f"{end_date.year}-{end_date.month:02d}-{end_day:02d}"

        # Start-date display
        start_display_date = f"{start_date.year}年{start_date.month}月"
        start_iso_date = f"{start_date.year}-{start_date.month:02d}-01"

        # If it spans years or covers a long period, include the year in the display
        if start_date.year != end_date.year:
            date_range = f"{start_date.year}年{start_date.month}月-{end_date.year}年{end_date.month}月"
        elif middle_date.month != start_date.month and middle_date.month != end_date.month:
            # If it is a quarter or half year, show the middle month
            date_range = f"{start_date.year}年{start_date.month}月-{middle_date.month}月-{end_date.month}月"
        elif start_date.month != end_date.month:
            date_range = f"{start_date.year}年{start_date.month}月-{end_date.month}月"
        else:
            date_range = f"{start_date.year}年{start_date.month}月"

        result = f"{date_range} (ISO日期范围: {start_iso_date} 至 {end_iso_date})"
        logger.info(f"Generated market analysis timeframe: {result}")
        return result
