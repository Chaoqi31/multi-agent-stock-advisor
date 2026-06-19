from datetime import datetime, timedelta

from src.us_stock_data_source import USStockDataSource


def main():
    data_source = USStockDataSource()
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")

    print("Basic info:")
    print(data_source.get_stock_basic_info("AAPL").head())

    print("\nRecent K data:")
    print(data_source.get_historical_k_data("AAPL", start_date, end_date).tail())

    print("\nMarket dates:")
    print(data_source.get_trade_dates(start_date, end_date).tail())


if __name__ == "__main__":
    main()
