import unittest
from unittest.mock import patch

import pandas as pd

from src.us_stock_data_source import USStockDataSource


def _fake_history(symbol, start, end, interval, auto_adjust):
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.0],
            "Close": [101.0, 102.0],
            "Volume": [1000, 1500],
        },
        index=index,
    )


_FAKE_INFO = {
    "symbol": "BRK-B",
    "shortName": "Berkshire Hathaway Inc.",
    "currency": "USD",
    "regularMarketPrice": 400.0,
    "marketCap": 900000000000,
    "exchange": "NYQ",
    "fullExchangeName": "New York Stock Exchange",
    "quoteType": "EQUITY",
    "marketState": "REGULAR",
    "returnOnEquity": 0.12,
    "profitMargins": 0.2,
}


class USStockDataSourceTests(unittest.TestCase):
    @patch("src.us_stock_data_source.yf_history", side_effect=_fake_history)
    def test_historical_k_data_maps_schema_and_uppercases_ticker(self, _mock):
        data_source = USStockDataSource()
        result = data_source.get_historical_k_data(code="aapl", start_date="2024-01-02", end_date="2024-01-03")
        self.assertEqual(result["code"].tolist(), ["AAPL", "AAPL"])
        self.assertEqual(result["date"].tolist(), ["2024-01-02", "2024-01-03"])
        self.assertEqual(result["close"].tolist(), ["101.0", "102.0"])
        self.assertIn("pctChg", result.columns)

    @patch("src.us_stock_data_source.yf_info", return_value=_FAKE_INFO)
    def test_basic_info_normalizes_dot_class_tickers(self, _mock):
        data_source = USStockDataSource()
        result = data_source.get_stock_basic_info("brk.b")
        self.assertEqual(result.loc[0, "code"], "BRK-B")
        self.assertEqual(result.loc[0, "code_name"], "Berkshire Hathaway Inc.")
        self.assertEqual(result.loc[0, "currency"], "USD")

    @patch("src.us_stock_data_source.yf_info", return_value={})
    def test_basic_info_falls_back_when_info_is_empty(self, _mock):
        data_source = USStockDataSource()
        result = data_source.get_stock_basic_info("AAPL")
        self.assertEqual(result.loc[0, "code"], "AAPL")
        self.assertEqual(result.loc[0, "code_name"], "Apple Inc.")

    @patch("src.us_stock_data_source.yf_info", return_value=_FAKE_INFO)
    def test_fundamental_metrics_expose_info_keys(self, _mock):
        data_source = USStockDataSource()
        metrics = data_source._fundamental_metrics("BRK.B")
        self.assertEqual(metrics["returnOnEquity"], 0.12)
        profit = data_source.get_profit_data("BRK.B", "2026", 1)
        self.assertIn("roeAvg", profit.columns)


if __name__ == "__main__":
    unittest.main()
