import unittest

import requests

from src.us_stock_data_source import USStockDataSource


class FakeResponse:
    def __init__(self, payload=None, text="", status_code=200):
        self._payload = payload
        self.content = text.encode()
        self.text = text
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error", response=self)
        return None

    def json(self):
        return self._payload


class FakeSession:
    def get(self, url, params=None, timeout=15):
        if "/v8/finance/chart/AAPL" in url:
            return FakeResponse(
                {
                    "chart": {
                        "result": [
                            {
                                "timestamp": [1704153600, 1704240000],
                                "indicators": {
                                    "quote": [
                                        {
                                            "open": [100.0, 101.0],
                                            "high": [102.0, 103.0],
                                            "low": [99.0, 100.0],
                                            "close": [101.0, 102.0],
                                            "volume": [1000, 1500],
                                        }
                                    ],
                                    "adjclose": [{"adjclose": [100.5, 101.5]}],
                                },
                            }
                        ],
                        "error": None,
                    }
                }
            )
        if "/v7/finance/quote" in url:
            return FakeResponse(
                {
                    "quoteResponse": {
                        "result": [
                            {
                                "symbol": "BRK-B",
                                "shortName": "Berkshire Hathaway Inc.",
                                "exchange": "NYQ",
                                "currency": "USD",
                                "marketState": "REGULAR",
                                "regularMarketPrice": 400.0,
                                "marketCap": 900000000000,
                            }
                        ]
                    }
                }
            )
        raise AssertionError(f"unexpected request: {url}")


class RateLimitedSession:
    def get(self, url, params=None, timeout=15):
        if "/v8/finance/chart/AAPL" in url:
            return FakeResponse(status_code=429)
        if "stooq.com" in url:
            return FakeResponse(
                text=(
                    "Date,Open,High,Low,Close,Volume\n"
                    "2024-01-02,100,102,99,101,1000\n"
                    "2024-01-03,101,103,100,102,1500\n"
                )
            )
        if "/v7/finance/quote" in url:
            return FakeResponse(status_code=429)
        raise AssertionError(f"unexpected request: {url}")


class FutureRangeSession:
    def get(self, url, params=None, timeout=15):
        if "/v8/finance/chart/AAPL" in url:
            return FakeResponse(status_code=429)
        if "stooq.com" in url and params and "d1" in params:
            return FakeResponse(text="No data")
        if "stooq.com" in url:
            return FakeResponse(
                text=(
                    "Date,Open,High,Low,Close,Volume\n"
                    "2024-01-02,100,102,99,101,1000\n"
                    "2024-01-03,101,103,100,102,1500\n"
                )
            )
        raise AssertionError(f"unexpected request: {url}")


class UnavailableExternalSession:
    def get(self, url, params=None, timeout=15):
        if "/v8/finance/chart/AAPL" in url:
            return FakeResponse(status_code=429)
        if "stooq.com" in url:
            return FakeResponse(status_code=404)
        raise AssertionError(f"unexpected request: {url}")


class USStockDataSourceTests(unittest.TestCase):
    def test_historical_k_data_uses_us_tickers_and_existing_columns(self):
        data_source = USStockDataSource(session=FakeSession())

        result = data_source.get_historical_k_data(
            code="aapl",
            start_date="2024-01-02",
            end_date="2024-01-03",
        )

        self.assertEqual(result["code"].tolist(), ["AAPL", "AAPL"])
        self.assertEqual(result["date"].tolist(), ["2024-01-02", "2024-01-03"])
        self.assertEqual(result["close"].tolist(), ["101.0", "102.0"])
        self.assertIn("pctChg", result.columns)

    def test_basic_info_normalizes_dot_class_tickers_for_yahoo(self):
        data_source = USStockDataSource(session=FakeSession())

        result = data_source.get_stock_basic_info("brk.b")

        self.assertEqual(result.loc[0, "code"], "BRK-B")
        self.assertEqual(result.loc[0, "code_name"], "Berkshire Hathaway Inc.")
        self.assertEqual(result.loc[0, "currency"], "USD")

    def test_basic_info_falls_back_when_quote_endpoint_is_rate_limited(self):
        data_source = USStockDataSource(session=RateLimitedSession())

        result = data_source.get_stock_basic_info("AAPL")

        self.assertEqual(result.loc[0, "code"], "AAPL")
        self.assertEqual(result.loc[0, "code_name"], "Apple Inc.")

    def test_historical_k_data_falls_back_to_stooq_when_yahoo_is_rate_limited(self):
        data_source = USStockDataSource(session=RateLimitedSession())

        result = data_source.get_historical_k_data("AAPL", "2024-01-02", "2024-01-03")

        self.assertEqual(result["code"].tolist(), ["AAPL", "AAPL"])
        self.assertEqual(result["date"].tolist(), ["2024-01-02", "2024-01-03"])
        self.assertEqual(result["close"].tolist(), ["101.0", "102.0"])

    def test_historical_k_data_uses_latest_stooq_data_when_requested_range_is_empty(self):
        data_source = USStockDataSource(session=FutureRangeSession())

        result = data_source.get_historical_k_data("AAPL", "2026-06-01", "2026-06-19")

        self.assertEqual(result["date"].tolist(), ["2024-01-02", "2024-01-03"])
        self.assertEqual(result["code"].tolist(), ["AAPL", "AAPL"])

    def test_historical_k_data_returns_static_sample_when_external_sources_are_unavailable(self):
        data_source = USStockDataSource(session=UnavailableExternalSession())

        result = data_source.get_historical_k_data("AAPL", "2026-06-01", "2026-06-19")

        self.assertEqual(result["code"].tolist(), ["AAPL", "AAPL"])
        self.assertIn("close", result.columns)


if __name__ == "__main__":
    unittest.main()
