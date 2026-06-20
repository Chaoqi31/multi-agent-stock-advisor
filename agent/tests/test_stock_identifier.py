import unittest

from src.utils.stock_identifier import extract_stock_info, normalize_stock_code


class StockIdentifierTests(unittest.TestCase):
    def test_extracts_parenthesized_us_ticker(self):
        company_name, stock_code = extract_stock_info("帮我看看 Apple (AAPL) 这只股票怎么样")

        self.assertEqual(company_name, "Apple")
        self.assertEqual(stock_code, "AAPL")

    def test_extracts_standalone_us_ticker(self):
        company_name, stock_code = extract_stock_info("AAPL 这个股票值得买吗？")

        self.assertIsNone(company_name)
        self.assertEqual(stock_code, "AAPL")

    def test_normalizes_us_ticker_for_yahoo_symbols(self):
        self.assertEqual(normalize_stock_code("brk.b"), "BRK-B")
        self.assertEqual(normalize_stock_code(" msft "), "MSFT")

    def test_does_not_turn_a_share_numeric_code_into_us_ticker(self):
        self.assertIsNone(normalize_stock_code("600519"))


if __name__ == "__main__":
    unittest.main()
