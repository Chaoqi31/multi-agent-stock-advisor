#!/usr/bin/env python3
"""
US stock information extraction test script.
"""

from src.utils.stock_identifier import extract_stock_info, normalize_stock_code


def test_extraction():
    test_cases = [
        ("分析 Apple (AAPL)", "Apple", "AAPL"),
        ("帮我看看 Tesla (TSLA) 这只股票怎么样", "Tesla", "TSLA"),
        ("我想了解一下 Microsoft (MSFT) 的投资价值", "Microsoft", "MSFT"),
        ("AAPL 这个股票值得买吗？", None, "AAPL"),
        ("分析 Berkshire Hathaway (BRK.B)", "Berkshire Hathaway", "BRK-B"),
        ("给我分析一下 NVIDIA (NVDA) 的财务状况", "NVIDIA", "NVDA"),
    ]

    passed = 0
    for query, expected_company, expected_code in test_cases:
        company_name, stock_code = extract_stock_info(query)
        ok = company_name == expected_company and stock_code == expected_code
        status = "PASS" if ok else "FAIL"
        print(f"{status}: {query}")
        print(f"  expected: company={expected_company}, code={expected_code}")
        print(f"  actual:   company={company_name}, code={stock_code}")
        if ok:
            passed += 1

    print(f"\n{passed}/{len(test_cases)} cases passed")


if __name__ == "__main__":
    print(f"normalize brk.b -> {normalize_stock_code('brk.b')}")
    test_extraction()
