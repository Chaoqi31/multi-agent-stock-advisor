import re
from typing import Optional, Tuple


US_TICKER_PATTERN = re.compile(r"\b([A-Za-z]{1,5}(?:[.-][A-Za-z]{1,2})?)\b")


def normalize_stock_code(stock_code: Optional[str]) -> Optional[str]:
    """Normalize a US ticker for Yahoo-style symbols."""
    if not stock_code:
        return None

    candidate = stock_code.strip().upper().replace(".", "-")
    if not re.fullmatch(r"[A-Z]{1,5}(?:-[A-Z]{1,2})?", candidate):
        return None
    return candidate


def extract_stock_info(query: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract a US company name and ticker from natural-language input."""
    stock_code = None
    company_name = None

    parenthesized = re.search(r"([^（(]+?)\s*[（(]\s*([A-Za-z]{1,5}(?:[.-][A-Za-z]{1,2})?)\s*[)）]", query)
    if parenthesized:
        company_name = _clean_company_name(parenthesized.group(1))
        stock_code = normalize_stock_code(parenthesized.group(2))
        return company_name, stock_code

    leading_ticker = re.search(r"^\s*([A-Za-z]{1,5}(?:[.-][A-Za-z]{1,2})?)\b", query)
    if leading_ticker:
        stock_code = normalize_stock_code(leading_ticker.group(1))
        if stock_code:
            return None, stock_code

    quoted_ticker = re.search(r"(?:股票代码|ticker|symbol|代码)[:：\s]*([A-Za-z]{1,5}(?:[.-][A-Za-z]{1,2})?)", query, re.IGNORECASE)
    if quoted_ticker:
        stock_code = normalize_stock_code(quoted_ticker.group(1))

    company_patterns = [
        r"分析一下\s*([^（）()\s]+?)(?:\s*的|\s|$)",
        r"分析\s*([^（）()\s]+)",
        r"帮我看看\s*([^（）()\s]+?)(?:\s*这只|\s*这个)?\s*股票",
        r"了解一下\s*([^（）()\s]+?)(?:\s*的|\s|$)",
        r"给我分析一下\s*([^（）()\s]+?)(?:\s*的|\s|$)",
        r"([^（）()\s]+?)\s*的\s*(?:财务表现|盈利能力|现金流状况|资产负债情况|技术面|股价走势|技术指标|技术面表现|估值水平|市盈率|市净率|估值|投资风险|风险因素|风险评估|投资价值|股票|基本面情况|基本面|财务状况)",
    ]
    for pattern in company_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            company_name = _clean_company_name(match.group(1))
            if company_name:
                break

    return company_name, stock_code


def _clean_company_name(value: str) -> Optional[str]:
    stop_words = ["的", "这个", "这只", "一下", "看看", "了解", "分析", "帮我", "我想", "给我", "财务状况", "投资价值", "基本面情况", "这只股票", "这个股票"]
    cleaned = value.strip()
    for word in stop_words:
        cleaned = cleaned.replace(word, "").strip()

    cleaned = cleaned.strip(" ，,。")
    return cleaned if len(cleaned) >= 2 else None
