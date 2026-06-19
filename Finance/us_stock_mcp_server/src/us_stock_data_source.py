import logging
import math
import os
import re
from datetime import datetime, timedelta, timezone
from io import StringIO
from typing import Dict, List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .data_source_interface import DataSourceError, FinancialDataSource, NoDataFoundError

logger = logging.getLogger(__name__)


def _local_news_models_enabled() -> bool:
    """Whether to load the local Qwen-LoRA risk/sentiment models for news scoring.

    Disabled by default so the server runs anywhere with no GPU or model files.
    Enable by setting USE_LOCAL_NEWS_MODELS=true together with the QWEN_*_MODEL paths.
    """
    return os.getenv("USE_LOCAL_NEWS_MODELS", "false").strip().lower() in ("1", "true", "yes")


DEFAULT_K_FIELDS = [
    "date",
    "code",
    "open",
    "high",
    "low",
    "close",
    "preclose",
    "volume",
    "amount",
    "adjustflag",
    "turn",
    "tradestatus",
    "pctChg",
    "peTTM",
    "pbMRQ",
    "psTTM",
    "pcfNcfTTM",
    "isST",
]

US_STOCK_UNIVERSE = [
    {"code": "AAPL", "code_name": "Apple Inc.", "exchange": "NASDAQ", "industry": "Consumer Electronics"},
    {"code": "MSFT", "code_name": "Microsoft Corporation", "exchange": "NASDAQ", "industry": "Software"},
    {"code": "NVDA", "code_name": "NVIDIA Corporation", "exchange": "NASDAQ", "industry": "Semiconductors"},
    {"code": "AMZN", "code_name": "Amazon.com, Inc.", "exchange": "NASDAQ", "industry": "Internet Retail"},
    {"code": "GOOGL", "code_name": "Alphabet Inc.", "exchange": "NASDAQ", "industry": "Internet Content"},
    {"code": "META", "code_name": "Meta Platforms, Inc.", "exchange": "NASDAQ", "industry": "Internet Content"},
    {"code": "TSLA", "code_name": "Tesla, Inc.", "exchange": "NASDAQ", "industry": "Auto Manufacturers"},
    {"code": "BRK-B", "code_name": "Berkshire Hathaway Inc.", "exchange": "NYSE", "industry": "Insurance"},
    {"code": "JPM", "code_name": "JPMorgan Chase & Co.", "exchange": "NYSE", "industry": "Banks"},
    {"code": "V", "code_name": "Visa Inc.", "exchange": "NYSE", "industry": "Credit Services"},
]

DOW_30 = [
    "AAPL",
    "AMGN",
    "AMZN",
    "AXP",
    "BA",
    "CAT",
    "CRM",
    "CSCO",
    "CVX",
    "DIS",
    "GS",
    "HD",
    "HON",
    "IBM",
    "JNJ",
    "JPM",
    "KO",
    "MCD",
    "MMM",
    "MRK",
    "MSFT",
    "NKE",
    "NVDA",
    "PG",
    "SHW",
    "TRV",
    "UNH",
    "V",
    "VZ",
    "WMT",
]

NASDAQ_100_SAMPLE = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "AVGO",
    "META",
    "TSLA",
    "GOOGL",
    "GOOG",
    "COST",
    "NFLX",
    "AMD",
    "ADBE",
    "PEP",
    "CSCO",
]

STATIC_PRICE_HISTORY = {
    "AAPL": [
        {"date": "2024-01-02", "open": 187.15, "high": 188.44, "low": 183.89, "close": 185.64, "volume": 82488700},
        {"date": "2024-01-03", "open": 184.22, "high": 185.88, "low": 183.43, "close": 184.25, "volume": 58414500},
    ],
    "MSFT": [
        {"date": "2024-01-02", "open": 373.86, "high": 375.90, "low": 366.77, "close": 370.87, "volume": 25258600},
        {"date": "2024-01-03", "open": 369.01, "high": 373.26, "low": 368.51, "close": 370.60, "volume": 23083500},
    ],
}


class USStockDataSource(FinancialDataSource):
    """Financial data source for US-listed equities using public Yahoo Finance endpoints."""

    def __init__(self, session: Optional[requests.Session] = None):
        self.session = session or requests.Session()
        if hasattr(self.session, "headers"):
            self.session.headers.update(
                {
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
                    )
                }
            )

    def get_historical_k_data(
        self,
        code: str,
        start_date: str,
        end_date: str,
        frequency: str = "d",
        adjust_flag: str = "3",
        fields: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        symbol = normalize_us_symbol(code)
        interval = _map_interval(frequency)
        period1 = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
        period2 = int((datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)).timestamp())
        try:
            payload = self._get_json(
                f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
                params={
                    "period1": period1,
                    "period2": period2,
                    "interval": interval,
                    "events": "div,splits",
                    "includeAdjustedClose": "true",
                },
            )
        except DataSourceError as exc:
            logger.warning(f"Yahoo chart lookup failed for {symbol}, trying Stooq fallback: {exc}")
            return self._get_stooq_historical_k_data(symbol, start_date, end_date, adjust_flag, fields)

        chart = payload.get("chart", {})
        if chart.get("error"):
            raise DataSourceError(f"Yahoo Finance chart error for {symbol}: {chart['error']}")

        results = chart.get("result") or []
        if not results:
            raise NoDataFoundError(f"No historical market data found for {symbol}.")

        result = results[0]
        timestamps = result.get("timestamp") or []
        quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
        adjclose = ((result.get("indicators") or {}).get("adjclose") or [{}])[0].get("adjclose") or []

        rows = []
        previous_close = None
        for index, timestamp in enumerate(timestamps):
            close = _list_get(quote.get("close"), index)
            if close is None:
                continue

            open_price = _list_get(quote.get("open"), index)
            high = _list_get(quote.get("high"), index)
            low = _list_get(quote.get("low"), index)
            volume = _list_get(quote.get("volume"), index) or 0
            adjusted_close = _list_get(adjclose, index)
            pct_change = ""
            if previous_close not in (None, 0):
                pct_change = _format_number(((close / previous_close) - 1) * 100)

            row = {
                "date": datetime.fromtimestamp(timestamp, timezone.utc).strftime("%Y-%m-%d"),
                "code": symbol,
                "open": _format_number(open_price),
                "high": _format_number(high),
                "low": _format_number(low),
                "close": _format_number(close),
                "preclose": _format_number(previous_close),
                "volume": _format_number(volume),
                "amount": _format_number(close * volume if close is not None else None),
                "adjustflag": adjust_flag,
                "turn": "",
                "tradestatus": "1",
                "pctChg": pct_change,
                "peTTM": "",
                "pbMRQ": "",
                "psTTM": "",
                "pcfNcfTTM": "",
                "isST": "0",
                "adjClose": _format_number(adjusted_close),
            }
            rows.append(row)
            previous_close = close

        if not rows:
            raise NoDataFoundError(f"No historical market data found for {symbol}.")

        requested_fields = fields or DEFAULT_K_FIELDS
        columns = [field for field in requested_fields if field in rows[0]]
        if "adjClose" in rows[0] and "adjClose" not in columns:
            columns.append("adjClose")
        return pd.DataFrame(rows)[columns]

    def get_stock_basic_info(self, code: str, fields: Optional[List[str]] = None) -> pd.DataFrame:
        symbol = normalize_us_symbol(code)
        quote = self._get_quote(symbol)
        if not quote:
            raise NoDataFoundError(f"No basic info found for {symbol}.")

        row = {
            "code": quote.get("symbol", symbol),
            "code_name": quote.get("shortName") or quote.get("longName") or symbol,
            "ipoDate": "",
            "outDate": "",
            "type": quote.get("quoteType", "EQUITY"),
            "status": quote.get("marketState", "UNKNOWN"),
            "exchange": quote.get("fullExchangeName") or quote.get("exchange", ""),
            "currency": quote.get("currency", "USD"),
            "regularMarketPrice": _format_number(quote.get("regularMarketPrice")),
            "marketCap": _format_number(quote.get("marketCap")),
        }
        df = pd.DataFrame([row])
        if fields:
            selected = [field for field in fields if field in df.columns]
            if not selected:
                raise ValueError(f"None of the requested fields {fields} are available.")
            return df[selected]
        return df

    def get_dividend_data(self, code: str, year: str, year_type: str = "report") -> pd.DataFrame:
        symbol = normalize_us_symbol(code)
        start = f"{year}-01-01"
        end = f"{year}-12-31"
        payload = self._chart_events(symbol, start, end, "div")
        dividends = ((payload.get("events") or {}).get("dividends") or {}).values()
        rows = [
            {
                "code": symbol,
                "dividOperateDate": datetime.fromtimestamp(item["date"], timezone.utc).strftime("%Y-%m-%d"),
                "dividCashPsBeforeTax": _format_number(item.get("amount")),
                "year": year,
            }
            for item in dividends
        ]
        return pd.DataFrame(rows or [{"code": symbol, "year": year, "message": "No dividend data found"}])

    def get_adjust_factor_data(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        symbol = normalize_us_symbol(code)
        payload = self._chart_events(symbol, start_date, end_date, "split")
        splits = ((payload.get("events") or {}).get("splits") or {}).values()
        rows = [
            {
                "code": symbol,
                "date": datetime.fromtimestamp(item["date"], timezone.utc).strftime("%Y-%m-%d"),
                "splitRatio": item.get("splitRatio", ""),
                "numerator": item.get("numerator", ""),
                "denominator": item.get("denominator", ""),
            }
            for item in splits
        ]
        return pd.DataFrame(rows or [{"code": symbol, "message": "No split adjustment events found"}])

    def get_profit_data(self, code: str, year: str, quarter: int) -> pd.DataFrame:
        metrics = self._fundamental_metrics(code)
        return pd.DataFrame(
            [
                {
                    "code": normalize_us_symbol(code),
                    "year": year,
                    "quarter": quarter,
                    "roeAvg": _percent(metrics.get("returnOnEquity")),
                    "npMargin": _percent(metrics.get("profitMargins")),
                    "gpMargin": _percent(metrics.get("grossMargins")),
                    "operatingMargin": _percent(metrics.get("operatingMargins")),
                }
            ]
        )

    def get_operation_data(self, code: str, year: str, quarter: int) -> pd.DataFrame:
        metrics = self._fundamental_metrics(code)
        return pd.DataFrame(
            [
                {
                    "code": normalize_us_symbol(code),
                    "year": year,
                    "quarter": quarter,
                    "revenuePerShare": _format_number(metrics.get("revenuePerShare")),
                    "grossProfits": _format_number(metrics.get("grossProfits")),
                    "ebitdaMargins": _percent(metrics.get("ebitdaMargins")),
                }
            ]
        )

    def get_growth_data(self, code: str, year: str, quarter: int) -> pd.DataFrame:
        metrics = self._fundamental_metrics(code)
        return pd.DataFrame(
            [
                {
                    "code": normalize_us_symbol(code),
                    "year": year,
                    "quarter": quarter,
                    "YOYRevenue": _percent(metrics.get("revenueGrowth")),
                    "YOYNI": _percent(metrics.get("earningsGrowth")),
                    "earningsQuarterlyGrowth": _percent(metrics.get("earningsQuarterlyGrowth")),
                }
            ]
        )

    def get_balance_data(self, code: str, year: str, quarter: int) -> pd.DataFrame:
        metrics = self._fundamental_metrics(code)
        return pd.DataFrame(
            [
                {
                    "code": normalize_us_symbol(code),
                    "year": year,
                    "quarter": quarter,
                    "currentRatio": _format_number(metrics.get("currentRatio")),
                    "debtToEquity": _format_number(metrics.get("debtToEquity")),
                    "totalDebt": _format_number(metrics.get("totalDebt")),
                    "totalCash": _format_number(metrics.get("totalCash")),
                }
            ]
        )

    def get_cash_flow_data(self, code: str, year: str, quarter: int) -> pd.DataFrame:
        metrics = self._fundamental_metrics(code)
        return pd.DataFrame(
            [
                {
                    "code": normalize_us_symbol(code),
                    "year": year,
                    "quarter": quarter,
                    "operatingCashflow": _format_number(metrics.get("operatingCashflow")),
                    "freeCashflow": _format_number(metrics.get("freeCashflow")),
                    "operatingCashflowPerShare": _format_number(metrics.get("operatingCashflowPerShare")),
                }
            ]
        )

    def get_dupont_data(self, code: str, year: str, quarter: int) -> pd.DataFrame:
        metrics = self._fundamental_metrics(code)
        return pd.DataFrame(
            [
                {
                    "code": normalize_us_symbol(code),
                    "year": year,
                    "quarter": quarter,
                    "dupontROE": _percent(metrics.get("returnOnEquity")),
                    "profitMargin": _percent(metrics.get("profitMargins")),
                    "debtToEquity": _format_number(metrics.get("debtToEquity")),
                }
            ]
        )

    def get_performance_express_report(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        metrics = self._fundamental_metrics(code)
        return pd.DataFrame(
            [
                {
                    "code": normalize_us_symbol(code),
                    "start_date": start_date,
                    "end_date": end_date,
                    "earningsDate": metrics.get("earningsDate", ""),
                    "epsForward": _format_number(metrics.get("forwardEps")),
                    "epsTrailingTwelveMonths": _format_number(metrics.get("trailingEps")),
                }
            ]
        )

    def get_forecast_report(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        metrics = self._fundamental_metrics(code)
        return pd.DataFrame(
            [
                {
                    "code": normalize_us_symbol(code),
                    "start_date": start_date,
                    "end_date": end_date,
                    "targetMeanPrice": _format_number(metrics.get("targetMeanPrice")),
                    "recommendationMean": _format_number(metrics.get("recommendationMean")),
                    "recommendationKey": metrics.get("recommendationKey", ""),
                }
            ]
        )

    def get_stock_industry(self, code: Optional[str] = None, date: Optional[str] = None) -> pd.DataFrame:
        if code:
            symbol = normalize_us_symbol(code)
            metrics = self._fundamental_metrics(symbol)
            basic = self.get_stock_basic_info(symbol).iloc[0].to_dict()
            return pd.DataFrame(
                [
                    {
                        "code": symbol,
                        "code_name": basic.get("code_name", symbol),
                        "industry": metrics.get("industry", ""),
                        "sector": metrics.get("sector", ""),
                        "date": date or datetime.now().strftime("%Y-%m-%d"),
                    }
                ]
            )
        return pd.DataFrame(US_STOCK_UNIVERSE)

    def get_dow30_stocks(self, date: Optional[str] = None) -> pd.DataFrame:
        return _constituent_df("Dow Jones Industrial Average", DOW_30, date)

    def get_sp500_stocks(self, date: Optional[str] = None) -> pd.DataFrame:
        return _constituent_df("S&P 500 sample", [item["code"] for item in US_STOCK_UNIVERSE], date)

    def get_nasdaq100_stocks(self, date: Optional[str] = None) -> pd.DataFrame:
        return _constituent_df("Nasdaq-100 sample", NASDAQ_100_SAMPLE, date)

    def get_sz50_stocks(self, date: Optional[str] = None) -> pd.DataFrame:
        return self.get_dow30_stocks(date)

    def get_hs300_stocks(self, date: Optional[str] = None) -> pd.DataFrame:
        return self.get_sp500_stocks(date)

    def get_zz500_stocks(self, date: Optional[str] = None) -> pd.DataFrame:
        return self.get_nasdaq100_stocks(date)

    def get_trade_dates(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        start = datetime.strptime(start_date or "2015-01-01", "%Y-%m-%d")
        end = datetime.strptime(end_date or datetime.now().strftime("%Y-%m-%d"), "%Y-%m-%d")
        holidays = _us_market_holidays(start.year, end.year)
        rows = []
        current = start
        while current <= end:
            date_text = current.strftime("%Y-%m-%d")
            is_trading_day = current.weekday() < 5 and date_text not in holidays
            rows.append({"calendar_date": date_text, "is_trading_day": "1" if is_trading_day else "0"})
            current += timedelta(days=1)
        return pd.DataFrame(rows)

    def get_all_stock(self, date: Optional[str] = None) -> pd.DataFrame:
        rows = []
        for item in US_STOCK_UNIVERSE:
            rows.append(
                {
                    "code": item["code"],
                    "code_name": item["code_name"],
                    "tradeStatus": "1",
                    "exchange": item["exchange"],
                    "date": date or datetime.now().strftime("%Y-%m-%d"),
                }
            )
        return pd.DataFrame(rows)

    def get_deposit_rate_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        return _macro_placeholder("US bank deposit rates", start_date, end_date)

    def get_loan_rate_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        return _macro_placeholder("US prime/lending rates", start_date, end_date)

    def get_required_reserve_ratio_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        year_type: str = "0",
        **kwargs,
    ) -> pd.DataFrame:
        return _macro_placeholder("US reserve requirement ratio", start_date, end_date)

    def get_money_supply_data_month(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        return _macro_placeholder("US money supply monthly", start_date, end_date)

    def get_money_supply_data_year(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        return _macro_placeholder("US money supply yearly", start_date, end_date)

    def crawl_news(self, query: str, top_k: int = 10) -> str:
        symbol_match = re.search(r"\b([A-Za-z]{1,5}(?:[.-][A-Za-z]{1,2})?)\b", query)
        symbol = normalize_us_symbol(symbol_match.group(1)) if symbol_match else None
        results = self._crawl_yahoo_rss(symbol, top_k) if symbol else []
        if not results:
            results = self._crawl_duckduckgo_news(query, top_k)
        if not results:
            return "未找到相关新闻。"

        output = "找到以下相关新闻：\n\n"
        risk_model, risk_tokenizer = self._load_risk_model()
        sentiment_model, sentiment_tokenizer = self._load_sentiment_model()
        for index, result in enumerate(results[:top_k], 1):
            content = result.get("summary") or result.get("title", "")
            risk = self._analyze_risk(content, risk_model, risk_tokenizer) if risk_model else "未分析"
            sentiment = self._analyze_sentiment(content, sentiment_model, sentiment_tokenizer) if sentiment_model else "未分析"
            output += f"{index}. {result.get('title', '')}\n"
            output += f"   来源: {result.get('source', 'Yahoo Finance')}\n"
            output += f"   内容: {content[:300]}\n"
            output += f"   风险分析: {risk}\n"
            output += f"   情感分析: {sentiment}\n"
            output += f"   链接: {result.get('link', '')}\n\n"
        return output

    def _get_json(self, url: str, params: Optional[Dict] = None) -> Dict:
        try:
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            raise DataSourceError(f"Request failed for {url}: {exc}") from exc

    def _get_quote(self, symbol: str) -> Dict:
        try:
            payload = self._get_json("https://query1.finance.yahoo.com/v7/finance/quote", params={"symbols": symbol})
        except DataSourceError as exc:
            logger.warning(f"Yahoo quote lookup failed for {symbol}, using fallback basic info: {exc}")
            return _fallback_quote(symbol)
        results = (payload.get("quoteResponse") or {}).get("result") or []
        return results[0] if results else _fallback_quote(symbol)

    def _get_stooq_historical_k_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        adjust_flag: str,
        fields: Optional[List[str]],
    ) -> pd.DataFrame:
        stooq_symbol = f"{symbol.lower().replace('-', '.')}.us"
        df = self._read_stooq_csv(
            {
                "s": stooq_symbol,
                "d1": start_date.replace("-", ""),
                "d2": end_date.replace("-", ""),
                "i": "d",
            }
        )
        if df.empty or "Close" not in df.columns:
            logger.warning(f"No Stooq data for requested range {start_date} to {end_date}; requesting latest available data for {symbol}.")
            df = self._read_stooq_csv({"s": stooq_symbol, "i": "d"})
        if df.empty or "Close" not in df.columns:
            logger.warning(f"No Stooq historical data found for {symbol}; using static sample data if available.")
            sample_df = _static_price_history_df(symbol, adjust_flag, fields)
            if not sample_df.empty:
                return sample_df
            raise NoDataFoundError(f"No Stooq historical market data found for {symbol}.")

        rows = []
        previous_close = None
        for _, item in df.iterrows():
            close = float(item["Close"])
            volume = float(item.get("Volume", 0) or 0)
            pct_change = ""
            if previous_close not in (None, 0):
                pct_change = _format_number(((close / previous_close) - 1) * 100)
            rows.append(
                {
                    "date": str(item["Date"]),
                    "code": symbol,
                    "open": _format_number(item.get("Open")),
                    "high": _format_number(item.get("High")),
                    "low": _format_number(item.get("Low")),
                    "close": _format_number(close),
                    "preclose": _format_number(previous_close),
                    "volume": _format_number(volume),
                    "amount": _format_number(close * volume),
                    "adjustflag": adjust_flag,
                    "turn": "",
                    "tradestatus": "1",
                    "pctChg": pct_change,
                    "peTTM": "",
                    "pbMRQ": "",
                    "psTTM": "",
                    "pcfNcfTTM": "",
                    "isST": "0",
                    "adjClose": "",
                }
            )
            previous_close = close

        requested_fields = fields or DEFAULT_K_FIELDS
        columns = [field for field in requested_fields if field in rows[0]]
        if "adjClose" not in columns:
            columns.append("adjClose")
        return pd.DataFrame(rows)[columns]

    def _read_stooq_csv(self, params: Dict[str, str]) -> pd.DataFrame:
        try:
            response = self.session.get("https://stooq.com/q/d/l/", params=params, timeout=15)
            response.raise_for_status()
            return pd.read_csv(StringIO(response.text))
        except (pd.errors.EmptyDataError, Exception) as exc:
            logger.warning(f"Stooq CSV request failed or returned no data: {exc}")
            return pd.DataFrame()

    def _quote_summary(self, symbol: str) -> Dict:
        modules = "price,summaryProfile,defaultKeyStatistics,financialData,summaryDetail,calendarEvents"
        try:
            payload = self._get_json(
                f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{symbol}",
                params={"modules": modules},
            )
        except DataSourceError:
            return {}
        results = ((payload.get("quoteSummary") or {}).get("result") or [{}])[0]
        return results or {}

    def _fundamental_metrics(self, code: str) -> Dict:
        symbol = normalize_us_symbol(code)
        quote = self._get_quote(symbol)
        summary = self._quote_summary(symbol)
        metrics = {}
        for section_name in ["price", "summaryProfile", "defaultKeyStatistics", "financialData", "summaryDetail"]:
            section = summary.get(section_name) or {}
            for key, value in section.items():
                metrics[key] = _raw_value(value)
        metrics.update({key: value for key, value in quote.items() if key not in metrics})
        calendar = summary.get("calendarEvents") or {}
        earnings = calendar.get("earnings") or {}
        earnings_dates = earnings.get("earningsDate") or []
        if earnings_dates:
            metrics["earningsDate"] = _raw_value(earnings_dates[0])
        return metrics

    def _chart_events(self, symbol: str, start_date: str, end_date: str, events: str) -> Dict:
        period1 = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
        period2 = int((datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)).timestamp())
        payload = self._get_json(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
            params={"period1": period1, "period2": period2, "interval": "1d", "events": events},
        )
        results = (payload.get("chart") or {}).get("result") or []
        return results[0] if results else {}

    def _crawl_yahoo_rss(self, symbol: str, top_k: int) -> List[Dict]:
        try:
            response = self.session.get(
                "https://feeds.finance.yahoo.com/rss/2.0/headline",
                params={"s": symbol, "region": "US", "lang": "en-US"},
                timeout=15,
            )
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "xml")
            results = []
            for item in soup.find_all("item")[:top_k]:
                results.append(
                    {
                        "title": item.title.get_text(strip=True) if item.title else "",
                        "summary": item.description.get_text(strip=True) if item.description else "",
                        "link": item.link.get_text(strip=True) if item.link else "",
                        "source": "Yahoo Finance RSS",
                    }
                )
            return results
        except Exception as exc:
            logger.warning(f"Yahoo RSS news lookup failed for {symbol}: {exc}")
            return []

    def _crawl_duckduckgo_news(self, query: str, top_k: int) -> List[Dict]:
        try:
            response = self.session.get(
                "https://duckduckgo.com/html/",
                params={"q": f"{query} stock news"},
                timeout=15,
            )
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            results = []
            for result in soup.select(".result")[:top_k]:
                link = result.select_one(".result__a")
                snippet = result.select_one(".result__snippet")
                if link:
                    results.append(
                        {
                            "title": link.get_text(strip=True),
                            "summary": snippet.get_text(strip=True) if snippet else "",
                            "link": link.get("href", ""),
                            "source": "DuckDuckGo",
                        }
                    )
            return results
        except Exception as exc:
            logger.warning(f"DuckDuckGo news lookup failed for {query}: {exc}")
            return []

    def _load_risk_model(self):
        if not _local_news_models_enabled():
            return None, None
        try:
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            risk_model_path = os.getenv("QWEN_RISK_MODEL", "")
            base_model_name = os.getenv("QWEN_BASE_MODEL", "")
            if not risk_model_path or not base_model_name:
                logger.info("QWEN_RISK_MODEL / QWEN_BASE_MODEL not set; skipping local risk model.")
                return None, None
            device = "cuda" if torch.cuda.is_available() else "cpu"
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            tokenizer.pad_token = tokenizer.eos_token
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
            )
            risk_model = PeftModel.from_pretrained(base_model, risk_model_path)
            if device == "cpu":
                risk_model = risk_model.to(device)
            return risk_model, tokenizer
        except Exception as exc:
            logger.error(f"Error loading risk model: {exc}")
            return None, None

    def _load_sentiment_model(self):
        if not _local_news_models_enabled():
            return None, None
        try:
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            sentiment_model_path = os.getenv("QWEN_SENTIMENT_MODEL", "")
            base_model_name = os.getenv("QWEN_BASE_MODEL", "")
            if not sentiment_model_path or not base_model_name:
                logger.info("QWEN_SENTIMENT_MODEL / QWEN_BASE_MODEL not set; skipping local sentiment model.")
                return None, None
            device = "cuda" if torch.cuda.is_available() else "cpu"
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            tokenizer.pad_token = tokenizer.eos_token
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
            )
            sentiment_model = PeftModel.from_pretrained(base_model, sentiment_model_path)
            if device == "cpu":
                sentiment_model = sentiment_model.to(device)
            return sentiment_model, tokenizer
        except Exception as exc:
            logger.error(f"Error loading sentiment model: {exc}")
            return None, None

    def _analyze_risk(self, content: str, model, tokenizer) -> str:
        try:
            if model is None or tokenizer is None:
                return "模型未加载"
            import torch

            device = next(model.parameters()).device
            prompt = f"""System: You are a financial expert specializing in risk assessment for stock recommendations. Based on a specific stock, provide a risk score from 1 to 5.

User: News to Stock Symbol -- AAPL: Apple (AAPL) increases 22%
Assistant: 3

User: News to Stock Symbol -- AAPL: Apple (AAPL) price decreased 30%
Assistant: 4

User: News to Stock Symbol -- STOCK: {content}
Assistant:"""
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False, temperature=0.1, pad_token_id=tokenizer.eos_token_id)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            score = int(generated_text.split("Assistant:")[-1].strip().split()[0])
            risk_map = {1: "极低风险", 2: "低风险", 3: "中等风险", 4: "高风险", 5: "极高风险"}
            return f"{score} ({risk_map[score]})" if score in risk_map else "无法分析风险"
        except Exception as exc:
            logger.error(f"Error during risk analysis: {exc}")
            return f"风险分析失败: {exc}"

    def _analyze_sentiment(self, content: str, model, tokenizer) -> str:
        try:
            if model is None or tokenizer is None:
                return "模型未加载"
            import torch

            device = next(model.parameters()).device
            prompt = f"""System: You are a financial expert with stock recommendation experience. Based on a specific stock, score sentiment from 1 to 5.

User: News to Stock Symbol -- AAPL: Apple (AAPL) increase 22%
Assistant: 5

User: News to Stock Symbol -- AAPL: Apple (AAPL) price decreased 30%
Assistant: 1

User: News to Stock Symbol -- STOCK: {content}
Assistant:"""
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False, temperature=0.1, pad_token_id=tokenizer.eos_token_id)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            score = int(generated_text.split("Assistant:")[-1].strip().split()[0])
            sentiment_map = {1: "负面", 2: "轻微负面", 3: "中性", 4: "正面", 5: "极正面"}
            return f"{score} ({sentiment_map[score]})" if score in sentiment_map else "无法分析情感"
        except Exception as exc:
            logger.error(f"Error during sentiment analysis: {exc}")
            return f"情感分析失败: {exc}"


def normalize_us_symbol(code: str) -> str:
    candidate = (code or "").strip().upper().replace(".", "-")
    if not re.fullmatch(r"[A-Z]{1,5}(?:-[A-Z]{1,2})?", candidate):
        raise ValueError(f"US stock ticker must be letters with optional class suffix, got: {code}")
    return candidate


def _map_interval(frequency: str) -> str:
    mapping = {"d": "1d", "w": "1wk", "m": "1mo", "5": "5m", "15": "15m", "30": "30m", "60": "60m"}
    if frequency not in mapping:
        raise ValueError(f"Invalid frequency '{frequency}'. Valid options are: {list(mapping)}")
    return mapping[frequency]


def _list_get(values, index):
    if not values or index >= len(values):
        return None
    value = values[index]
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return value


def _format_number(value) -> str:
    value = _raw_value(value)
    if value is None or value == "":
        return ""
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return ""
        return str(round(value, 4))
    return str(value)


def _percent(value) -> str:
    raw = _raw_value(value)
    if raw in (None, ""):
        return ""
    return _format_number(raw * 100 if abs(raw) <= 1 else raw)


def _raw_value(value):
    if isinstance(value, dict):
        if "raw" in value:
            return value["raw"]
        if "fmt" in value:
            return value["fmt"]
    return value


def _constituent_df(index_name: str, symbols: List[str], date: Optional[str]) -> pd.DataFrame:
    rows = [{"index": index_name, "code": symbol, "date": date or datetime.now().strftime("%Y-%m-%d")} for symbol in symbols]
    return pd.DataFrame(rows)


def _fallback_quote(symbol: str) -> Dict:
    known = next((item for item in US_STOCK_UNIVERSE if item["code"] == symbol), None)
    return {
        "symbol": symbol,
        "shortName": known["code_name"] if known else symbol,
        "exchange": known["exchange"] if known else "",
        "currency": "USD",
        "quoteType": "EQUITY",
        "marketState": "UNKNOWN",
    }


def _static_price_history_df(symbol: str, adjust_flag: str, fields: Optional[List[str]]) -> pd.DataFrame:
    history = STATIC_PRICE_HISTORY.get(symbol, [])
    if not history:
        return pd.DataFrame()

    rows = []
    previous_close = None
    for item in history:
        close = item["close"]
        volume = item["volume"]
        pct_change = ""
        if previous_close not in (None, 0):
            pct_change = _format_number(((close / previous_close) - 1) * 100)
        rows.append(
            {
                "date": item["date"],
                "code": symbol,
                "open": _format_number(item["open"]),
                "high": _format_number(item["high"]),
                "low": _format_number(item["low"]),
                "close": _format_number(close),
                "preclose": _format_number(previous_close),
                "volume": _format_number(volume),
                "amount": _format_number(close * volume),
                "adjustflag": adjust_flag,
                "turn": "",
                "tradestatus": "1",
                "pctChg": pct_change,
                "peTTM": "",
                "pbMRQ": "",
                "psTTM": "",
                "pcfNcfTTM": "",
                "isST": "0",
                "adjClose": "",
                "dataSource": "static_sample",
            }
        )
        previous_close = close

    requested_fields = fields or DEFAULT_K_FIELDS
    columns = [field for field in requested_fields if field in rows[0]]
    for optional in ["adjClose", "dataSource"]:
        if optional not in columns:
            columns.append(optional)
    return pd.DataFrame(rows)[columns]


def _macro_placeholder(metric: str, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "metric": metric,
                "start_date": start_date or "",
                "end_date": end_date or "",
                "data_source": "Not configured",
                "note": "No API key is required for stock analysis; configure a macro data provider for this series if needed.",
            }
        ]
    )


def _us_market_holidays(start_year: int, end_year: int) -> set:
    holidays = set()
    for year in range(start_year, end_year + 1):
        holidays.update(
            {
                _observed_holiday(datetime(year, 1, 1)).strftime("%Y-%m-%d"),
                _nth_weekday(year, 1, 0, 3).strftime("%Y-%m-%d"),
                _nth_weekday(year, 2, 0, 3).strftime("%Y-%m-%d"),
                _good_friday(year).strftime("%Y-%m-%d"),
                _last_weekday(year, 5, 0).strftime("%Y-%m-%d"),
                _observed_holiday(datetime(year, 6, 19)).strftime("%Y-%m-%d"),
                _observed_holiday(datetime(year, 7, 4)).strftime("%Y-%m-%d"),
                _nth_weekday(year, 9, 0, 1).strftime("%Y-%m-%d"),
                _nth_weekday(year, 11, 3, 4).strftime("%Y-%m-%d"),
                _observed_holiday(datetime(year, 12, 25)).strftime("%Y-%m-%d"),
            }
        )
    return holidays


def _observed_holiday(date: datetime) -> datetime:
    if date.weekday() == 5:
        return date - timedelta(days=1)
    if date.weekday() == 6:
        return date + timedelta(days=1)
    return date


def _nth_weekday(year: int, month: int, weekday: int, occurrence: int) -> datetime:
    date = datetime(year, month, 1)
    while date.weekday() != weekday:
        date += timedelta(days=1)
    return date + timedelta(weeks=occurrence - 1)


def _last_weekday(year: int, month: int, weekday: int) -> datetime:
    date = datetime(year, month + 1, 1) - timedelta(days=1) if month < 12 else datetime(year, 12, 31)
    while date.weekday() != weekday:
        date -= timedelta(days=1)
    return date


def _good_friday(year: int) -> datetime:
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return datetime(year, month, day) - timedelta(days=2)
