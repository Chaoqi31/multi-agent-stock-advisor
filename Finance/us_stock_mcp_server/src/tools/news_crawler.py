"""
News crawler tool module.
Provides news search and crawling functionality.
"""

import logging
from typing import List, Dict
from mcp.server.fastmcp import FastMCP
from ..data_source_interface import FinancialDataSource

logger = logging.getLogger(__name__)

def register_news_crawler_tools(app: FastMCP, data_source: FinancialDataSource):
    """
    Register the news crawler tool.

    Args:
        app: The FastMCP app instance
        data_source: The data source instance
    """

    @app.tool()
    def crawl_news(query: str, top_k: int = 10) -> str:
        """
        Crawl relevant news.

        Uses Yahoo Finance RSS and web search to crawl news articles related to the query, and returns formatted results.

        Args:
            query: Search query, e.g. "AAPL", "Apple earnings", etc.
            top_k: Number of news items to return; defaults to 10

        Returns:
            A formatted news result string containing titles, content summaries, links, and other information

        Example:
            >>> crawl_news("AAPL", 5)
            "找到以下相关新闻：

            1. Apple reports quarterly earnings
               来源: Yahoo Finance RSS
               内容: Apple reported quarterly earnings...
               链接: https://example.com/news/123
            "
        """
        try:
            logger.info(f"Starting news crawl, query: {query}, count: {top_k}")
            result = data_source.crawl_news(query, top_k)
            logger.info(f"News crawl complete, result length: {len(result)}")
            return result
        except Exception as e:
            logger.error(f"Error while crawling news: {e}")
            return f"爬取新闻时出错: {str(e)}"
    
    
