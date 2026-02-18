from typing import Any

from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

from src.client import FMPClient


# --- Input Schema ---
class TickerInput(BaseModel):
    """Input schema for news tools."""

    ticker: str = Field(description="The stock ticker symbol (e.g., AAPL, TSLA, NVDA).")


# --- Tools ---


@tool("fetch_press_releases", args_schema=TickerInput)
async def fetch_press_releases(ticker: str) -> list[dict[str, Any]]:
    """
    Fetches official company press releases (earnings, contracts, mergers, dividends).
    Returns the 10 most recent press releases with full text content.
    This is the PRIMARY source for corporate news and announcements.
    """
    client = FMPClient()
    try:
        releases = await client.get_press_releases(ticker, limit=10)
        if releases:
            return [r.model_dump() for r in releases]
        return [{"error": "No press releases found."}]
    except Exception as e:
        return [{"error": str(e)}]
    finally:
        await client.close()


# Tavily search tool — use "general" for broader coverage (better for small/mid-cap tickers)
tavily_news_tool = TavilySearch(max_results=5, topic="general", include_raw_content=True)


class SearchInput(BaseModel):
    """Input schema for search tool."""

    query: str = Field(
        description="A detailed search query including the company name AND ticker "
        "(e.g., 'Tutor Perini TPC latest contract wins earnings news 2025 2026')."
    )


@tool("search_company_news", args_schema=SearchInput)
async def search_company_news(query: str) -> Any:
    """
    Searches for the latest news, analysis, and market commentary about the company.
    Covers third-party sources: Reuters, Seeking Alpha, CNBC, PR Newswire, etc.
    COMPLEMENTS press releases with external analyst perspectives and market reactions.
    IMPORTANT: Include the FULL COMPANY NAME in the query for better results, not just the ticker.
    """
    result = await tavily_news_tool.ainvoke({"query": query})
    return result
