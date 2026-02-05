import asyncio
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.client import FMPClient
from src.schemas import KeyMetrics, MarketNews, StockProfile


# --- 1. Define Input Schemas (What the LLM sends to the tool) ---
class TickerInput(BaseModel):
    """Input schema for stock ticker operations."""

    ticker: str = Field(description="The stock ticker symbol (e.g., AAPL, TSLA, NVDA).")


# --- 2. Define Tools (The functions the LLM can call) ---


@tool("get_company_profile", args_schema=TickerInput)
async def get_company_profile(ticker: str) -> dict[str, Any]:
    """
    Fetches the company profile (CEO, description, sector, price, market cap).
    Useful for understanding what the company does and its general standing.
    """
    client = FMPClient()
    try:
        profile: StockProfile | None = await client.get_profile(ticker)
        if profile:
            # Return model as dictionary for the LLM to read
            return profile.model_dump(by_alias=True)
        return {"error": f"Company profile not found for ticker: {ticker}"}
    except Exception as e:
        return {"error": str(e)}
    finally:
        await client.close()


@tool("get_financial_ratios", args_schema=TickerInput)
async def get_financial_ratios(ticker: str) -> dict[str, Any]:
    """
    Fetches key financial ratios (PE, EPS, ROE, Debt/Equity) for the trailing twelve months (TTM).
    Useful for evaluating valuation and profitability.
    """
    client = FMPClient()
    try:
        metrics: KeyMetrics | None = await client.get_key_metrics(ticker)
        if metrics:
            return metrics.model_dump(by_alias=True)
        return {"error": f"Financial metrics not found for ticker: {ticker}"}
    except Exception as e:
        return {"error": str(e)}
    finally:
        await client.close()


@tool("get_stock_news", args_schema=TickerInput)
async def get_stock_news(ticker: str) -> list[dict[str, Any]]:
    """
    Fetches the latest market news related to the stock.
    Useful for sentiment analysis and identifying recent events.
    """
    client = FMPClient()
    try:
        news_list: list[MarketNews] | None = await client.get_news(ticker, limit=5)
        if news_list:
            return [news.model_dump(by_alias=True) for news in news_list]
        return [{"error": "No recent news found."}]
    except Exception as e:
        return [{"error": str(e)}]
    finally:
        await client.close()


@tool("get_financial_statements", args_schema=TickerInput)
async def get_financial_statements(ticker: str) -> dict[str, Any]:
    """
    Fetches the Income Statement, Balance Sheet, and Cash Flow Statement for the last 4 years.
    CRITICAL for valuation (DCF), margin analysis, and calculating growth rates.
    """
    client = FMPClient()
    try:
        # Execute 3 API calls in parallel to reduce latency
        income, balance, cash = await asyncio.gather(
            client.get_financial_statements(ticker, "income-statement"),
            client.get_financial_statements(ticker, "balance-sheet-statement"),
            client.get_financial_statements(ticker, "cash-flow-statement"),
        )

        if income or balance or cash:
            return {
                "income_statement": [stmt.model_dump(by_alias=True) for stmt in income] if income else [],
                "balance_sheet": [stmt.model_dump(by_alias=True) for stmt in balance] if balance else [],
                "cash_flow": [stmt.model_dump(by_alias=True) for stmt in cash] if cash else [],
            }
        return {"error": f"Financial statements not found for ticker: {ticker}"}
        
    except Exception as e:
        return {"error": f"Failed to fetch financials: {str(e)}"}
    finally:
        await client.close()
