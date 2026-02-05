import os
from typing import Any

import httpx
from dotenv import load_dotenv

from src.schemas import FinancialStatement, KeyMetrics, MarketNews, StockProfile

# Load environment variables from .env file
load_dotenv()


class FMPClient:
    """
    Async client for the Financial Modeling Prep (FMP) API.
    Includes fallback logic for restricted/legacy endpoints on new accounts.
    """

    BASE_URL = "https://financialmodelingprep.com/stable"

    def __init__(self):
        self.api_key = os.getenv("FMP_API_KEY")
        if not self.api_key:
            raise ValueError("FMP_API_KEY environment variable is not set")

        # Initialize async client with a timeout
        self.client = httpx.AsyncClient(timeout=10.0)

    async def close(self):
        """Closes the underlying HTTP client session."""
        await self.client.aclose()

    async def _get(self, endpoint: str, params: dict | None = None) -> Any:
        """
        Internal helper method to execute GET requests with error handling.
        """
        req_params = params.copy() if params else {}
        req_params["apikey"] = self.api_key
        url = f"{self.BASE_URL}/{endpoint}"

        try:
            response = await self.client.get(url, params=req_params)
            response.raise_for_status()  # Raises exception for 4xx/5xx codes
            return response.json()

        except httpx.HTTPStatusError as e:
            # Print detailed error from FMP (e.g., "Legacy Endpoint")
            print(f"❌ HTTP Error {e.response.status_code} at endpoint '{endpoint}': {e.response.text}")
            return []
        except Exception as e:
            print(f"❌ Request Error: {e}")
            return []

    async def get_profile(self, ticker: str) -> StockProfile | None:
        """
        Fetches company profile.
        FALLBACK STRATEGY:
        If the standard 'profile' endpoint returns 403 (Legacy User restriction),
        it attempts to fetch basic data from the 'quote' endpoint instead.
        """
        # 1. Try the standard profile endpoint (stable API uses query params)
        endpoint = "profile"
        data = await self._get(endpoint, {"symbol": ticker})

        if isinstance(data, list) and len(data) > 0:
            return StockProfile(**data[0])

        # 2. Fallback: Try 'quote' endpoint if profile failed or returned empty
        print("⚠️  'Profile' endpoint failed or empty. Attempting fallback to 'Quote'...")
        quote_endpoint = "quote"
        quote_data = await self._get(quote_endpoint, {"symbol": ticker})

        if isinstance(quote_data, list) and len(quote_data) > 0:
            q = quote_data[0]
            # Manual mapping from Quote data to StockProfile model
            return StockProfile(
                symbol=q.get("symbol"),
                companyName=q.get("name"),  # 'quote' uses 'name', 'profile' uses 'companyName'
                price=q.get("price"),
                mktCap=q.get("marketCap"),
                description="Description unavailable (Source: Quote Endpoint)",
                sector="N/A",
                industry="N/A",
                ceo="N/A",
                website="N/A",
            )

        return None

    async def get_key_metrics(self, ticker: str) -> KeyMetrics | None:
        """
        Fetches key financial metrics (TTM) using stable API.
        Combines data from ratios-ttm (PE, EPS, Debt/Equity, Dividend Yield)
        and key-metrics-ttm (ROE) for comprehensive metrics.
        """
        import asyncio

        # Fetch from both endpoints in parallel
        ratios_data, metrics_data = await asyncio.gather(
            self._get("ratios-ttm", {"symbol": ticker}),
            self._get("key-metrics-ttm", {"symbol": ticker}),
        )

        # Merge the data (ratios-ttm as base, add ROE from key-metrics-ttm)
        merged = {}
        if isinstance(ratios_data, list) and len(ratios_data) > 0:
            merged.update(ratios_data[0])
        if isinstance(metrics_data, list) and len(metrics_data) > 0:
            merged.update(metrics_data[0])

        if merged:
            return KeyMetrics(**merged)
        return None

    async def get_news(self, ticker: str, limit: int = 5) -> list[MarketNews]:
        """
        Fetches stock market news using stable API.
        """
        endpoint = "news/stock"
        params = {"symbols": ticker, "limit": limit}
        data = await self._get(endpoint, params)

        news_list = []
        if isinstance(data, list):
            for item in data:
                # Validate and append only valid news items
                try:
                    news_list.append(MarketNews(**item))
                except Exception:
                    # Skip items that don't match the schema
                    continue

        return news_list

    async def get_financial_statements(
        self, ticker: str, statement_type: str, limit: int = 4
    ) -> list[FinancialStatement]:
        """
        Fetches financial statements (Income, Balance Sheet, Cash Flow).
        Returns a list of Pydantic models with dot-notation access.
        """
        endpoint = statement_type
        params = {"symbol": ticker, "limit": limit}

        data = await self._get(endpoint, params)

        statements = []
        if isinstance(data, list):
            for item in data:
                try:
                    statements.append(FinancialStatement(**item))
                except Exception:
                    continue

        return statements
