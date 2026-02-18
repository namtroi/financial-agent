import asyncio
import json
import os
from datetime import datetime

from src.client import FMPClient


async def main():
    ticker = "AAPL"
    limit = 50

    print(f"--- Fetching news for {ticker} (limit={limit}) ---\n")

    client = FMPClient()

    try:
        news = await client.get_news(ticker, limit=limit)

        if not news:
            print("❌ No news found.")
            return

        # --- Console summary ---
        print(f"✅ Fetched {len(news)} articles\n")
        for idx, item in enumerate(news, 1):
            print(f"  {idx}. [{item.date}] {item.title}")
            print(f"     {item.url}\n")

        # --- Save to logs/ ---
        os.makedirs("logs", exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"logs/news_{ticker}_{timestamp}.json"

        output = {
            "ticker": ticker,
            "fetched_at": datetime.now().isoformat(),
            "total_articles": len(news),
            "articles": [article.model_dump(by_alias=True) for article in news],
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"💾 Saved to {filename}")

    except Exception as e:
        print(f"\n❌ Error: {e}")

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
