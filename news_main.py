import asyncio
import sys

from langchain_core.messages import HumanMessage

from src.news_graph import news_app
from src.news_logger import NewsLogger


async def main():
    # Accept ticker from CLI args, default to NVDA
    ticker = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
    print(f"📰 Starting News Analysis for: {ticker}...\n")

    # 1. Init Logger
    logger = NewsLogger(ticker)

    # 2. Init State
    initial_state = {
        "messages": [HumanMessage(content=f"Gather the latest news for {ticker} stock.")],
        "ticker": ticker,
    }

    # Track logged messages to avoid duplicates
    logged_count = 0

    # 3. Run Graph
    async for event in news_app.astream(initial_state, stream_mode="values"):
        messages = event.get("messages")
        if messages:
            new_messages = messages[logged_count:]

            for msg in new_messages:
                if isinstance(msg, HumanMessage):
                    continue

                logger.log(msg)

                print(f"✅ Processed: {type(msg).__name__}")
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    tool_names = [tc["name"] for tc in msg.tool_calls]
                    print(f"   -> Calling: {', '.join(tool_names)}")

            logged_count = len(messages)

    # 4. Save
    logger.flush()
    print("\n🎉 News Analysis Complete!")
    print(f"   📄 JSON: logs/{logger.json_filename}")
    print(f"   📝 MD:   logs/{logger.md_filename}")


if __name__ == "__main__":
    asyncio.run(main())
