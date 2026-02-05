import asyncio

from src.client import FMPClient


async def main():
    ticker = "AAPL"
    print(f"--- Fetching data for {ticker} ---")

    client = FMPClient()

    try:
        # ---------------------------------------------------------
        # 1. Test Profile
        # ---------------------------------------------------------
        print("\n[1] Fetching Profile...")
        profile = await client.get_profile(ticker)
        if profile:
            print(f"Success: {profile.company_name} | CEO: {profile.ceo} | Price: ${profile.price}")
            # Debug: Print full object data
            # print(profile.model_dump())
        else:
            print("Failed to fetch profile.")

        # ---------------------------------------------------------
        # 2. Test Key Metrics
        # ---------------------------------------------------------
        print("\n[2] Fetching Key Metrics...")
        metrics = await client.get_key_metrics(ticker)
        if metrics:
            print(f"Success: PE: {metrics.pe_ratio} | EPS: {metrics.eps} | ROE: {metrics.roe}")

            # Using model_dump() to iterate over fields safely
            # for key, value in metrics.model_dump().items():
            #     print(f"  {key}: {value}")
        else:
            print("Failed to fetch metrics.")

        # ---------------------------------------------------------
        # 3. Test News
        # ---------------------------------------------------------
        print("\n[3] Fetching News...")
        news = await client.get_news(ticker, limit=2)
        if news:
            for idx, item in enumerate(news, 1):
                print(f"  {idx}. {item.title} ({item.date})")
                print(f"     URL: {item.url}")
        else:
            print("No news found.")

        # ---------------------------------------------------------
        # 4. Test Financial Statements
        # ---------------------------------------------------------
        print("\n[4] Fetching Financial Statements...")

        statement_types = ["income-statement", "balance-sheet-statement", "cash-flow-statement"]

        for stmt_type in statement_types:
            print(f"  > Fetching {stmt_type}...")

            statements = await client.get_financial_statements(ticker, statement_type=stmt_type, limit=1)

            if statements:
                latest = statements[0]

                print(f"    ✅ Success: Period {latest.period} ({latest.date})")

                if stmt_type == "income-statement":
                    rev = getattr(latest, "revenue", 0)
                    net = getattr(latest, "netIncome", 0)
                    print(f"       Revenue: ${rev:,}")
                    print(f"       Net Income: ${net:,}")

                elif stmt_type == "balance-sheet-statement":
                    assets = getattr(latest, "totalAssets", 0)
                    debt = getattr(latest, "totalDebt", 0)
                    print(f"       Total Assets: ${assets:,}")
                    print(f"       Total Debt: ${debt:,}")

                elif stmt_type == "cash-flow-statement":
                    ocf = getattr(latest, "operatingCashFlow", 0)
                    print(f"       Operating Cash Flow: ${ocf:,}")
            else:
                print(f"    ❌ Failed or empty data for {stmt_type}")

    except Exception as e:
        print(f"\nAn error occurred in main execution: {e}")

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
