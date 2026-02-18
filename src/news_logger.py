import json
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage


class NewsLogger:
    """
    Logger for the News Analysis module.
    Saves raw news data + analyzed articles to JSON and MD files.
    """

    def __init__(self, ticker: str, log_dir: Path | None = None):
        self.ticker = ticker
        self.log_dir = log_dir or Path(__file__).parent.parent / "logs"
        self.log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.json_filename = f"NEWS_{ticker}_{timestamp}.json"
        self.md_filename = f"NEWS_{ticker}_{timestamp}.md"

        self.json_path = self.log_dir / self.json_filename
        self.md_path = self.log_dir / self.md_filename

        # Data structure
        self.data: dict[str, Any] = {
            "metadata": {
                "ticker": ticker,
                "timestamp": datetime.now().isoformat(),
                "tools_called": [],
            },
            "raw_news": {
                "press_releases": [],
                "search_results": [],
            },
            "analyzed_articles": None,
        }

        # Track tool call IDs to tool names
        self._tool_call_map: dict[str, str] = {}

    def log(self, message: BaseMessage) -> None:
        """Process a LangChain message and extract relevant data."""

        if isinstance(message, AIMessage):
            # Track tool calls for mapping
            if hasattr(message, "tool_calls") and message.tool_calls:
                for tc in message.tool_calls:
                    self._tool_call_map[tc["id"]] = tc["name"]

            # If this is the final analyst output (no tool calls, has content)
            if message.content and not getattr(message, "tool_calls", None):
                self.data["analyzed_articles"] = message.content
                self._save_md(message.content)

        elif isinstance(message, ToolMessage):
            tool_name = self._tool_call_map.get(message.tool_call_id, "unknown")

            if tool_name not in self.data["metadata"]["tools_called"]:
                self.data["metadata"]["tools_called"].append(tool_name)

            # Parse tool output
            try:
                content = json.loads(message.content) if isinstance(message.content, str) else message.content
            except (json.JSONDecodeError, TypeError):
                content = message.content

            # Store in appropriate section
            if tool_name == "fetch_press_releases":
                self.data["raw_news"]["press_releases"] = content
            elif tool_name == "search_company_news":
                self.data["raw_news"]["search_results"] = content

    def flush(self) -> None:
        """Save all data to JSON file."""
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False, default=str)

    def _save_md(self, content: str) -> None:
        """Save the analyzed articles as markdown."""
        header = f"# News Analysis: {self.ticker}\n"
        header += f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"

        with open(self.md_path, "w", encoding="utf-8") as f:
            f.write(header + content + "\n")
