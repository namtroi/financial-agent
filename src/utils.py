import json
import os
from datetime import datetime
from typing import Any, Dict, List, Union

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage


class MarkdownLogger:
    """
    A Dual-Logger that streams agent interactions to a readable Markdown file
    while simultaneously archiving the full raw context into a structured JSON file.

    - Markdown (.md): For human readability - contains final report + trace (no raw data)
    - JSON (.json): For machine processing - contains metadata, messages, and extracted data
    """

    def __init__(self, ticker: str):
        # Ensure the logs directory exists
        os.makedirs("logs", exist_ok=True)

        # Create a unique timestamped base filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_filename = f"logs/{ticker}_RESEARCH_{timestamp}"
        self.ticker = ticker
        self.start_time = datetime.now().isoformat()

        # Initialize filenames
        self.md_filename = f"{self.base_filename}.md"
        self.json_filename = f"{self.base_filename}.json"

        # Initialize the Markdown file with a header
        with open(self.md_filename, "w", encoding="utf-8") as f:
            f.write(f"# 🕵️‍♂️ Financial Research Log: ${ticker}\n")
            f.write(f"*Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
            f.write(
                f"*Raw Data Context:* [`{os.path.basename(self.json_filename)}`]"
                f"(./{os.path.basename(self.json_filename)})\n\n"
            )
            f.write("---\n\n")

        # Initialize the structured JSON data
        self.json_data: Dict[str, Any] = {
            "metadata": {
                "ticker": ticker,
                "timestamp": self.start_time,
                "tools_called": [],
            },
            "raw_messages": [],
            "extracted_data": {
                "profile": None,
                "metrics": None,
                "news": [],
                "income_statement": [],
                "balance_sheet": [],
                "cash_flow": [],
            },
        }

        print(
            f"📝 Dual Logger initialized:\n   - Human Log: {self.md_filename}\n   - Machine Data: {self.json_filename}"
        )

    def _format_json(self, data: Union[str, Dict, List]) -> str:
        """Helper to pretty-print JSON objects for the Markdown report."""
        if isinstance(data, str):
            try:
                parsed = json.loads(data)
                return json.dumps(parsed, indent=2)
            except ValueError:
                return data
        return json.dumps(data, indent=2)

    def _extract_tool_data(self, tool_name: str, content: Any):
        """Extract and store tool output data in the structured extracted_data section."""
        if not isinstance(content, (dict, list)):
            # Try to parse if it's a JSON string
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except (ValueError, TypeError):
                    return

        # Map tool outputs to extracted_data fields
        if tool_name == "get_company_profile":
            self.json_data["extracted_data"]["profile"] = content
        elif tool_name == "get_financial_ratios":
            self.json_data["extracted_data"]["metrics"] = content
        elif tool_name == "get_stock_news":
            self.json_data["extracted_data"]["news"] = content if isinstance(content, list) else [content]
        elif tool_name == "get_financial_statements":
            if isinstance(content, dict):
                if "income_statement" in content:
                    self.json_data["extracted_data"]["income_statement"] = content["income_statement"]
                if "balance_sheet" in content:
                    self.json_data["extracted_data"]["balance_sheet"] = content["balance_sheet"]
                if "cash_flow" in content:
                    self.json_data["extracted_data"]["cash_flow"] = content["cash_flow"]

    def _save_json(self):
        """Save the structured JSON data to disk."""
        with open(self.json_filename, "w", encoding="utf-8") as f:
            json.dump(self.json_data, f, indent=2, ensure_ascii=False)

    def log(self, message: BaseMessage):
        """
        Processes a LangChain message:
        1. Appends the raw message data to the JSON context file.
        2. Formats and writes the message to the Markdown log file.
        """

        # --- PART A: JSON Archiving (Machine Readable) ---

        # Extract content (handle cases where content is a JSON string)
        msg_content = message.content
        if isinstance(message, ToolMessage):
            try:
                msg_content = json.loads(message.content)
            except Exception:
                pass  # Keep as string if parsing fails

        # Construct the log entry for raw_messages
        entry = {"type": type(message).__name__, "timestamp": datetime.now().isoformat(), "content": msg_content}

        # Capture tool calls if present (for AI reasoning steps)
        if hasattr(message, "tool_calls") and message.tool_calls:
            entry["tool_calls"] = message.tool_calls
            # Track tools called in metadata
            for tool in message.tool_calls:
                tool_name = tool.get("name", "")
                if tool_name and tool_name not in self.json_data["metadata"]["tools_called"]:
                    self.json_data["metadata"]["tools_called"].append(tool_name)

        # Capture tool call ID if present (for Tool outputs)
        if hasattr(message, "tool_call_id"):
            entry["tool_call_id"] = message.tool_call_id

        # Update raw_messages
        self.json_data["raw_messages"].append(entry)

        # Extract data from ToolMessage for structured storage
        if isinstance(message, ToolMessage):
            # Find which tool this output belongs to
            tool_name = self._get_tool_name_from_id(message.tool_call_id)
            if tool_name:
                self._extract_tool_data(tool_name, msg_content)

        # Save JSON to disk
        self._save_json()

        # --- PART B: Markdown Logging (Human Readable) ---

        with open(self.md_filename, "a", encoding="utf-8") as f:
            # 1. Human Message
            if isinstance(message, HumanMessage):
                f.write("## 👤 User Request\n")
                f.write(f"> {message.content}\n\n")

            # 2. AI Message
            elif isinstance(message, AIMessage):
                # Case: AI is executing tools
                if message.tool_calls:
                    f.write("## 🤖 Agent Reasoning & Actions\n")
                    if message.content:
                        f.write(f"{message.content}\n\n")

                    for tool in message.tool_calls:
                        f.write(f"### 🛠️ Executing Tool: `{tool['name']}`\n")
                        f.write(f"```json\n{json.dumps(tool['args'], indent=2)}\n```\n")

                # Case: AI is providing the final answer
                else:
                    f.write("## 📝 Final Output\n")
                    f.write(f"{message.content}\n\n")

            # 3. Tool Message - Only reference, no raw data
            elif isinstance(message, ToolMessage):
                tool_name = self._get_tool_name_from_id(message.tool_call_id)
                f.write(f"### 📬 Tool Output: `{tool_name or message.tool_call_id}`\n")
                f.write(
                    f"> ✅ Data received. Full data: [{os.path.basename(self.json_filename)}]"
                    f"(./{os.path.basename(self.json_filename)})\n\n"
                )

            # 4. System Message
            elif isinstance(message, SystemMessage):
                f.write(f"**System Context:** *{message.content[:100]}...*\n\n")

    def _get_tool_name_from_id(self, tool_call_id: str) -> str:
        """Find the tool name from a tool_call_id by searching previous messages."""
        for msg in self.json_data["raw_messages"]:
            if "tool_calls" in msg:
                for tool in msg["tool_calls"]:
                    if tool.get("id") == tool_call_id:
                        return tool.get("name", "")
        return ""
