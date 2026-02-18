from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.news_tools import fetch_press_releases, search_company_news
from src.state import AgentState

load_dotenv()

# --- 1. SETUP LLM & TOOLS ---
llm = ChatOpenAI(model="gpt-5-mini", temperature=0)

news_tools = [
    fetch_press_releases,
    search_company_news,
]

llm_with_news_tools = llm.bind_tools(news_tools)


# --- 2. DEFINE NODES ---


def gatherer_node(state: AgentState):
    """
    The Gatherer Node.
    Calls all news tools in parallel to collect press releases + search results.
    """
    messages = state["messages"]

    if len(messages) == 1 and isinstance(messages[0], HumanMessage):
        ticker = state["ticker"]
        system_msg = SystemMessage(
            content=f"""
        You are a News Research Assistant focused on gathering news for ticker: {ticker}.
        
        You MUST call these tools to collect news data:
        1. fetch_press_releases — Official company press releases (use ticker: {ticker})
        2. search_company_news — Third-party news and analysis
        
        CRITICAL RULES:
        - Call ALL tools listed above. Do NOT skip any.
        - Execute tools IN PARALLEL for speed.
        - For search_company_news: you MUST include the FULL COMPANY NAME in the query,
          not just the ticker symbol. Short tickers like "TPC" return irrelevant results.
          Example: "Tutor Perini Corporation TPC latest contract wins earnings news 2025"
        - After tools return data, respond with ONLY: "News gathering complete."
        - Do NOT summarize, interpret, or analyze the data.
        """
        )
        messages = [system_msg] + messages

    response = llm_with_news_tools.invoke(messages)
    return {"messages": [response]}


def analyst_node(state: AgentState):
    """
    The Analyst Node.
    Synthesizes raw news data into 5 deep-analysis articles.
    """
    messages = state["messages"]
    ticker = state["ticker"]

    prompt = f"""
    You are a Senior Financial News Analyst writing for investors who are just getting started.
    Synthesize all collected news into ONE compelling, easy-to-understand analysis article for {ticker}.

    ---
    ### 🎨 WRITING STYLE (THE "COLSON" STYLE — Smart Mentor)
    - Sophisticated but accessible — like a smart friend explaining the news over coffee
    - Use vivid metaphors and analogies to explain complex concepts
    - Avoid jargon; if you must use a financial term, briefly explain it
    - Use paragraphs for the main narrative, bullet points ONLY for key data metrics
    - Catchy headline format: `[What Happened]: [Why It Matters in Plain English]`
      * Good: "Record $21.6B Backlog: Why This Builder Has Work Lined Up for Years"
      * Bad: "Tutor Perini Corporation Reports Q3 Results" (boring, generic)

    ---
    ### 📝 ARTICLE STRUCTURE (5-7 paragraphs total)

    **[Catchy Headline]: [A Hook That Makes You Want to Read More]**
    *{ticker} | [Current Date] | Sources: [key sources]*

    1. **The Hook (1 paragraph)**: Start with the most exciting development. What just happened
       and why should anyone care? Make it punchy and specific — lead with the dollar figure.

    2. **The Bigger Picture (1-2 paragraphs)**: Connect this news to other recent developments.
       Don't just list events — tell a STORY. "This $900M contract didn't happen in a vacuum..."
       Help the reader see the pattern and the company's direction.

    3. **Show Me the Numbers (1 paragraph)**: Key financial figures that prove the story.
       Revenue, margins, backlog, cash flow — but explain what they MEAN.
       "They generated $574M in operating cash flow — think of it as the company's paycheck
       after paying all its bills."

    4. **What It Means for You (1 paragraph)**: Plain-English investor takeaway.
       Is this good news or bad news? What should someone watching this stock know?
       Use "Smart Mentor" framing — "Here's what experienced investors are watching..."

    5. **The Risk You Should Know (1 paragraph)**: One or two key risks, explained simply.
       "The catch is..." / "The one thing that could spoil the party is..."

    ---
    ### QUALITY RULES
    - NEVER fabricate data — only use information from the provided tool outputs
    - This is ONE article, NOT multiple — synthesize all news into a single narrative
    - Use SPECIFIC numbers — never say "significant" without a dollar figure
    - Tone: "Smart Mentor" — imagine explaining this to a curious 25-year-old who just
      opened their first brokerage account

    Output ONLY the article. No meta-commentary, no introduction, no conclusion.
    """

    response = llm.invoke(messages + [HumanMessage(content=prompt)])
    return {"messages": [response]}


# --- 3. CONDITIONAL EDGES ---


def should_continue(state: AgentState) -> Literal["tools", "analyst"]:
    """Route: if tool calls exist -> tools, otherwise -> analyst."""
    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        return "tools"
    return "analyst"


# --- 4. BUILD THE GRAPH ---

news_workflow = StateGraph(AgentState)

news_workflow.add_node("gatherer", gatherer_node)
news_workflow.add_node("tools", ToolNode(news_tools))
news_workflow.add_node("analyst", analyst_node)

news_workflow.add_edge(START, "gatherer")
news_workflow.add_conditional_edges(
    "gatherer",
    should_continue,
    {
        "tools": "tools",
        "analyst": "analyst",
    },
)
news_workflow.add_edge("tools", "analyst")
news_workflow.add_edge("analyst", END)

news_app = news_workflow.compile()
