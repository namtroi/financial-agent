from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """
    The state of the agent graph.
    It holds the conversation history and the target ticker.
    """
    ticker: str