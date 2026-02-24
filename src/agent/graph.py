"""LangGraph StateGraph for the AI watch agent.

Pipeline:
    fetch_sources → filter_github → combine_items → enrich_and_brief
"""

from langgraph.graph import END, StateGraph

from src.agent.nodes import combine_items, enrich_and_brief, fetch_sources, filter_github
from src.agent.state import AgentState


def build_graph() -> StateGraph:
    """Build the agent graph."""
    graph = StateGraph(AgentState)

    graph.add_node("fetch_sources", fetch_sources)
    graph.add_node("filter_github", filter_github)
    graph.add_node("combine_items", combine_items)
    graph.add_node("enrich_and_brief", enrich_and_brief)

    graph.set_entry_point("fetch_sources")

    graph.add_edge("fetch_sources", "filter_github")
    graph.add_edge("filter_github", "combine_items")
    graph.add_edge("combine_items", "enrich_and_brief")
    graph.add_edge("enrich_and_brief", END)

    return graph


def compile_graph():
    """Compile and return the runnable graph."""
    return build_graph().compile()
