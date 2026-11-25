"""LangGraph pipeline: intent -> images -> vectorize -> explain."""
from langgraph.graph import END, StateGraph

from app.agent.nodes.explain import explain
from app.agent.nodes.image_gen import generate_images
from app.agent.nodes.intent import parse_intent
from app.agent.nodes.vectorize import vectorize
from app.agent.state import AgentState


def _after_intent(state: AgentState) -> str:
    return "images" if state.get("intent") else END


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("intent", parse_intent)
    g.add_node("images", generate_images)
    g.add_node("vectorize", vectorize)
    g.add_node("explain", explain)
    g.set_entry_point("intent")
    g.add_conditional_edges("intent", _after_intent)
    g.add_edge("images", "vectorize")
    g.add_edge("vectorize", "explain")
    g.add_edge("explain", END)
    return g.compile()


graph = build_graph()
