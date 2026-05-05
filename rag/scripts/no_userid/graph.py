from langgraph.graph import StateGraph, START, END

from state import AgentState
from agent_runtime import guarded_node, route_next_step

try:
    from nodes.intent import intent_node
    from nodes.vision import vision_node
    from nodes.diagnosis import diagnosis_node
    from nodes.rag import rag_node
    from nodes.final import final_node
    from nodes.physical import physical_node
    from nodes.pathology import pathology_node
except Exception:
    from intent import intent_node
    from vision import vision_node
    from diagnosis import diagnosis_node
    from rag import rag_node
    from final import final_node
    from physical import physical_node
    from pathology import pathology_node


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("intent_node", guarded_node("intent_node", intent_node))
    builder.add_node("vision_node", guarded_node("vision_node", vision_node))
    builder.add_node("diagnosis_node", guarded_node("diagnosis_node", diagnosis_node))
    builder.add_node("physical_node", guarded_node("physical_node", physical_node))
    builder.add_node("pathology_node", guarded_node("pathology_node", pathology_node))
    builder.add_node("rag_node", guarded_node("rag_node", rag_node))
    builder.add_node("final_node", guarded_node("final_node", final_node))

    all_routes = {
        "vision_node": "vision_node",
        "diagnosis_node": "diagnosis_node",
        "physical_node": "physical_node",
        "pathology_node": "pathology_node",
        "rag_node": "rag_node",
        "final_node": "final_node",
    }

    builder.add_edge(START, "intent_node")
    for node_name in [
        "intent_node",
        "vision_node",
        "diagnosis_node",
        "physical_node",
        "pathology_node",
        "rag_node",
    ]:
        builder.add_conditional_edges(node_name, route_next_step, all_routes)

    builder.add_edge("final_node", END)
    return builder.compile()


agent_graph = build_graph()
