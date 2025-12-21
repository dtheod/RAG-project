from langgraph.graph import StateGraph, END
from typing import TypedDict
from src.agents import (
    cars_agent,
    countries_agent,
    math_agent,
    general_agent,
    classifier_agent,
    format_agent
)
from langchain.schema.runnable.config import RunnableConfig
import chainlit as cl
from langfuse.langchain import CallbackHandler

from typing import List

class State(TypedDict):
    query: str
    formatted_query: str
    category: str
    classifier_reason: str
    response: str
    contexts: List[str]
    sources: List[str]



def route_query(state):
    """Route to appropriate agent based on category."""
    category = state["category"]
    
    if category == "cars":
        return "cars_agent"
    elif category == "countries":
        return "countries_agent"
    elif category == "math":
        return "math_agent"
    else:
        return "general_agent"

# Build the graph
workflow = StateGraph(State)

# Add nodes
workflow.add_node("classifier", classifier_agent)
workflow.add_node("cars_agent", cars_agent)
workflow.add_node("countries_agent", countries_agent)
workflow.add_node("math_agent", math_agent)
workflow.add_node("general_agent", general_agent)
workflow.add_node("format_agent", format_agent)

# Set entry point
workflow.set_entry_point("format_agent")
workflow.add_edge("format_agent", "classifier")

# Add conditional routing
workflow.add_conditional_edges(
    "classifier",
    route_query,
    {
        "cars_agent": "cars_agent",
        "countries_agent": "countries_agent", 
        "math_agent": "math_agent",
        "general_agent": "general_agent"
    }
)

# All agents end the workflow
workflow.add_edge("cars_agent", END)
workflow.add_edge("countries_agent", END)
workflow.add_edge("math_agent", END)
workflow.add_edge("general_agent", END)

# Compile the graph

langfuse_handler = CallbackHandler()
app = workflow.compile().with_config({"callbacks": [langfuse_handler]})

# Write graph to an image file
png_graph = app.get_graph().draw_mermaid_png()
with open("./outputs/workflow_graph.png", "wb") as f:
    f.write(png_graph)


def process_query(query):
    """Process a query through the workflow."""

    result = app.invoke({"query": query})    
    return result
