"""
Baseline Graph - Zero-shot explanation without iterative refinement.

This graph provides a baseline comparison by generating a single explanation
without student feedback or iterative refinement. Used to measure the value
added by the adaptive refinement process.
"""

from typing import Dict, List, Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END

from src.agents.teacher_agent import baseline_teacher_node
from src.agents.student_agent import single_answer_node
from src.agents.grading_agent import grading_node

from IPython.display import Image


class State(TypedDict, total=False):
    """State shared across all nodes in the baseline graph."""
    gpqa_question: Dict[str, Any]
    explanation: str
    threshold: float
    max_iters: int
    single_answer: str
    single_explanation: str
    quiz_results: Dict[str, Any]


def create_baseline_graph() -> StateGraph:
    """
    Create and configure the baseline graph.
    
    This graph implements a simple zero-shot approach where:
    1. Teacher generates explanation from question (no refinement)
    2. Students answer questions based on explanation
    3. Grading agent evaluates final learning outcomes
    
    This provides a baseline to measure improvement from adaptive refinement.
    
    Returns:
        StateGraph: Compiled LangGraph ready for execution
    """
    
    # Initialize graph with shared state
    graph = StateGraph(State)
    
    # Add agent nodes
    graph.add_node("teacher", baseline_teacher_node)
    graph.add_node("single answer", single_answer_node)
    graph.add_node("grading", grading_node)
    
    # Define edge flow
    graph.add_edge("teacher", "single answer")
    graph.add_edge("single answer", "grading")
    graph.add_edge("grading", END)
    
    # Set entry point
    graph.set_entry_point("teacher")
    
    # Compile and return
    return graph.compile()


def create_initial_state(
    gpqa_question: Dict[str, Any],
    threshold: float = 0.7,
    max_iters: int = 1  # Not used in baseline, but kept for compatibility
) -> Dict:
    """
    Create initial state for the baseline graph.
    
    Args:
        gpqa_question: GPQA quiz question
        threshold: Convergence threshold (not used in baseline)
        max_iters: Maximum iterations (not used in baseline)
        
    Returns:
        Dict: Initial state dictionary
    """
    return {
        "gpqa_question": gpqa_question,
        "threshold": threshold,
        "max_iters": max_iters,
    }


def visualize_graph(compiled_graph):
    """
    Generate a visual representation of the graph structure.
    
    Args:
        compiled_graph: The compiled StateGraph
        
    Returns:
        IPython Image object for display in notebooks
    """
    return Image(compiled_graph.get_graph().draw_mermaid_png())