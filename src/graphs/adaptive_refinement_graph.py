"""
Adaptive Refinement Graph - Multi-agent system for iterative explanation improvement.

This graph orchestrates a teacher agent that creates educational explanations,
which are then critiqued by diverse student agents. The explanations are refined
iteratively based on student feedback until convergence criteria are met.
"""

from typing import Dict, List, Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END

from src.agents.teacher_agent import adaptive_teacher_node
from src.agents.student_agent import student_critiques_node, single_answer_node
from src.agents.critique_eval_agent import reward_node
from src.agents.stopping_agent import stopper_node
from src.agents.grading_agent import grading_node

from IPython.display import Image


class State(TypedDict, total=False):
    """State shared across all nodes in the adaptive refinement graph."""
    gpqa_question: Dict[str, Any]
    explanation: str
    student_responses: Dict[str, str]
    reward_scores: Dict[str, int]
    history: List[Dict]
    iteration: int
    threshold: float
    max_iters: int
    decision: str
    reason: str
    single_answer: str
    single_explanation: str
    quiz_results: Dict[str, Any]


def create_adaptive_refinement_graph() -> StateGraph:
    """
    Create and configure the adaptive refinement graph.
    
    This graph implements a multi-agent system where:
    1. Teacher gets question from state and generates an explanation
    2. Students (multiple personas) provide critiques
    3. Reward agent evaluates critique quality
    4. Stopper decides whether to continue or finalize
    5. Students answer questions based on explanation
    6. Grading agent evaluates final learning outcomes
    
    Returns:
        StateGraph: Compiled LangGraph ready for execution
    """

    # Initialize graph with shared state
    graph = StateGraph(State)
    
    # Add agent nodes
    graph.add_node("teacher", adaptive_teacher_node)
    graph.add_node("student critiques", student_critiques_node)
    graph.add_node("reward", reward_node)
    graph.add_node("stopper", stopper_node)
    graph.add_node("single answer", single_answer_node)
    graph.add_node("grading", grading_node)
    
    # Define edge flow
    graph.add_edge("teacher", "student critiques")
    graph.add_edge("student critiques", "reward")
    graph.add_edge("reward", "stopper")
    graph.add_edge("single answer", "grading")
    graph.add_edge("grading", END)
    
    # Conditional routing from stopper
    def route_from_stop(state: State) -> str:
        """Route to single answer if STOP, otherwise back to teacher for refinement."""
        return "single answer" if state.get("decision") == "STOP" else "teacher"
    
    graph.add_conditional_edges(
        "stopper",
        route_from_stop,
        {"single answer": "single answer", "teacher": "teacher"}
    )
    
    # Set entry point
    graph.set_entry_point("teacher")
    
    # Compile and return
    return graph.compile()


def create_initial_state(
    gpqa_question: Dict[str, Any],
    threshold: float = 0.7,
    max_iters: int = 5
) -> Dict:
    """
    Create initial state for the adaptive refinement graph.
    
    Args:
        gpqa_question: GPQA quiz question
        threshold: Convergence threshold for stopping criterion (0-1)
        max_iters: Maximum number of refinement iterations
        
    Returns:
        Dict: Initial state dictionary
    """
    return {
        "gpqa_question": gpqa_question,
        "threshold": threshold,
        "max_iters": max_iters
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