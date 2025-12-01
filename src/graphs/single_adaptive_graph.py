"""
Single Adaptive Graph - Single-student iterative explanation improvement.

This graph orchestrates a teacher agent that creates educational explanations,
which are then critiqued by a single student agent. The explanations are refined
iteratively based on student feedback until convergence criteria are met.
"""

from typing import Dict, List, Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END

from src.agents.teacher_agent import single_student_adaptive_teacher_node
from src.agents.student_agent import single_student_critique_node, single_answer_node
from src.agents.stopping_agent import stopper_node
from src.agents.grading_agent import grading_node
from src.agents.answer_leak_checker import check_answer_leakage_node

from IPython.display import Image


class State(TypedDict, total=False):
    """State shared across all nodes in the single adaptive graph."""
    gpqa_question: Dict[str, Any]
    explanation: str
    single_student_critique: str
    single_student_response: Dict[str, Any]
    critique_history: List[str]
    iteration: int
    threshold: float
    max_iters: int
    decision: str
    reason: str
    single_answer: str
    single_explanation: str
    quiz_results: Dict[str, Any]
    tool_calls: List[Dict[str, Any]]  # Track teacher's tool usage
    answer_leakage_detected: bool
    leakage_feedback: str


def create_single_adaptive_graph() -> StateGraph:
    """
    Create and configure the single adaptive graph.
    
    This graph implements a single-student system where:
    1. Teacher gets question from state and generates an explanation
    2. Single student provides critique
    3. Stopper decides whether to continue or finalize
    4. Leakage checker verifies teacher hasn't leaked the answer
    5. Student answers question based on explanation
    6. Grading agent evaluates final learning outcomes
    
    Returns:
        StateGraph: Compiled LangGraph ready for execution
    """

    # Initialize graph with shared state
    graph = StateGraph(State)
    
    # Add agent nodes
    graph.add_node("teacher", single_student_adaptive_teacher_node)
    graph.add_node("student critique", single_student_critique_node)
    graph.add_node("stopper", stopper_node)
    graph.add_node("leakage checker", check_answer_leakage_node)
    graph.add_node("single answer", single_answer_node)
    graph.add_node("grading", grading_node)
    
    # Define edge flow
    graph.add_edge("teacher", "student critique")
    graph.add_edge("student critique", "stopper")
    graph.add_edge("single answer", "grading")
    graph.add_edge("grading", END)
    
    # Conditional routing from stopper
    def route_from_stop(state: State) -> str:
        """Route to leakage checker if STOP, otherwise back to teacher for refinement."""
        return "leakage checker" if state.get("decision") == "STOP" else "teacher"
    
    graph.add_conditional_edges(
        "stopper",
        route_from_stop,
        {"leakage checker": "leakage checker", "teacher": "teacher"}
    )
    
    # Conditional routing from leakage checker
    def route_from_leakage(state: State) -> str:
        """Route back to teacher if leakage detected, otherwise to single answer."""
        return "teacher" if state.get("answer_leakage_detected") else "single answer"
    
    graph.add_conditional_edges(
        "leakage checker",
        route_from_leakage,
        {"teacher": "teacher", "single answer": "single answer"}
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
    Create initial state for the single adaptive graph.
    
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
        "max_iters": max_iters,
        "iteration": 0,
        "single_student_critique": "",
        "critique_history": []
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