"""
Teacher Agent - Generates educational explanations in baseline or adaptive mode.

Baseline mode: Zero-shot explanation without examples or feedback
Adaptive mode: Few-shot with iterative refinement based on student feedback
"""

from __future__ import annotations
import os
from typing import Any, Dict, List, Optional, Literal
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

from src.config.agent_config import _llm
from src.dspy_pipeline.manager import (
    run_dspy_teacher_pass,
    should_use_dspy_teacher_backend,
)
from src.dspy_pipeline.base_prompts import get_prompt

try:
    from trulens.core.otel.instrument import instrument  # type: ignore
except Exception:  # pragma: no cover
    def instrument(*args, **kwargs):  # type: ignore
        def deco(fn):
            return fn
        return deco


# =============================================================================
# TEACHER TOOLS - RAG and Web Search
# =============================================================================

@tool
def search_physics_knowledge(query: str) -> str:
    """
    Search the LibreTexts physics textbook for relevant information.
    
    Use this tool when you need to:
    - Look up specific physics constants or values
    - Verify formulas or equations
    - Get detailed information about specific phenomena
    - Clarify technical terms or concepts you're uncertain about
    
    Args:
        query: A specific search query about physics concepts, constants, or phenomena
    
    Returns:
        Relevant excerpts from LibreTexts physics textbook
    """
    if not os.getenv("USE_RAG", "").lower() in ("1", "true", "yes"):
        return "[RAG not enabled - set USE_RAG=1 in .env]"
    
    try:
        from src.rag.retriever import retrieve_physics_context
        result = retrieve_physics_context(query, k=5)
        if result:
            print(f"    [RAG] Retrieved {len(result)} chars for: {query[:50]}...")
            return result
        return "No relevant information found in knowledge base."
    except Exception as e:
        return f"Error searching knowledge base: {e}"


@tool  
def search_web(query: str) -> str:
    """
    Search the web for current information and specific facts.
    
    Use this tool when you need to:
    - Look up specific numerical values (star properties, constants, etc.)
    - Verify recent discoveries or current scientific consensus
    - Find information about specific objects (stars, particles, materials)
    - Get authoritative sources for niche facts
    
    Args:
        query: A specific search query for factual information
    
    Returns:
        Relevant excerpts from web sources
    """
    if not os.getenv("USE_WEB_SEARCH", "").lower() in ("1", "true", "yes"):
        return "[Web search not enabled - set USE_WEB_SEARCH=1 in .env]"
    
    try:
        from src.utils.web_search import search_web as tavily_search
        result = tavily_search(query, max_results=3)
        if result:
            print(f"    [WebSearch] Retrieved {len(result)} chars for: {query[:50]}...")
            return result[:2000]  # Limit response size
        return "No relevant information found."
    except Exception as e:
        return f"Error searching web: {e}"


def _get_available_tools() -> List:
    """Get list of tools available based on env config."""
    tools = []
    if os.getenv("USE_RAG", "").lower() in ("1", "true", "yes"):
        tools.append(search_physics_knowledge)
    if os.getenv("USE_WEB_SEARCH", "").lower() in ("1", "true", "yes"):
        tools.append(search_web)
    return tools


def _get_tools_description() -> str:
    """Get description of available tools for system prompt."""
    tools = _get_available_tools()
    if not tools:
        return ""
    
    desc = "\n\nAVAILABLE TOOLS:\n"
    desc += "You have access to tools to look up information. "
    desc += "ALWAYS use at least one tool before providing your explanation to ensure accuracy.\n\n"
    
    if os.getenv("USE_RAG", "").lower() in ("1", "true", "yes"):
        desc += "- search_physics_knowledge: Search LibreTexts physics textbook for concepts, formulas, constants\n"
    if os.getenv("USE_WEB_SEARCH", "").lower() in ("1", "true", "yes"):
        desc += "- search_web: Search the web for specific facts, star properties, current values\n"
    
    desc += "\nTOOL USAGE GUIDELINES:\n"
    desc += "- Use search_physics_knowledge for: equations, derivations, conceptual explanations, physical constants\n"
    desc += "- Use search_web for: specific objects (stars, materials), numerical values, recent discoveries\n"
    desc += "- You SHOULD use tools for most questions to ground your explanation in authoritative sources\n"
    desc += "- Only skip tools if the question is purely conceptual with no specific facts to verify"
    return desc


def _build_teacher_prompt(
    mode: Literal["baseline", "adaptive"],
    question: str,
    correct_answer: Optional[str] = None,
    student_feedback: Optional[str] = None,
    word_cap: int = 600,
) -> tuple[SystemMessage, HumanMessage]:
    """
    Build system and human messages for the teacher agent based on mode.
    
    Args:
        mode: "baseline" for zero-shot, "adaptive" for few-shot with refinement
        question: The question to explain
        correct_answer: The correct answer (for guidance, not to reveal)
        student_feedback: Feedback from student personas (adaptive mode only)
        word_cap: Maximum word count for explanation
        
    Returns:
        Tuple of (system_message, human_message)
    """
    # NOTE: correct_answer is no longer passed to the teacher to prevent answer leakage
    answer_context = ""
    
    # Shared base prompt for both modes
    base_prompt = (
        "You are an expert physics teacher teaching undergraduate Physics students with "
        "varying skills and backgrounds. "

        "CRITICAL CONSTRAINTS:\n"
        "- NEVER directly state the correct answer letter or value\n"
        "- NEVER use specific numbers from the question in your examples\n"
        "- Teach the underlying concepts generically so students must apply them\n\n"

        "Role: Produce a clear, self-contained explanation that helps students understand " 
        "the given question and its underlying concepts. "
    )
    
    # Shared output format and example
    output_format = (
        f"Output format: Single block of prose (no headings). Aim for {word_cap} words. "
        "Include: "
        "(1) short intuitive orientation, "
        "(2) core mechanism step-by-step with a tiny numeric example (at most one), "
        "(3) brief visual/spatial analogy if helpful, " 
        "(4) short rigorous note (key definitions/equations) where appropriate. "
        "Each sentence should add new information. "
        f"Limit explanation to no more than {word_cap} words. "
        "CRITICAL: DO NOT directly reference the given question or reveal the correct "
        "answer. If you include any examples in your explanation, do not use any "
        "information directly mentioned in the problem. Teach the underlying concepts "
        "generically so students can apply them to solve the problem independently. "
        "Example of the explanation style:\n"
        "Question: An electron is at rest (not moving). A relativistic positron is moving " 
        "horizontally from the left with a constant speed.\nAfter hitting the electron, " 
        "both annihilate producing 2 photons.\n\nThe direction of one of the  photons is " 
        "in the upper-right direction. The angle between this direction and the horizontal " 
        "line/axis is 60 degrees. The photon energy is 0.613 MeV (1.2 times the rest mass " 
        "of an electron). \n\nWhat was the speed of the positron (expresses as a fraction " 
        "of the speed of light c):\n"
        "Explanation: When matter and antimatter collide and annihilate, they convert their "
        "mass-energy into photons, conserving both energy and momentum. For relativistic "
        "particles moving at speeds comparable to light, we use E² = (pc)² + (mc²)² where E "
        "is total energy, p is momentum, m is rest mass, and c is light speed. For photons "
        "with zero rest mass, E = pc. The Lorentz factor γ = 1/√(1 - v²/c²) relates particle "
        "speed to energy: E = γmc² and momentum p = γmv. Apply conservation laws: total " 
        "initial energy equals sum of photon energies; initial momentum vector equals vector "
        "sum of photon momenta. Break momentum into horizontal and vertical components. The "
        "photon angle and energy reveal the initial particle's momentum and thus its Lorentz "
        "factor. From γ, extract speed using v/c = √(1 - 1/γ²). This framework applies broadly "
        "to two-body decay and annihilation processes. "
        "Use this explanation as a guide for the question provided by the user."
    )

    if mode == "baseline":
        # Baseline: Same prompt as adaptive, but no refinement instructions
        sys = SystemMessage(
            content=base_prompt + output_format
        )
        hum = HumanMessage(
            content=f"Question: {question}{answer_context}\n\nProvide the explanation."
        )
        
    else:  
        # Adaptive: Include refinement instructions for handling feedback
        has_feedback = bool(student_feedback and student_feedback.strip())
        
        refinement_instructions = (
            "On first iteration, create a well-structured explanation covering key concepts. "
            "On later rounds, you will receive feedback from the TOP-RANKED student critiques "
            "(only the most important issues identified by independent judges). "
            "Revise based on this feedback. Prefer tightening, clarifying, or replacing over "
            "adding new material. "
            "IGNORE feedback that:\n"
            "- Focuses on tangential topics not needed for the solution\n"
            "- Misunderstands the core physics principles\n"
            "- Requests information that would give away the answer\n\n"
        )
        
        sys = SystemMessage(
            content=base_prompt + refinement_instructions + output_format
        )
        
        # Build human message with optional feedback
        if has_feedback:
            fb_text = f"\n\nStudent feedback (top-ranked critiques only):\n{student_feedback}"
        else:
            fb_text = "\n\nNo significant issues identified in previous iteration."
        
        hum = HumanMessage(
            content=f"Question: {question}{answer_context}{fb_text}\n\nProvide the explanation."
        )
    
    return sys, hum


@instrument()
def teacher_explain(
    question: str,
    mode: Literal["baseline", "adaptive"] = "adaptive",
    correct_answer: Optional[str] = None,
    student_feedback: Optional[str] = None,
    word_cap: int = 300,
    max_tokens: int = 5000
) -> str:
    """
    Generate an explanation for the given question.
    
    Args:
        mode: "baseline" for zero-shot or "adaptive" for iterative refinement
        question: The question to explain
        correct_answer: The correct answer (for guidance, not to reveal)
        student_feedback: Feedback from student personas (adaptive mode only)
        word_cap: Maximum word count for explanation
        
    Returns:
        Generated explanation text
    """
    llm = _llm(role="teacher", max_tokens=max_tokens)

    # Build prompts based on mode
    sys, hum = _build_teacher_prompt(mode, question, correct_answer, student_feedback, word_cap)

    # Generate explanation
    resp = llm.invoke([sys, hum])
    content = resp.content if isinstance(resp.content, str) else str(resp.content)
    
    # Clean and truncate
    text = " ".join(content.strip().split())
    words = text.split()
    if len(words) > word_cap:
        text = " ".join(words[:word_cap])
    
    return text


def adaptive_teacher_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Teacher node for adaptive refinement graph.
    
    Uses adaptive mode with student feedback for iterative improvement.
    Teacher has access to RAG and web search tools to verify facts when uncertain.
    """
    iteration = int(state.get("iteration", 0))

    # Extract question and options from gpqa_question
    gpqa_question = state.get("gpqa_question", {})
    if not gpqa_question:
        raise ValueError("gpqa_question not found in state")
    question = gpqa_question.get("question", "")
    options = gpqa_question.get("options", [])
    # NOTE: correct_answer intentionally not extracted - prevents answer leakage to teacher

    if should_use_dspy_teacher_backend():
        result = run_dspy_teacher_pass(gpqa_question, teacher_persona="general")
        explanation = result.get("refined_explanation") or result.get("initial_explanation", "")
        return {
            "explanation": explanation,
            "iteration": iteration + 1,
            "dspy_payload": result,
        }

    # Get filtered feedback from previous iteration
    filtered_feedback = state.get("filtered_critiques", "")

    # Build human message with options and feedback
    human_parts = [f"Question: {question}"]
    if options:
        human_parts.append(f"\nOPTIONS:\n" + "\n".join(options))
    if filtered_feedback and filtered_feedback.strip():
        human_parts.append(f"\nStudent feedback (address these gaps):\n{filtered_feedback}")
    human_parts.append("\nProvide the explanation.")

    # Build system prompt with tools description
    tools_desc = _get_tools_description()
    sys, _ = _build_teacher_prompt("adaptive", question, None, filtered_feedback, 600)
    sys_content = sys.content + tools_desc
    
    sys = SystemMessage(content=sys_content)
    hum = HumanMessage(content="\n".join(human_parts))
    
    # Get available tools and create LLM
    tools = _get_available_tools()
    llm = _llm(role="teacher", max_tokens=5000)
    
    # Track tool calls for logging
    tool_call_logs = []
    
    # If tools available, bind them and run agentic loop
    if tools:
        llm_with_tools = llm.bind_tools(tools)
        messages = [sys, hum]
        
        # Agentic loop - let teacher use tools as needed
        max_tool_calls = 5  # Prevent infinite loops
        tool_calls_made = 0
        
        while tool_calls_made < max_tool_calls:
            response = llm_with_tools.invoke(messages)
            messages.append(response)
            
            # Check if model wants to use tools
            if not response.tool_calls:
                # No more tool calls - we have the final response
                break
            
            # Execute tool calls
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]
                
                # Execute the appropriate tool
                if tool_name == "search_physics_knowledge":
                    result = search_physics_knowledge.invoke(tool_args)
                    # Log RAG call with context preview
                    tool_call_logs.append({
                        "tool": "rag",
                        "query": tool_args.get("query", ""),
                        "context_preview": result[:300] + "..." if len(result) > 300 else result
                    })
                elif tool_name == "search_web":
                    result = search_web.invoke(tool_args)
                    # Log web search call with query
                    tool_call_logs.append({
                        "tool": "web_search",
                        "query": tool_args.get("query", "")
                    })
                else:
                    result = f"Unknown tool: {tool_name}"
                
                # Add tool result to messages
                messages.append(ToolMessage(content=result, tool_call_id=tool_id))
                tool_calls_made += 1
        
        # Extract final content
        content = response.content if isinstance(response.content, str) else str(response.content)
    else:
        # No tools - simple invocation
        resp = llm.invoke([sys, hum])
        content = resp.content if isinstance(resp.content, str) else str(resp.content)
    
    explanation = " ".join(content.strip().split())
    
    # Build result dict - only include tool_calls if any were made
    result = {"explanation": explanation, "iteration": iteration + 1}
    if tool_call_logs:
        result["tool_calls"] = tool_call_logs
    
    return result


def baseline_teacher_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Teacher node for baseline graph.
    
    Uses zero-shot mode without examples or feedback.
    Teacher has access to RAG and web search tools to verify facts when uncertain.
    """
    # Extract question and options from gpqa_question
    gpqa_question = state.get("gpqa_question", {})
    if not gpqa_question:
        raise ValueError("gpqa_question not found in state")
    question = gpqa_question.get("question", "")
    options = gpqa_question.get("options", [])
    
    # Build human message with options
    human_parts = [f"Question: {question}"]
    if options:
        human_parts.append(f"\nOPTIONS:\n" + "\n".join(options))
    human_parts.append("\nProvide a clear explanation.")
    
    # Build system prompt with tools description
    tools_desc = _get_tools_description()
    sys, _ = _build_teacher_prompt("baseline", question, None, None, 600)
    sys_content = sys.content + tools_desc
    
    sys = SystemMessage(content=sys_content)
    hum = HumanMessage(content="\n".join(human_parts))
    
    # Get available tools and create LLM
    tools = _get_available_tools()
    llm = _llm(role="teacher", max_tokens=5000)
    
    # Track tool calls for logging
    tool_call_logs = []
    
    # If tools available, bind them and run agentic loop
    if tools:
        llm_with_tools = llm.bind_tools(tools)
        messages = [sys, hum]
        
        # Agentic loop - let teacher use tools as needed
        max_tool_calls = 5  # Prevent infinite loops
        tool_calls_made = 0
        
        while tool_calls_made < max_tool_calls:
            response = llm_with_tools.invoke(messages)
            messages.append(response)
            
            # Check if model wants to use tools
            if not response.tool_calls:
                # No more tool calls - we have the final response
                break
            
            # Execute tool calls
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]
                
                # Execute the appropriate tool
                if tool_name == "search_physics_knowledge":
                    result = search_physics_knowledge.invoke(tool_args)
                    # Log RAG call with context preview
                    tool_call_logs.append({
                        "tool": "rag",
                        "query": tool_args.get("query", ""),
                        "context_preview": result[:300] + "..." if len(result) > 300 else result
                    })
                elif tool_name == "search_web":
                    result = search_web.invoke(tool_args)
                    # Log web search call with query
                    tool_call_logs.append({
                        "tool": "web_search",
                        "query": tool_args.get("query", "")
                    })
                else:
                    result = f"Unknown tool: {tool_name}"
                
                # Add tool result to messages
                messages.append(ToolMessage(content=result, tool_call_id=tool_id))
                tool_calls_made += 1
        
        # Extract final content
        content = response.content if isinstance(response.content, str) else str(response.content)
    else:
        # No tools - simple invocation
        resp = llm.invoke([sys, hum])
        content = resp.content if isinstance(resp.content, str) else str(resp.content)
    
    explanation = " ".join(content.strip().split())
    
    # Build result dict - only include tool_calls if any were made
    result = {"explanation": explanation, "iteration": 1}
    if tool_call_logs:
        result["tool_calls"] = tool_call_logs
    
    return result