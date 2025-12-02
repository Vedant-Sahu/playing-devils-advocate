"""
Teacher Agent - Generates step-by-step solution guidance in three modes.

Modes:
- baseline: Zero-shot explanation without feedback
- single_student_adaptive: Iterative refinement with one student persona
- multi_student_adaptive: Iterative refinement with multiple student personas
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
    use_rag = os.getenv("USE_RAG", "")
    use_web = os.getenv("USE_WEB_SEARCH", "")
    print(f"    [DEBUG] USE_RAG={use_rag!r}, USE_WEB_SEARCH={use_web!r}")
    if use_rag.lower() in ("1", "true", "yes"):
        tools.append(search_physics_knowledge)
    if use_web.lower() in ("1", "true", "yes"):
        tools.append(search_web)
    print(f"    [DEBUG] Available tools: {[t.name for t in tools]}")
    return tools


def _get_tools_description() -> str:
    """Get description of available tools for system prompt."""
    tools = _get_available_tools()
    if not tools:
        return ""
    
    desc = "\n\nAVAILABLE TOOLS - USE THEM LIBERALLY:\n"
    desc += "You have access to tools to look up and VERIFY information. "
    desc += "Be intellectually humble - even experts make mistakes on specific values and edge cases.\n\n"
    
    desc += "WHEN TO USE TOOLS (err on the side of using them):\n"
    desc += "- ANY specific numerical value (constants, properties, measurements)\n"
    desc += "- Formulas you haven't used recently - verify the exact form\n"
    desc += "- Properties of specific objects (stars, materials, particles)\n"
    desc += "- Boundary conditions or special cases\n"
    desc += "- Anything where being wrong would mislead students\n\n"
    
    if os.getenv("USE_RAG", "").lower() in ("1", "true", "yes"):
        desc += "- search_physics_knowledge: Search LibreTexts physics textbook for concepts, formulas, constants, derivations\n"
    if os.getenv("USE_WEB_SEARCH", "").lower() in ("1", "true", "yes"):
        desc += "- search_web: Search the web for specific facts, star properties, material constants, current values\n"
    
    desc += "\nTOOL USAGE GUIDELINES:\n"
    desc += "- ALWAYS verify specific constants and values before using them in explanations\n"
    desc += "- Use search_physics_knowledge FIRST for physics concepts and standard formulas\n"
    desc += "- Use search_web for domain-specific data (astronomy, materials science, etc.)\n"
    desc += "- If a question mentions specific objects or phenomena, look them up\n"
    desc += "- Better to verify and be correct than assume and mislead students\n\n"
    desc += "**MANDATORY**: You MUST use at least one tool call before providing your explanation. "
    desc += "Search for relevant physics concepts or verify key facts from the question first. "
    desc += "Do NOT skip tool usage - this is required for quality assurance.\n"
    return desc


def _build_teacher_prompt(
    mode: Literal["baseline", "single_student_adaptive", "multi_student_adaptive"],
    question: str,
    correct_answer: Optional[str] = None,
    student_feedback: Optional[str] = None,
    word_cap: int = 600,
) -> tuple[SystemMessage, HumanMessage]:
    """
    Build system and human messages for the teacher agent based on mode.
    
    Args:
        mode: Operating mode (baseline, single_student_adaptive, multi_student_adaptive)
        question: The question to explain
        correct_answer: The correct answer for guidance
        student_feedback: Feedback from student persona(s) (adaptive modes only)
        word_cap: Maximum word count for explanation
        
    Returns:
        Tuple of (system_message, human_message)
    """
    # Answer context for guidance
    if correct_answer:
        answer_context = f"\n\nCORRECT ANSWER (for your guidance only - DO NOT reveal): {correct_answer}"
    else:
        answer_context = ""
    
    # Base prompt shared across all modes
    base_prompt = (
        "You are an expert physics teacher helping undergraduate Physics students understand "
        "how to solve problems through step-by-step guidance.\n\n"
        
        "ANSWER AWARENESS:\n"
        "You have access to the correct answer to guide your explanation toward the solution path. "
        "Use this to:\n"
        "- Focus on the specific approach and steps needed to solve THIS problem\n"
        "- Structure your guidance to lead students toward the correct solution\n"
        "- Provide relevant concepts and formulas needed for this specific solution path\n\n"
        
        "CRITICAL CONSTRAINTS:\n"
        "- NEVER directly state the correct answer letter or final numerical value\n"
        "- NEVER work through the problem using the specific numbers from the question\n"
        "- Provide a step-by-step APPROACH to solving problems of this type\n"
        "- Teach the method generically so students must apply it to the specific numbers\n"
        "- Frame your guidance as 'how to solve this type of problem' not 'the answer is...'\n\n"
        
        "YOUR TASK:\n"
        "Provide a clear, structured guide on HOW TO SOLVE this problem. Think of it as giving "
        "students a recipe they must execute themselves. Your explanation should:\n"
        "1. Identify what type of problem this is and what approach to use\n"
        "2. List the key steps in order (e.g., 'First, identify the forces...', 'Then, apply conservation of...')\n"
        "3. Specify which equations or principles to use at each step\n"
        "4. Explain how to combine the results to reach the final answer\n"
        "5. Include a generic worked example with different numbers if helpful\n\n"
    )
    
    # Mode-specific additions
    if mode == "baseline":
        mode_specific = (
            "MODE: Baseline (Zero-shot)\n"
            "Generate your best explanation without prior attempts or feedback. "
            "Focus on clarity and completeness from the start.\n\n"
        )
    elif mode == "single_student_adaptive":
        mode_specific = (
            "MODE: Single Student Adaptive\n"
            "You are in an iterative refinement process with ONE student providing feedback. "
            "On first iteration, create a well-structured solution guide. "
            "On later iterations, you will receive feedback identifying gaps or confusion. "
            "Revise your explanation to address the specific issue raised. "
            "Prefer clarifying or restructuring over adding new material.\n\n"
        )
    else:  # multi_student_adaptive
        mode_specific = (
            "MODE: Multi-Student Adaptive\n"
            "You are in an iterative refinement process with MULTIPLE students of different personas. "
            "On first iteration, create a well-structured solution guide. "
            "On later iterations, you will receive feedback from the TOP-RANKED critiques only "
            "(the most important issues identified by independent judges). "
            "Different students look for different types of issues (misconceptions, clarity, rigor, etc.). "
            "Revise based on the feedback. Prefer tightening, clarifying, or replacing over adding new material.\n\n"
            
            "IGNORE feedback that:\n"
            "- Focuses on tangential topics not needed for the solution\n"
            "- Misunderstands the core physics principles\n"
            "- Requests information that would give away the answer\n\n"
        )
    
    # Output format
    output_format = (
        f"OUTPUT FORMAT:\n"
        f"Write in clear prose paragraphs (no bullet points or headers). Target {word_cap} words.\n"
        f"Structure your response as:\n"
        f"1. Brief problem type identification (1-2 sentences)\n"
        f"2. Step-by-step solution approach with specific actions to take\n"
        f"3. Key equations/formulas to use (where in the process to use them)\n"
        f"4. How to interpret and combine results\n"
        f"5. Optional: Generic example with different numbers to illustrate the method\n\n"
        
        f"Example style (for a different problem):\n"
        f"\"This is a relativistic collision problem requiring conservation of energy and momentum. "
        f"Start by writing the energy conservation equation: total initial energy equals sum of final photon energies. "
        f"The initial energy includes the positron's kinetic energy plus both rest masses. "
        f"Next, write momentum conservation in vector form, breaking into horizontal and vertical components. "
        f"The electron starts at rest, so initial momentum equals the positron's momentum. "
        f"Use the photon angle and energy to determine each photon's momentum components (recall E = pc for photons). "
        f"Solve the component equations simultaneously to find the positron's momentum. "
        f"Convert momentum to velocity using the relativistic relation p = γmv where γ = 1/√(1 - v²/c²). "
        f"For example, if you found γ = 1.5 for some particle, you would solve 1.5 = 1/√(1 - v²/c²) to get v/c = √(1 - 1/1.5²) = 0.745.\"\n\n"
        
        f"REMEMBER: Teach the METHOD, not the specific solution. Students must apply your steps to their numbers.\n"
    )
    
    # Construct system message
    sys_content = base_prompt + mode_specific + output_format
    sys = SystemMessage(content=sys_content)
    
    # Construct human message with feedback if applicable
    human_parts = [f"Question: {question}{answer_context}"]
    
    if student_feedback and student_feedback.strip():
        if mode == "single_student_adaptive":
            human_parts.append(f"\n\nSTUDENT FEEDBACK:\n{student_feedback}")
        else:  # multi_student_adaptive
            human_parts.append(f"\n\nTOP-RANKED STUDENT CRITIQUES:\n{student_feedback}")
        human_parts.append("\n\nRevise your explanation to address this feedback.")
    else:
        human_parts.append("\n\nProvide your step-by-step solution guide.")
    
    hum = HumanMessage(content="".join(human_parts))
    
    return sys, hum


@instrument()
def teacher_explain(
    question: str,
    mode: Literal["baseline", "single_student_adaptive", "multi_student_adaptive"] = "baseline",
    correct_answer: Optional[str] = None,
    student_feedback: Optional[str] = None,
    word_cap: int = 600,
    max_tokens: int = 5000
) -> Dict[str, Any]:
    """
    Generate a step-by-step solution guide for the given question.
    
    Args:
        question: The question to explain
        mode: Operating mode (baseline, single_student_adaptive, multi_student_adaptive)
        correct_answer: The correct answer for guidance
        student_feedback: Feedback from student persona(s) (adaptive modes only)
        word_cap: Maximum word count for explanation
        max_tokens: Maximum tokens for model completion
        
    Returns:
        Dictionary with 'explanation' and optionally 'tool_calls'
    """
    # Build prompts based on mode
    sys, hum = _build_teacher_prompt(mode, question, correct_answer, student_feedback, word_cap)
    
    # Add tools description if available
    tools_desc = _get_tools_description()
    if tools_desc:
        sys = SystemMessage(content=sys.content + tools_desc)
    
    # Get available tools and create LLM
    tools = _get_available_tools()
    llm = _llm(role="teacher", max_tokens=max_tokens)
    
    # Track tool calls for logging
    tool_call_logs = []
    
    # If tools available, bind them and run agentic loop
    if tools:
        print(f"    [DEBUG] Binding {len(tools)} tools to LLM")
        llm_with_tools = llm.bind_tools(tools)
        messages = [sys, hum]
        
        # Agentic loop - let teacher use tools as needed
        max_tool_iterations = 5  # Prevent infinite loops
        tool_calls_made = 0
        
        while tool_calls_made < max_tool_iterations:
            response = llm_with_tools.invoke(messages)
            messages.append(response)
            
            # Check if model wants to use tools
            if not response.tool_calls:
                # No more tool calls - we have the final response
                print(f"    [DEBUG] No tool calls in response, finishing")
                break
            else:
                print(f"    [DEBUG] Model requested {len(response.tool_calls)} tool call(s)")
            
            # Execute tool calls
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]
                
                # Execute the appropriate tool
                if tool_name == "search_physics_knowledge":
                    result = search_physics_knowledge.invoke(tool_args)
                    tool_call_logs.append({
                        "tool": "rag",
                        "query": tool_args.get("query", ""),
                        "context_preview": result[:300] + "..." if len(result) > 300 else result
                    })
                elif tool_name == "search_web":
                    result = search_web.invoke(tool_args)
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
    
    # Clean and truncate
    text = " ".join(content.strip().split())
    words = text.split()
    if len(words) > word_cap:
        text = " ".join(words[:word_cap])
    
    # Build result
    result = {"explanation": text}
    if tool_call_logs:
        print(f"    [DEBUG] Recording {len(tool_call_logs)} tool calls in result")
        result["tool_calls"] = tool_call_logs
    else:
        print(f"    [DEBUG] No tool calls to record")
    
    return result


def baseline_teacher_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Teacher node for baseline mode.
    
    Zero-shot explanation without feedback.
    Extracts question from gpqa_question in state.
    """
    # Extract question from gpqa_question
    gpqa_question = state.get("gpqa_question", {})
    if not gpqa_question:
        raise ValueError("gpqa_question not found in state")
    question = gpqa_question.get("question", "")
    correct_answer = gpqa_question.get("correct_answer", "")
    
    # Generate explanation in baseline mode
    result = teacher_explain(
        question=question,
        mode="baseline",
        correct_answer=correct_answer,
        student_feedback=None,
        word_cap=600
    )
    
    return {
        "explanation": result["explanation"],
        "iteration": 1,
        "tool_calls": result.get("tool_calls", [])
    }


def single_student_adaptive_teacher_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Teacher node for single student adaptive mode.
    
    Iterative refinement with feedback from one student persona.
    Extracts question from gpqa_question in state.
    """
    iteration = int(state.get("iteration", 0))
    
    # Extract question from gpqa_question
    gpqa_question = state.get("gpqa_question", {})
    if not gpqa_question:
        raise ValueError("gpqa_question not found in state")
    question = gpqa_question.get("question", "")
    correct_answer = gpqa_question.get("correct_answer", "")
    
    # Check if using DSPy backend
    if should_use_dspy_teacher_backend():
        result = run_dspy_teacher_pass(gpqa_question, teacher_persona="general")
        explanation = result.get("refined_explanation") or result.get("initial_explanation", "")
        return {
            "explanation": explanation,
            "iteration": iteration + 1,
            "dspy_payload": result,
        }
    
    # Get feedback from previous iteration
    student_feedback = state.get("single_student_critique", "")
    
    # Generate explanation in single student adaptive mode
    result = teacher_explain(
        question=question,
        mode="single_student_adaptive",
        correct_answer=correct_answer,
        student_feedback=student_feedback,
        word_cap=600
    )
    
    return {
        "explanation": result["explanation"],
        "iteration": iteration + 1,
        "tool_calls": result.get("tool_calls", [])
    }


def multi_student_adaptive_teacher_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Teacher node for multi-student adaptive mode.
    
    Iterative refinement with feedback from multiple student personas.
    Extracts question from gpqa_question in state.
    """
    iteration = int(state.get("iteration", 0))
    
    # Extract question from gpqa_question
    gpqa_question = state.get("gpqa_question", {})
    if not gpqa_question:
        raise ValueError("gpqa_question not found in state")
    question = gpqa_question.get("question", "")
    correct_answer = gpqa_question.get("correct_answer", "")
    
    # Check if using DSPy backend
    if should_use_dspy_teacher_backend():
        result = run_dspy_teacher_pass(gpqa_question, teacher_persona="general")
        explanation = result.get("refined_explanation") or result.get("initial_explanation", "")
        return {
            "explanation": explanation,
            "iteration": iteration + 1,
            "dspy_payload": result,
        }
    
    # Get filtered feedback from previous iteration (top-ranked critiques)
    filtered_feedback = state.get("filtered_critiques", "")
    
    # Generate explanation in multi-student adaptive mode
    result = teacher_explain(
        question=question,
        mode="multi_student_adaptive",
        correct_answer=correct_answer,
        student_feedback=filtered_feedback,
        word_cap=600
    )
    
    return {
        "explanation": result["explanation"],
        "iteration": iteration + 1,
        "tool_calls": result.get("tool_calls", [])
    }