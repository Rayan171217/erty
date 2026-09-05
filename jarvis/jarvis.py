"""
JARVIS - Personal AI Assistant built with LangGraph
USING CLAUDE (Anthropic) API
"""

import os
import uuid
from typing import TypedDict, Optional
from dotenv import load_dotenv

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# 🔥 CHANGE: Use Anthropic instead of OpenAI
from langchain_anthropic import ChatAnthropic

load_dotenv()

# ============================================================
# STEP 1: Define the State
# ============================================================

class JarvisState(TypedDict):
    user_input: str
    intent: Optional[str]
    extracted_details: Optional[dict]
    scheduler_output: Optional[str]
    research_output: Optional[str]
    communication_output: Optional[str]
    risk_check: Optional[str]  # "PASS" or "FAIL"
    final_response: str
    needs_approval: bool
    approved: bool
    iteration: int


# ============================================================
# STEP 2: 🔥 Claude-powered LLM
# ============================================================

def get_llm():
    """Use Claude 3.5 Sonnet or Haiku (cheaper)"""
    return ChatAnthropic(
        model="claude-3-5-sonnet-20241022",  # Best reasoning
        # model="claude-3-haiku-20240307",   # Cheaper, faster
        temperature=0.3,
        max_tokens=1024,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )


# ============================================================
# STEP 3: Agent Nodes (Identical logic, just using Claude)
# ============================================================

def supervisor_node(state: JarvisState) -> dict:
    user_input = state["user_input"]
    llm = get_llm()

    response = llm.invoke(f"""
    Classify this user request into ONE of these categories:
    - schedule: for calendar, meetings, reminders
    - research: for searching, finding information, learning
    - communication: for emails, messages, drafting replies
    - general: for casual chat, greetings, or anything else

    Request: "{user_input}"

    Return ONLY the category name (one word): schedule, research, communication, or general.
    """)

    intent = response.content.strip().lower()
    if intent not in ["schedule", "research", "communication", "general"]:
        intent = "general"

    return {
        "intent": intent,
        "iteration": state.get("iteration", 0) + 1
    }


def scheduler_node(state: JarvisState) -> dict:
    user_input = state["user_input"]
    llm = get_llm()

    # Deterministic rule (hard limit)
    meetings_today = 2
    max_meetings = 3

    if meetings_today >= max_meetings:
        return {
            "scheduler_output": f"❌ Cannot schedule: You already have {meetings_today} meetings today (max {max_meetings}).",
            "needs_approval": False
        }

    response = llm.invoke(f"""
    You are Jarvis's Scheduler Agent.
    User request: "{user_input}"

    If this is a scheduling request, extract:
    - What is the event?
    - When (time/date)?
    - Any participants?

    Return a structured proposal like:
    "Proposal: [Event] on [Date] at [Time]. Participants: [list]."

    If it's not a clear scheduling request, say "No scheduling needed."
    """)

    return {
        "scheduler_output": response.content,
        "needs_approval": True
    }


def research_node(state: JarvisState) -> dict:
    user_input = state["user_input"]
    llm = get_llm()

    response = llm.invoke(f"""
    You are Jarvis's Research Agent.
    User question: "{user_input}"

    RULES:
    1. If you know the answer, provide it concisely with confidence.
    2. If you don't know, say "I don't have that information."
    3. NEVER invent facts, dates, or sources.
    4. If the user asks for current data (weather, stocks, news), say "I need to use a live API for that. Currently in paper mode."

    Answer:
    """)

    return {
        "research_output": response.content,
        "needs_approval": False
    }


def communication_node(state: JarvisState) -> dict:
    user_input = state["user_input"]
    llm = get_llm()

    response = llm.invoke(f"""
    You are Jarvis's Communication Agent.
    User request: "{user_input}"

    If this involves drafting a message, email, or reply, write a professional draft.
    Keep it concise (under 100 words).
    Include a subject line if it's an email.

    If no communication is requested, say "No communication needed."
    """)

    draft = response.content
    needs_approval = "No communication needed" not in draft and len(draft) > 20

    return {
        "communication_output": draft,
        "needs_approval": needs_approval
    }


def risk_manager_node(state: JarvisState) -> dict:
    all_text = " ".join([
        state.get("scheduler_output", ""),
        state.get("research_output", ""),
        state.get("communication_output", ""),
    ]).lower()

    sensitive_patterns = ["password", "ssn", "credit card", "123-45-", "secret"]

    for pattern in sensitive_patterns:
        if pattern in all_text:
            return {
                "risk_check": f"FAIL - Contains sensitive pattern: '{pattern}'",
                "final_response": "🔒 I cannot process that request for your safety."
            }

    if state.get("needs_approval", False):
        return {
            "risk_check": "PASS - Needs user approval",
            "final_response": state.get("final_response", "Awaiting approval...")
        }

    return {"risk_check": "PASS - Safe to proceed"}


def finalizer_node(state: JarvisState) -> dict:
    intent = state.get("intent", "general")
    user_input = state["user_input"]

    if intent == "schedule" and state.get("scheduler_output"):
        final = f"🗓️ {state['scheduler_output']}"
    elif intent == "research" and state.get("research_output"):
        final = f"📚 {state['research_output']}"
    elif intent == "communication" and state.get("communication_output"):
        final = f"📧 {state['communication_output']}"
    else:
        llm = get_llm()
        response = llm.invoke(f"""
        You are Jarvis, a helpful personal assistant.
        User said: "{user_input}"

        Respond conversationally, concisely, and helpfully.
        If asked about capabilities, mention you can help with scheduling, research, and drafting communications.
        """)
        final = response.content

    return {"final_response": final}


# ============================================================
# STEP 4: Routing Logic
# ============================================================

def route_after_supervisor(state: JarvisState) -> str:
    intent = state.get("intent", "general")
    if intent == "schedule":
        return "scheduler"
    elif intent == "research":
        return "research"
    elif intent == "communication":
        return "communication"
    else:
        return "finalizer"


def route_after_risk(state: JarvisState) -> str:
    risk = state.get("risk_check", "FAIL")
    if risk.startswith("PASS"):
        return "finalizer"
    else:
        return END


# ============================================================
# STEP 5: Build the Graph
# ============================================================

def build_jarvis_graph():
    builder = StateGraph(JarvisState)

    builder.add_node("supervisor", supervisor_node)
    builder.add_node("scheduler", scheduler_node)
    builder.add_node("research", research_node)
    builder.add_node("communication", communication_node)
    builder.add_node("risk_manager", risk_manager_node)
    builder.add_node("finalizer", finalizer_node)

    builder.set_entry_point("supervisor")

    builder.add_conditional_edges("supervisor", route_after_supervisor)
    builder.add_edge("scheduler", "risk_manager")
    builder.add_edge("research", "risk_manager")
    builder.add_edge("communication", "risk_manager")
    builder.add_conditional_edges("risk_manager", route_after_risk)
    builder.add_edge("finalizer", END)

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


# ============================================================
# STEP 6: Runner
# ============================================================

def run_jarvis():
    graph = build_jarvis_graph()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print("=" * 50)
    print("🤖 JARVIS (Claude-powered): Ready for duty.")
    print("💡 I can help with: scheduling, research, communications, and general chat.")
    print("⚠️  All actions require your approval before I execute.")
    print("🛑 Type 'exit' or 'quit' to shut me down.")
    print("=" * 50)

    while True:
        user_input = input("\n👤 You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("🤖 Jarvis: Goodbye, sir.")
            break
        if not user_input:
            continue

        initial_state: JarvisState = {
            "user_input": user_input,
            "intent": None,
            "extracted_details": None,
            "scheduler_output": None,
            "research_output": None,
            "communication_output": None,
            "risk_check": None,
            "final_response": "",
            "needs_approval": False,
            "approved": False,
            "iteration": 0,
        }

        try:
            result = graph.invoke(initial_state, config)
        except Exception as e:
            print(f"❌ Error: {e}")
            continue

        if result.get("needs_approval", False) and result.get("risk_check", "").startswith("PASS"):
            print(f"\n🤖 Jarvis says: {result.get('final_response', '')}")
            approve = input("\n✅ Approve and proceed? (y/n): ").strip().lower()
            if approve == "y":
                print("✅ Approved! Executing...")
                print("📝 [Paper Mode] Action logged.")
            else:
                print("❌ Action cancelled.")
        else:
            print(f"\n🤖 Jarvis: {result.get('final_response', '')}")


if __name__ == "__main__":
    run_jarvis()
