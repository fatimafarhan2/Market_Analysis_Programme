
from typing import Annotated, TypedDict, Literal, Any, List, Dict
import operator
import difflib
import json
import ast
import pandas as pd
import os
import json
from dotenv import load_dotenv
from supabase import create_client
import re   

# --- Config / load env ---
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Please set SUPABASE_URL and SUPABASE_KEY in your environment or .env file")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# LangChain imports
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from tool1 import swot_analysis as swot_tool
from market_tool import deep_market_analysis
from reviews_summary_tool import fetch_reviews_summary

tools = [swot_tool, deep_market_analysis, fetch_reviews_summary]
llm_gen = init_chat_model("google_genai:gemini-2.0-flash")
llm_reflector = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.7
)
llm_with_tools = llm_gen.bind_tools(tools)

# -------------------------
# State definition
# -------------------------

class AgentState(TypedDict):
    product: str
    country: str
    messages: Annotated[list, add_messages]
    history: Annotated[list, operator.add]
    iteration: int
    last_generator_text: str
    last_reflector_text: str
    final_analysis: str

# -------------------------
# Nodes
# -------------------------

def node_generator(state: AgentState) -> dict:
    """
    Generator node with improved prompting for tool usage
    """
    iteration = state.get("iteration", 0) + 1

    # SIMPLIFIED SYSTEM MESSAGE - Focus on what to do, not abstract theory
    system_msg = """You are a strategic market analyst with access to three data analysis tools.

Your job: Generate 4 data-driven insights with actionable recommendations.

AVAILABLE TOOLS (No parameters needed - they access pre-filtered data automatically):

1. **deep_market_analysis** - Returns market aggregates, brand rankings, market shares, competition metrics
   Output includes: total_revenue_usd, total_units_sold, brand_summary_topN, market_share, HHI, correlations
   
2. **swot_tool** - Returns competitive SWOT analysis with numeric comparisons
   Output includes: Strengths, Weaknesses, Opportunities, Threats with metrics
   
3. **fetch_reviews_summary** - Returns customer sentiment, keywords, return rates, sample reviews
   Output includes: avg_rating, total_num_reviews, top_common_keywords, avg_returns_rate

HOW TO USE TOOLS:

When you need data, simply call the appropriate tool(s). The model will automatically invoke them.
After receiving tool outputs, use the numeric data to create insights.

Example workflow:
1. User asks about market trends for shampoo in Japan
2. You think: "I need market data" → Call deep_market_analysis
3. Tool returns data with revenue, market shares, etc.
4. You think: "I also need customer sentiment" → Call fetch_reviews_summary  
5. Tool returns ratings, keywords, etc.
6. Generate 4 insights using the ACTUAL NUMBERS from both tools

CRITICAL RULES:
- Every insight MUST reference specific numbers from tool outputs (e.g., "Revenue of $45M", "Market share 33%")
- Never generate insights BEFORE calling tools
- Never invent numbers - only use data from tool outputs
- Call multiple tools if needed for comprehensive analysis

OUTPUT FORMAT:
```json
{
  "insights": [
    {"insight": "Specific finding with numbers", "action": "Practical recommendation"},
    {"insight": "Another finding with metrics", "action": "Another action"},
    {"insight": "Third insight with data", "action": "Third action"},
    {"insight": "Fourth insight with evidence", "action": "Fourth action"}
  ],
  "tools_used": ["tool1", "tool2"],
  "provenance": {
    "tool1": {"key_metrics": "values used"},
    "tool2": {"key_metrics": "values used"}
  }
}
```

Remember: Call tools FIRST, then generate insights from their outputs!"""

    # Reuse preserved conversation messages
    messages = list(state.get("messages", []))

    # Only add system message if not already present
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages.append(SystemMessage(content=system_msg))

    # Simpler user prompt that doesn't repeat instructions
    user_prompt = (
        "Generate 4 concise, data-driven insights based on the user's query. "
        "Each insight should include:\n"
        "1. A specific finding with numeric evidence from tool outputs\n"
        "2. A practical, actionable recommendation\n\n"
        "First call the appropriate tools to gather data, then create insights from the results."
    )

    # Only add user prompt on first iteration
    if iteration == 1:
        messages.append(HumanMessage(content=user_prompt))

    # Call LLM with tools
    result = llm_with_tools.invoke(messages)

    # Extract content
    content = getattr(result, "content", getattr(result, "text", str(result)))
   
    # Append assistant message
    messages.append(AIMessage(content=content))

    history_entry = {"iteration": iteration, "generator_text": content}

    return {
        "iteration": iteration,
        "last_generator_text": content,
        "history": [history_entry],
        "messages": messages,
    }


def node_reflector(state: AgentState) -> dict:
    """
    Reflector node with improved critique logic
    """
    system_msg = """You are a quality control analyst reviewing market insights.

Check if the generator:
1. ✓ Actually CALLED tools (not just talked about calling them)
2. ✓ Used REAL NUMBERS from tool outputs in insights
3. ✓ Provided 4 complete insights with actions
4. ✓ Made insights specific and actionable (not generic)

If ANY of these are missing, respond with:
"decision: REVISE" followed by what needs to be fixed.

If ALL requirements are met, respond with:
"decision: CONVERGED" followed by brief confirmation.

Be specific about what's wrong (e.g., "No tool calls detected" or "Insights lack numeric evidence")."""

    generator_text = state.get("last_generator_text", "")
    messages = list(state.get("messages", []))
    
    messages.append(SystemMessage(content=system_msg))
    messages.append(
        HumanMessage(
            content=(
                f"Generator output:\n{generator_text}\n\n"
                "Review this output. Did they call tools and use real data? "
                "Respond with 'decision: CONVERGED' or 'decision: REVISE' plus brief explanation."
            )
        )
    )

    reflector_result = llm_reflector.invoke(messages)
    critique_text = getattr(reflector_result, "content", getattr(reflector_result, "text", str(reflector_result)))

    messages.append(AIMessage(content=critique_text))
    history_entry = {"iteration": state.get("iteration", 0), "reflector_text": critique_text}
    
    return {
        "last_reflector_text": critique_text, 
        "history": [history_entry], 
        "messages": messages
    }


# -------------------------
# Stop criteria / conditional edge
# -------------------------
MAX_ITERS = 3
SIM_THRESHOLD = 0.86

def should_continue(state: AgentState) -> Literal["reflector", "end"]:
    it = state.get("iteration", 0)
    if it >= MAX_ITERS:
        return END

    last_ref = (state.get("last_reflector_text") or "").lower()
    if "converge" in last_ref or "converged" in last_ref:
        return END

    # Check similarity between last two iterations
    hist = state.get("history", []) or []
    gen_texts = [h.get("generator_text") for h in hist if h.get("generator_text")]
    if len(gen_texts) >= 2:
        a, b = gen_texts[-2], gen_texts[-1]
        sim = difflib.SequenceMatcher(None, (a or ""), (b or "")).ratio()
        if sim >= SIM_THRESHOLD:
            return END

    return "reflector"


# -------------------------
# Build graph
# -------------------------
def build_agency_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("generator", node_generator)
    graph.add_node("reflector", node_reflector)

    graph.add_edge(START, "generator")
    graph.add_conditional_edges("generator", should_continue, ["reflector", END])
    graph.add_edge("reflector", "generator")
    return graph


# -------------------------
# Final analysis composer
# -------------------------

def compose_final_analysis(state: AgentState) -> Any:
    """
    Extract and format the final insights
    """
    hist = state.get("history", []) or []
    gen_texts = [h.get("generator_text") for h in hist if h.get("generator_text")]
    if not gen_texts:
        gen_texts = [state.get("last_generator_text", "")] if state.get("last_generator_text") else []

    last = gen_texts[-1] if gen_texts else ""

    def insights_to_text(obj):
        if isinstance(obj, dict) and "insights" in obj and isinstance(obj["insights"], list):
            paragraphs = []
            for i, itm in enumerate(obj["insights"], 1):
                insight = itm.get("insight") or itm.get("text") or ""
                action = itm.get("action") or itm.get("recommendation") or ""
                if insight:
                    paragraphs.append(f"{i}. {insight.strip()}")
                    if action:
                        paragraphs.append(f"   Action: {action.strip()}")
            return "\n".join(paragraphs)
        return json.dumps(obj, indent=2, ensure_ascii=False)

    def extract_json_like(text: str):
        if not text:
            return None
        m = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
        if not m:
            return None
        candidate = m.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            try:
                return ast.literal_eval(candidate)
            except Exception:
                return None

    parsed = extract_json_like(last)
    if parsed is not None:
        return insights_to_text(parsed)

    try:
        parsed = json.loads(last)
        return insights_to_text(parsed)
    except Exception:
        try:
            parsed = ast.literal_eval(last)
            return insights_to_text(parsed)
        except Exception:
            return last.strip()


# -------------------------
# Invocation / entrypoint
# -------------------------
if __name__ == "__main__":
    initial_state: AgentState = {
        "product": "shampoo",
        "country": "Japan",
        "messages": [
            HumanMessage(content="What are market trends for shampoo brands in Japan?")
        ],
        "history": [],
        "iteration": 0,
        "last_generator_text": "",
        "last_reflector_text": "",
        "final_analysis": "",
    }

    graph = build_agency_graph()
    compiled_agent = graph.compile()

    print("🚀 Starting market analysis agent...\n")
    final_state = compiled_agent.invoke(initial_state)

    final_analysis = compose_final_analysis(final_state)
    final_state["final_analysis"] = final_analysis

    print("\n" + "="*60)
    print("ITERATION HISTORY")
    print("="*60)
    for entry in final_state.get("history", []):
        iter_num = entry.get("iteration", "?")
        if "generator_text" in entry:
            text = entry["generator_text"][:200] + "..." if len(entry.get("generator_text", "")) > 200 else entry.get("generator_text", "")
            print(f"\n[Iteration {iter_num} - Generator]")
            print(text)
        if "reflector_text" in entry:
            print(f"\n[Iteration {iter_num} - Reflector]")
            print(entry["reflector_text"])

    print("\n" + "="*60)
    print("FINAL INSIGHTS")
    print("="*60)
    print(final_analysis)