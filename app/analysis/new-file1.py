"""
langgraph_swot_graph.py

Cleaned version: LangGraph pipeline that fetches top5, normalizes, runs a generator (LLM with tools),
and a reflector in a loop until stop criteria. History and messages are preserved between nodes.

Requirements (same as before):
 - langgraph
 - langchain (init_chat_model, @tool)
 - pandas
 - user_functions.py providing:
     fetch_top5_brands_by_revenue(product, country) -> List[Dict]
     normalize_and_validate(list_of_dicts) -> pd.DataFrame
     rows_to_json_safe_all_columns(df) -> List[Dict]
     swot_analysis(df, pivot_brand) -> Dict
"""

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

# from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from tools import swot_analysis as swot_tool
from market_tool import deep_market_analysis
from reviews_summary_tool import fetch_reviews_summary

# Import user-provided functions
try:
    from user_functions import (
        fetch_top5_brands_by_revenue as fetch_top5_brands,
        normalize_and_validate,
        rows_to_json_safe_all_columns as df_to_json,
    )
except Exception as e:
    raise ImportError(
        "Couldn't import required user functions. Provide them in user_functions.py "
        "or update the import path. Required functions:\n"
        "- fetch_top5_brands_by_revenue(product, country) -> List[Dict]\n"
        "- normalize_and_validate(list_of_dicts) -> pd.DataFrame\n"
        "- rows_to_json_safe_all_columns(df) -> List[Dict]\n"
        "- swot_analysis(df, pivot_brand) -> Dict\n"
    ) from e


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
    raw_top5: List[Dict[str, Any]]
    pivot: str
    last_df_json: List[Dict[str, Any]]
    # messages preserved across nodes (use add_messages to indicate message semantics to the graph)
    messages: Annotated[list, add_messages]
    # history is append-only between iterations
    history: Annotated[list, operator.add]
    iteration: int
    last_generator_text: str
    last_reflector_text: str
    final_analysis: str


# -------------------------
# Nodes
# -------------------------

def node_fetch_top5(state: AgentState) -> dict:
    product = state["product"]
    country = state["country"]
    raw_list = fetch_top5_brands(product, country)
    return {"raw_top5": raw_list}


def node_normalize(state: AgentState) -> dict:
    raw = state.get("raw_top5", [])
    df = normalize_and_validate(raw)
    pivot_brand = state["raw_top5"][0]["brand_name"] if state["raw_top5"] else ""
    state["pivot"] = pivot_brand
    df_json = df_to_json(df)
    # also keep the df itself in case you want to inspect it later (not required by graph/state schema)
    return {"last_df_json": df_json}


def node_generator(state: AgentState) -> dict:
    """
    Generator node:
      - increments iteration
      - builds prompt including a small df preview
      - invokes the LLM_gen bound to tools (so model may call swot_tool, deep_market_analysis, fetch_reviews_summary)
      - saves the LLM_gen response text to state and appends iteration entry to history
      - persists messages (conversation) back into state so reflector/next generator see the context
    """
    iteration = state.get("iteration", 0) + 1
    df_preview = json.dumps(state.get("last_df_json", [])[:5], default=str)

    system_msg = (
        "You are a strategic insights generator that decides intelligently when to use tools or respond directly."
        "You have access to the following tools:"
        " - `swot_tool(df_json, pivot_brand)`: returns a dictionary  Dict[str, List[str]], containing fields strength, weakness, opportuunities and threats as SWOT analysis for the pivot brand."
        " - `deep_market_analysis: returns a detailed Dict[str, Any] market analysis."
        " - `fetch_reviews_summary: returns a dictionary Dict[str, Any] summarizing customer reviews."
        "Always reason about which tool to use before generating insights. Clearly mention the tool name when you use one."
        "Only generate data-driven insights using results from these tools — never fabricate or infer data beyond what is provided or returned by a tool."
        "If no relevant tool applies or data is insufficient, state that clearly instead of guessing."
        "For small or straightforward queries (e.g., factual, definitional, or conversational), handle them directly without invoking any tool or sending output to the reflector."
        "For deeper analytical or brand-specific queries, use the appropriate tool and generate structured insights based on its output."
    )

    # Reuse preserved conversation messages if present
    messages = list(state.get("messages", []))  # copy to avoid mutating in place

    messages.append(SystemMessage(content=system_msg))

    user_prompt = (
        "Task: Generate 4 concise, data-driven insights for the given pivot brand (each 1-2 sentences), "
        "followed by a clear, practical action suggestion for each insight. Return the output as a well-structured JSON object.\n\n"
        "Requirements:\n"
        "1. Every insight must reference at least one concrete numeric value or measurable comparison from the provided data preview "
        "(e.g., total_revenue_usd, avg_rating, avg_sentiment_score, units_sold, or competitor benchmarks).\n"
        "2. Insights should be factual, not speculative — rely strictly on the provided data or tool outputs.\n"
        "3. If numeric or comparative context is missing in the data preview, use an appropriate tool to obtain it.\n\n"
        f"Data Preview:\n{df_preview}\n\n"
        "Available Tools:\n"
        "- `swot_tool(df_json, pivot_brand)`: Returns a structured SWOT analysis JSON for the pivot brand.\n"
        "- `deep_market_analysis(df_json, pivot_brand)`: Returns detailed quantitative insights about market position and trends.\n"
        "- `fetch_reviews_summary(df_json, pivot_brand)`: Returns aggregated customer review insights.\n\n"
        "Usage Rules:\n"
        "- Reason about which tool best supports the insight generation and call it if needed.\n"
        "- If you use a tool, incorporate its returned data directly into your insights and mention that tool by name.\n"
        "- If the data preview already provides sufficient numeric evidence, generate insights directly without tool invocation.\n"
        "- Never invent or approximate figures — use only what exists in the data or tool outputs.\n\n"
        "Output Format:\n"
        "{\n"
        '  "insights": [\n'
        '    {"insight": "...", "action": "..."},\n'
        '    {"insight": "...", "action": "..."},\n'
        '    {"insight": "...", "action": "..."},\n'
        '    {"insight": "...", "action": "..."}\n'
        "  ]\n"
        "}\n\n"
        "Respond with a single assistant message containing either the tool call (if needed) followed by the final JSON insights."
    )

    messages.append(HumanMessage(content=user_prompt))

    # Call LLM (LangChain will run tools if model requests them)
    result = llm_with_tools.invoke(messages)

    # Extract text/content
    content = getattr(result, "content", getattr(result, "text", str(result)))
   
    # append a proper assistant message into messages to preserve structure
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
    Reflector node:
      - critiques the generator output
      - replies with 'decision: CONVERGED' or 'decision: REVISE' plus a short critique and checks
      - persists its message and appends to history
    """
    system_msg = (
        "You are a critical analyst. Given the generator output, point out missing evidence, "
        "contradictions, and whether the work should 'CONVERGE' or 'REVISE'. Be concise and provide up to 3 checks."
    )
    generator_text = state.get("last_generator_text", "")

    messages = list(state.get("messages", []))
    messages.append(SystemMessage(content=system_msg))
    messages.append(
        HumanMessage(
            content=(
                f"Generator output:\n{generator_text}\n\n"
                "Please respond starting with: 'decision: CONVERGED' or 'decision: REVISE', "
                "then a short critique and up to 3 checks/questions."
            )
        )
    )

    # Use plain llm (no tools) to reflect
    reflector_result = llm_reflector.invoke(messages)
    critique_text = getattr(reflector_result, "content", getattr(reflector_result, "text", str(reflector_result)))

    messages.append(AIMessage(content=critique_text))
    history_entry = {"iteration": state.get("iteration", 0), "reflector_text": critique_text}
    return {"last_reflector_text": critique_text, "history": [history_entry], "messages": messages}



# -------------------------
# Stop criteria / conditional edge
# -------------------------
MAX_ITERS = 3
SIM_THRESHOLD = 0.86  # textual similarity threshold to detect convergence

def should_continue(state: AgentState) -> Literal["reflector", "end"]:
    it = state.get("iteration", 0)
    if it >= MAX_ITERS:
        return END

    last_ref = (state.get("last_reflector_text") or "").lower()
    if "converge" in last_ref or "converged" in last_ref:
        return END

    # textual similarity check between last two generator outputs in history
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

    graph.add_node("fetch_top5", node_fetch_top5)
    graph.add_node("normalize", node_normalize)
    graph.add_node("generator", node_generator)
    graph.add_node("reflector", node_reflector)

    graph.add_edge(START, "fetch_top5")
    graph.add_edge("fetch_top5", "normalize")
    graph.add_edge("normalize", "generator")

    graph.add_conditional_edges("generator", should_continue, ["reflector", END])
    graph.add_edge("reflector", "generator")

    return graph


# -------------------------
# Final analysis composer (returns human-readable language)
# -------------------------
import re   # add this along with your other imports at the top of the file

def compose_final_analysis(state: AgentState) -> Any:
    """
    Attempt to produce a human-readable final analysis:
      - If last generator text contains JSON (insights), extract it robustly and generate a readable summary.
      - If parsing fails, try literal_eval, or fall back to the latest generator text (trimmed).
    """
    hist = state.get("history", []) or []
    gen_texts = [h.get("generator_text") for h in hist if h.get("generator_text")]
    if not gen_texts:
        gen_texts = [state.get("last_generator_text", "")] if state.get("last_generator_text") else []

    last = gen_texts[-1] if gen_texts else ""

    # Helper to convert insights JSON -> readable paragraphs
    def insights_to_text(obj):
        if isinstance(obj, dict) and "insights" in obj and isinstance(obj["insights"], list):
            paragraphs = []
            for i, itm in enumerate(obj["insights"], 1):
                insight = itm.get("insight") or itm.get("text") or itm.get("summary") or ""
                action = itm.get("action") or itm.get("recommendation") or ""
                paragraphs.append(f"{i}. {insight.strip()} Action: {action.strip()}")
            return "\n".join(paragraphs)
        # generic list/dict -> pretty json string
        return json.dumps(obj, indent=2, ensure_ascii=False)

    # --- New helper: extract first JSON-like object/array from a larger LLM string ---
    def extract_json_like(text: str):
        if not text:
            return None
        # Find the first {...} or [...] block (greedy inside, DOTALL so newlines are included)
        m = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
        if not m:
            return None
        candidate = m.group(1).strip()
        # Try strict JSON
        try:
            return json.loads(candidate)
        except Exception:
            # Try Python literal eval (accepts single quotes, tuples, etc.)
            try:
                return ast.literal_eval(candidate)
            except Exception:
                return None

    # Try to extract JSON-like content from the model output
    parsed = extract_json_like(last)
    if parsed is not None:
        return insights_to_text(parsed)

    # If extraction failed, fall back to the earlier approach (keep it readable)
    # previous fallback: try json.loads(last) -> ast.literal_eval(last) -> plain preview
    try:
        parsed = json.loads(last)
        return insights_to_text(parsed)
    except Exception:
        try:
            parsed = ast.literal_eval(last)
            return insights_to_text(parsed)
        except Exception:
            preview = " ".join(gen_texts[-3:])
            return preview.strip()


# -------------------------
# Invocation / entrypoint
# -------------------------
if __name__ == "__main__":
    initial_state: AgentState = {
        "product": "shampoo",
        "country": "Japan",
        # "raw_top5": [],
        # "pivot": "",
        # "last_df_json": [],
         "messages": [
            {"role": "user", "content": "how are reviews like for shampoo brands in Japan."}
        ],
        "history": [],
        "iteration": 0,
        "last_generator_text": "",
        "last_reflector_text": "",
        "final_analysis": "",
    }

    graph = build_agency_graph()
    compiled_agent = graph.compile()
    final_state = compiled_agent.invoke(initial_state)


    final_analysis = compose_final_analysis(final_state)
    final_state["final_analysis"] = final_analysis

    print("=== HISTORY ===")
    for entry in final_state.get("history", []):
        print(entry)
    print("\n=== FINAL ANALYSIS ===")
    # print(final_state["messages"][-1].content)

    print(final_analysis)
