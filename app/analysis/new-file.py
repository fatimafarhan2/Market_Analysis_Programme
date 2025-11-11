# # with new swot from tool1.py and agent state not having an access to dataset

# """
# langgraph_swot_graph.py

# Cleaned version: LangGraph implementation where generator (LLM with tools),
# and a reflector in a loop until stop criteria. History and messages are preserved between nodes.

# Requirements (same as before):
#  - langgraph
#  - langchain (init_chat_model, @tool)
#  - pandas
# """

# from typing import Annotated, TypedDict, Literal, Any, List, Dict
# import operator
# import difflib
# import json
# import ast
# import pandas as pd
# import os
# import json
# from dotenv import load_dotenv
# from supabase import create_client
# import re   

# # --- Config / load env ---
# load_dotenv()
# SUPABASE_URL = os.getenv("SUPABASE_URL")
# SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# if not SUPABASE_URL or not SUPABASE_KEY:
#     raise RuntimeError("Please set SUPABASE_URL and SUPABASE_KEY in your environment or .env file")

# supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# # LangGraph imports
# from langgraph.graph import StateGraph, START, END
# from langgraph.graph.message import add_messages

# # LangChain imports
# from langchain_core.tools import tool
# from langchain.chat_models import init_chat_model
# from langchain_groq import ChatGroq
# from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# # from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
# from tool1 import swot_analysis as swot_tool
# from market_tool import deep_market_analysis
# from reviews_summary_tool import fetch_reviews_summary

# tools = [swot_tool, deep_market_analysis, fetch_reviews_summary]
# llm_gen = init_chat_model("google_genai:gemini-2.0-flash")
# llm_reflector = ChatGroq(
#     model_name="llama-3.3-70b-versatile",
#     temperature=0.7
# )
# llm_with_tools = llm_gen.bind_tools(tools)

# # -------------------------
# # State definition
# # -------------------------

# class AgentState(TypedDict):
#     product: str
#     country: str
#     # messages preserved across nodes (use add_messages to indicate message semantics to the graph)
#     messages: Annotated[list, add_messages]
#     # history is append-only between iterations
#     history: Annotated[list, operator.add]
#     iteration: int
#     last_generator_text: str
#     last_reflector_text: str
#     final_analysis: str

# # -------------------------
# # Nodes
# # -------------------------

# def node_generator(state: AgentState) -> dict:
#     """
#     Generator node:
#       - increments iteration
#       - invokes the LLM_gen bound to tools (so model may call swot_tool, deep_market_analysis, fetch_reviews_summary)
#       - saves the LLM_gen response text to state and appends iteration entry to history
#       - persists messages (conversation) back into state so reflector/next generator see the context
#     """
#     iteration = state.get("iteration", 0) + 1

#     system_msg = (
#     "You are a ReAct-based strategic insights generator. ReAct here means you must alternate "
#     "between explicit reasoning steps and concrete actions (tool calls). Follow the ReAct loop: "
#     "1) Thought — short private reasoning about what is needed, 2) Action — call the appropriate tool(s), "
#     "3) Observation — ingest the tool output, 4) Thought — decide next step, and 5) Final Answer — produce the requested insights. "
#     "Every time you call a tool, record which tool you called and why. Do not skip the Thought → Action → Observation pattern when a tool is needed.\n\n"
#     "For the tool call you don't need any data, tools will access the dataset directly.\n\n"

#     "Primary rules and guardrails:\n"
#     "- Only produce final, data-driven insights after you have used the available data and/or tool outputs. Never invent numbers or facts. "
#     "- When a query is small, factual, or conversational and does not require data aggregation or cross-row analysis, you may respond directly without calling tools. "
#     "- For any brand- or market-level insight that depends on numeric comparisons, distributions, or aggregated metrics, you MUST call the appropriate tool(s) and base conclusions strictly on their output. "
#     "- Always name the specific tool(s) you used in the final answer (for provenance).\n\n"

#     "Available tools and expected outputs (use these names exactly when referencing):\n"
#     "1) deep_market_analysis"
#     "   - Purpose: produce aggregated market-level metrics and segmented summaries.\n"
#     "   - Typical output keys: notes, errors, meta_rows_fetched, global_summary (with total_revenue_usd, total_units_sold, total_brands, total_products, avg_rating, avg_sentiment_score, avg_roms, avg_marketing_efficiency_ratio), regions, countries, brand_summary_topN, product_summary_topN, brand_market_share_topN, country_market_share_topN, global_brand_top_bottom, global_product_top_bottom, hhi_by_brand, hhi_by_country, correlation_matrix, strategy_insights.\n"
#     "   - Use-case: market share, cross-country splits, HHIs, correlations, top/bottom lists, global aggregates.\n\n"
#     "2) swot_tool"
#     "   - Purpose: return a structured SWOT.\n"
#     "   - Typical output: string or JSON-like sections Strengths, Weaknesses, Opportunities, Threats with numeric comparisons where available (e.g., 'avg_rating: 3.93 vs avg 3.98').\n"
#     "   - Use-case: to ground recommendations in relative strengths/weaknesses vs competitors.\n\n"
#     "3) fetch_reviews_summary"
#     "   - Purpose: return aggregated reviews and qualitative signals (top keywords, avg_rating, avg_returns_rate, total_num_reviews, representative review snippets, product-level summary rows).\n"
#     "   - Typical output keys: notes, errors, meta_rows_fetched, reviews_summary (list of product-level dicts with product_id, product_name, brand_id, brand_name, total_units_sold, avg_returns_rate, avg_rating, total_num_reviews, top_common_keywords, review[]).\n"
#     "   - Use-case: surface customer sentiment, common complaints/praise, return rates, and qualitative evidence for actions.\n\n"

#     "How to reason and choose tools:\n"
#     "- Start with a one-line Thought describing the analytic gap you must fill (example: 'Need market share by country to compare pivot; call deep_market_analysis').\n"
#     "- Prefer the minimal toolset: call swot_tool if you need a structured strength/weakness comparison; call fetch_reviews_summary if you need qualitative signals or product-level sentiment; call deep_market_analysis for numeric aggregates, shares and correlations. You may call multiple tools in sequence if the question requires both quantitative and qualitative grounding.\n"
#     "- After each tool call, treat the returned JSON as authoritative. Extract numeric values directly from the tool output and cite them in insights.\n\n"

#     "Formatting rules and outputs:\n"
#     "- Final output must be machine-friendly JSON when asked for insights. Example final shape for 4 insights:\n"
#     '  { "insights": [ {"insight": "... (1-2 sentences referencing numeric figures)", "action": "..."}, ... 4 items ], "tools_used": ["deep_market_analysis","swot_tool"], "provenance": { "deep_market_analysis": <the tool output or reference>, "swot_tool": <the tool output or reference> } }\n'
#     "- In the 'insight' text always reference at least one numeric field from tool output (e.g., 'total_revenue_usd = 307,110,755.88' or 'share = 0.3379 (34%)').\n"
#     "- Include a short 'tools_used' list naming every tool invoked. Include a 'provenance' field that either embeds the tool output or a pointer to it. If embedding full outputs would be too large, embed the key numeric lines used.\n"
#     "- Keep insights concise (1-2 sentences) and each action practical and measurable.\n\n"

#     "How to handle incomplete or insufficient data:\n"
#     "- If the provided preview lacks required numeric fields, call the appropriate tool. If a tool returns errors or insufficient rows (errors is non-empty or meta_rows_fetched == 0), return a JSON object: {\"insufficient_data\": true, \"errors\": <tool errors>} and stop—do not guess.\n"
#     "- If you can compute a derived metric reliably from tool output (for example: percentage share = brand_revenue / total_revenue_usd), compute it and show the calculation briefly in provenance.\n\n"

#     "On hallucinations and fabrication:\n"
#     "- Never invent competitor names, numbers, or causal claims not supported by tool output. If you infer something (for example: 'lower sentiment may indicate product fit issues'), label the sentence as an inference and keep it a suggestion, not a fact.\n\n"

#     "Final behaviour summary:\n"
#     "- For small, direct queries: answer without tools. For any insight that depends on aggregated numeric evidence or cross-entity comparison, run the ReAct loop and call tools. Always name tools used, include provenance, and return results as structured JSON. If data is insufficient, return a clear insufficient_data response rather than guessing."
# )


#     # Reuse preserved conversation messages if present
#     messages = list(state.get("messages", []))  # copy to avoid mutating in place

#     messages.append(SystemMessage(content=system_msg))

#     user_prompt = (
#         "Task: Generate 4 concise, data-driven insights for the user query.  (each 1-2 sentences), "
#         "followed by a clear, practical action suggestion for each insight. Return the output as a well-structured JSON object.\n\n"
#         "Requirements:\n"
#         "1. Every insight must reference at least one concrete numeric value or measurable comparison from the provided data preview "
#         "(e.g., total_revenue_usd, avg_rating, avg_sentiment_score, units_sold, or competitor benchmarks).\n"
#         "2. Insights should be factual, not speculative — rely strictly on the provided data or tool outputs.\n"
#         "3. If numeric or comparative context is missing in the data preview, use an appropriate tool to obtain it.\n\n"
#         "Available Tools:\n"
#         "- `swot_tool"
#         "- `deep_market_analysis"
#         "- `fetch_reviews_summary"
#         "Usage Rules:\n"
#         "- Reason about which tool best supports the insight generation and call it if needed.\n"
#         "- If you use a tool, incorporate its returned data directly into your insights and mention that tool by name.\n"
#         "- Never invent or approximate figures — use only what exists in the data or tool outputs.\n\n"
#         "Output Format:\n"
#         "{\n"
#         '  "insights": [\n'
#         '    {"insight": "...", "action": "..."},\n'
#         '    {"insight": "...", "action": "..."},\n'
#         '    {"insight": "...", "action": "..."},\n'
#         '    {"insight": "...", "action": "..."}\n'
#         "  ]\n"
#         "}\n\n"
#         "Respond with a single assistant message containing either the tool call (if needed) followed by the final JSON insights."
#     )

#     messages.append(HumanMessage(content=user_prompt))

#     # Call LLM (LangChain will run tools if model requests them)
#     result = llm_with_tools.invoke(messages)

#     # Extract text/content
#     content = getattr(result, "content", getattr(result, "text", str(result)))
   
#     # append a proper assistant message into messages to preserve structure
#     messages.append(AIMessage(content=content))

#     history_entry = {"iteration": iteration, "generator_text": content}

#     return {
#         "iteration": iteration,
#         "last_generator_text": content,
#         "history": [history_entry],
#         "messages": messages,
#     }


# def node_reflector(state: AgentState) -> dict:
#     """
#     Reflector node:
#       - critiques the generator output
#       - replies with 'decision: CONVERGED' or 'decision: REVISE' plus a short critique and checks
#       - persists its message and appends to history
#     """
#     system_msg = (
#         "You are a critical analyst. Given the generator output, point out missing evidence, "
#         "contradictions, and whether the work should 'CONVERGE' or 'REVISE'. Be concise and provide up to 3 checks."
#     )
#     generator_text = state.get("last_generator_text", "")

#     messages = list(state.get("messages", []))
#     messages.append(SystemMessage(content=system_msg))
#     messages.append(
#         HumanMessage(
#             content=(
#                 f"Generator output:\n{generator_text}\n\n"
#                 "Please respond starting with: 'decision: CONVERGED' or 'decision: REVISE', "
#                 "then a short critique and up to 3 checks/questions."
#             )
#         )
#     )

#     # Use plain llm (no tools) to reflect
#     reflector_result = llm_reflector.invoke(messages)
#     critique_text = getattr(reflector_result, "content", getattr(reflector_result, "text", str(reflector_result)))

#     messages.append(AIMessage(content=critique_text))
#     history_entry = {"iteration": state.get("iteration", 0), "reflector_text": critique_text}
#     return {"last_reflector_text": critique_text, "history": [history_entry], "messages": messages}



# # -------------------------
# # Stop criteria / conditional edge
# # -------------------------
# MAX_ITERS = 3
# SIM_THRESHOLD = 0.86  # textual similarity threshold to detect convergence

# def should_continue(state: AgentState) -> Literal["reflector", "end"]:
#     it = state.get("iteration", 0)
#     if it >= MAX_ITERS:
#         return END

#     last_ref = (state.get("last_reflector_text") or "").lower()
#     if "converge" in last_ref or "converged" in last_ref:
#         return END

#     # textual similarity check between last two generator outputs in history
#     hist = state.get("history", []) or []
#     gen_texts = [h.get("generator_text") for h in hist if h.get("generator_text")]
#     if len(gen_texts) >= 2:
#         a, b = gen_texts[-2], gen_texts[-1]
#         sim = difflib.SequenceMatcher(None, (a or ""), (b or "")).ratio()
#         if sim >= SIM_THRESHOLD:
#             return END

#     return "reflector"


# # -------------------------
# # Build graph
# # -------------------------
# def build_agency_graph() -> StateGraph:
#     graph = StateGraph(AgentState)

#     graph.add_node("generator", node_generator)
#     graph.add_node("reflector", node_reflector)

#     graph.add_edge(START, "generator")
#     graph.add_conditional_edges("generator", should_continue, ["reflector", END])
#     graph.add_edge("reflector", "generator")
#     return graph


# # -------------------------
# # Final analysis composer (returns human-readable language)
# # -------------------------

# def compose_final_analysis(state: AgentState) -> Any:
#     """
#     Attempt to produce a human-readable final analysis:
#       - If last generator text contains JSON (insights), extract it robustly and generate a readable summary.
#       - If parsing fails, try literal_eval, or fall back to the latest generator text (trimmed).
#     """
#     hist = state.get("history", []) or []
#     gen_texts = [h.get("generator_text") for h in hist if h.get("generator_text")]
#     if not gen_texts:
#         gen_texts = [state.get("last_generator_text", "")] if state.get("last_generator_text") else []

#     last = gen_texts[-1] if gen_texts else ""

#     # Helper to convert insights JSON -> readable paragraphs
#     def insights_to_text(obj):
#         if isinstance(obj, dict) and "insights" in obj and isinstance(obj["insights"], list):
#             paragraphs = []
#             for i, itm in enumerate(obj["insights"], 1):
#                 insight = itm.get("insight") or itm.get("text") or itm.get("summary") or ""
#                 action = itm.get("action") or itm.get("recommendation") or ""
#                 paragraphs.append(f"{i}. {insight.strip()} Action: {action.strip()}")
#             return "\n".join(paragraphs)
#         # generic list/dict -> pretty json string
#         return json.dumps(obj, indent=2, ensure_ascii=False)

#     # --- New helper: extract first JSON-like object/array from a larger LLM string ---
#     def extract_json_like(text: str):
#         if not text:
#             return None
#         # Find the first {...} or [...] block (greedy inside, DOTALL so newlines are included)
#         m = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
#         if not m:
#             return None
#         candidate = m.group(1).strip()
#         # Try strict JSON
#         try:
#             return json.loads(candidate)
#         except Exception:
#             # Try Python literal eval (accepts single quotes, tuples, etc.)
#             try:
#                 return ast.literal_eval(candidate)
#             except Exception:
#                 return None

#     # Try to extract JSON-like content from the model output
#     parsed = extract_json_like(last)
#     if parsed is not None:
#         return insights_to_text(parsed)

#     # If extraction failed, fall back to the earlier approach (keep it readable)
#     # previous fallback: try json.loads(last) -> ast.literal_eval(last) -> plain preview
#     try:
#         parsed = json.loads(last)
#         return insights_to_text(parsed)
#     except Exception:
#         try:
#             parsed = ast.literal_eval(last)
#             return insights_to_text(parsed)
#         except Exception:
#             preview = " ".join(gen_texts[-3:])
#             return preview.strip()

# # -------------------------
# # Invocation / entrypoint
# # -------------------------
# if __name__ == "__main__":
#     initial_state: AgentState = {
#         "product": "shampoo",
#         "country": "Japan",
#          "messages": [
#             {"role": "user", "content": "pwhat are market trends for shampoo brands in Japan."}
#         ],
#         "history": [],
#         "iteration": 0,
#         "last_generator_text": "",
#         "last_reflector_text": "",
#         "final_analysis": "",
#     }

#     graph = build_agency_graph()
#     compiled_agent = graph.compile()

#     final_state = compiled_agent.invoke(initial_state)


#     final_analysis = compose_final_analysis(final_state)
#     final_state["final_analysis"] = final_analysis

#     print("=== HISTORY ===")
#     for entry in final_state.get("history", []):
#         print(entry)
#     print("\n=== FINAL ANALYSIS ===")

#     print(final_analysis)
# Fixed version with clearer prompts for tool usage

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