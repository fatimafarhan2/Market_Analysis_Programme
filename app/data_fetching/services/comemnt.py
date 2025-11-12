from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict, Literal, Any, Dict, Optional
import operator
import difflib
import json
import os
from dotenv import load_dotenv
from supabase import create_client

# LangChain Groq import
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

# Custom tools
from app.analysis.tool1 import swot_analysis as swot_tool
from app.analysis.market_tool import deep_market_analysis
from app.analysis.reviews_summary_tool import fetch_reviews_summary

# Agents
from app.data_fetching.agents.intent_agent import detect_intent
from app.data_fetching.agents.sql_agent import generate_sql
from app.data_fetching.agents.validator_agent import validate_sql_and_execute

# --- Config / load env ---
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Please set SUPABASE_URL and SUPABASE_KEY in your environment or .env file")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# -----------------------------Model loading--------------------
llm_generator = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.7)
llm_reflector = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.7)

TOOLS = [swot_tool, deep_market_analysis, fetch_reviews_summary]

# Create tool map for execution
TOOL_MAP = {
    "swot_analysis": swot_tool,
    "deep_market_analysis": deep_market_analysis,
    "fetch_reviews_summary": fetch_reviews_summary
}

# Bind tools to the model
llm_generator_with_tools = llm_generator.bind_tools(TOOLS)

# ---------------- PIPELINE STATE ----------------
class PipelineState(TypedDict):
    user_query: Optional[str]
    filters: Optional[Dict[str, Any]]
    sql_query: Optional[str]
    result: Optional[Any]
    error: Optional[str]
    inserted: Optional[bool]
    loop_back: Optional[bool]
    retry_count: Optional[int]
    messages: Annotated[list, add_messages]
    history: Annotated[list, operator.add]
    iteration: int
    last_generator_text: str
    last_reflector_text: str
    final_analysis: str

# ---------------- NODES ----------------
def input_node(state: PipelineState, **kwargs):
    """Prompt user for input or refined query in case of errors or loopbacks."""
    temp_query_holder = kwargs.setdefault("temp_query_holder", {})
    print("\n[INPUT NODE] Awaiting user input...")
    MAX_RETRIES = 5

    current_retry = temp_query_holder.get("retry_count", 0)
    if current_retry >= MAX_RETRIES:
        print("[INPUT NODE] Maximum retries reached.")
        return {**state, "error": "Maximum retries reached.", "loop_back": False}

    if temp_query_holder.get("loop_back") or temp_query_holder.get("error"):
        temp_query_holder["retry_count"] = current_retry + 1
        temp_query_holder["user_query"] = input("Please enter a refined query: ")
        print(f"[INPUT NODE] Refined query received: {temp_query_holder['user_query']}")
    elif "user_query" not in temp_query_holder or not temp_query_holder["user_query"]:
        temp_query_holder["user_query"] = input("Please enter your query: ")
        print(f"[INPUT NODE] Initial query received: {temp_query_holder['user_query']}")

    temp_query_holder["error"] = None
    temp_query_holder["loop_back"] = False

    return {**state, "user_query": temp_query_holder["user_query"], "error": None, "loop_back": False,
            "retry_count": temp_query_holder.get("retry_count", 0)}

def intent_node(state: PipelineState, **kwargs):
    """Detect user intent from query and extract filters for SQL."""
    temp_query_holder = kwargs.setdefault("temp_query_holder", {})
    user_query = state.get("user_query")
    print("\n[INTENT NODE] Step 1: Detecting intent from query...")
    print(f"[INTENT NODE] User query: {user_query}")
    try:
        intent = detect_intent(user_query)
        temp_query_holder["filters"] = intent.get("filters", {})
        print(f"[INTENT NODE] Detected filters: {temp_query_holder['filters']}")
        return {**state, "filters": intent.get("filters", {}), "loop_back": False, "error": None}
    except Exception as e:
        print(f"[INTENT NODE] Error in detecting intent: {str(e)}")
        temp_query_holder["error"] = str(e)
        temp_query_holder["loop_back"] = True
        return {**state, "error": str(e), "loop_back": True}

def sqlgen_node(state: PipelineState, **kwargs):
    """Generate SQL query from detected filters."""
    temp_query_holder = kwargs.setdefault("temp_query_holder", {})
    if state.get("error"):
        return state

    print("\n[SQL NODE] Step 2: Generating SQL query...")
    try:
        filters = state.get("filters", {})
        sql_query = generate_sql(filters)
        print(f"[SQL NODE] Generated SQL: {sql_query}")
        return {**state, "sql_query": sql_query}
    except Exception as e:
        print(f"[SQL NODE] Error generating SQL: {str(e)}")
        temp_query_holder["error"] = str(e)
        temp_query_holder["loop_back"] = True
        return {**state, "error": str(e), "loop_back": True}

def validation_node(state: PipelineState, **kwargs):
    """Validate and execute SQL, handle user-correctable errors."""
    temp_query_holder = kwargs.setdefault("temp_query_holder", {})
    if state.get("error"):
        return state

    print("\n[VALIDATION NODE] Step 3: Validating and executing SQL...")
    validation_output = validate_sql_and_execute(state["sql_query"])

    if "error" in validation_output:
        msg = validation_output["error"]
        user_correctable = validation_output.get("user_correctable", False)
        print(f"[VALIDATION NODE] SQL Error: {msg}, user_correctable={user_correctable}")
        if user_correctable:
            temp_query_holder["loop_back"] = True
            temp_query_holder["retry_count"] = temp_query_holder.get("retry_count", 0) + 1
        temp_query_holder["error"] = msg
        return {**state, "error": msg, "loop_back": user_correctable}

    print(f"[VALIDATION NODE] SQL executed successfully, rows returned: {len(validation_output.get('data', []))}")
    return {**state, "result": validation_output.get("data", []), "loop_back": False, "error": None}

def insert_node(state: PipelineState):
    """Insert fetched data into Supabase table."""
    if state.get("error") or not state.get("result"):
        return state
    table_name = "product_flat_table"
    try:
        print(f"[INSERT NODE] Clearing old data in table {table_name}...")
        del_response = supabase.table(table_name).delete().neq("product_id", "").execute()
        if hasattr(del_response, "error") and del_response.error:
            print(f"[INSERT NODE] Error clearing old data: {del_response.error}")
            return {**state, "error": f"Failed to clear old data: {del_response.error}"}

        print(f"[INSERT NODE] Inserting {len(state['result'])} rows into table {table_name}...")
        response = supabase.table(table_name).insert(state["result"]).execute()
        if hasattr(response, "error") and response.error:
            print(f"[INSERT NODE] Error inserting data: {response.error}")
            return {**state, "error": f"Insert failed: {response.error}"}
        print("[INSERT NODE] Data insertion successful.")
        return {**state, "inserted": True}
    except Exception as e:
        print(f"[INSERT NODE] Exception during insert: {str(e)}")
        return {**state, "error": f"Insert exception: {str(e)}"}

# ---------------- FIXED ReAct generator node ----------------
def node_generator(state: PipelineState) -> dict:
    """ReAct-style generator that properly handles tool calls from Groq."""
    user_query = state.get("user_query")
    filters = state.get("filters", {})
    iteration = state.get("iteration", 0) + 1
    messages = list(state.get("messages", []))

    print(f"\n[GENERATOR NODE] Iteration {iteration} - Reasoning with LLM...")

    # Add system message with clear instructions about tools
    if iteration == 1:
        system_prompt = f"""You are a market analysis assistant ReAct Agent with access to specialized tools.

User Query: {user_query}
Detected Filters: {json.dumps(filters, indent=2)}

You have access to these tools:
1. swot_analysis - Performs SWOT analysis on products
2. deep_market_analysis - Provides deep market insights
3. fetch_reviews_summary - Fetches and summarizes product reviews

ReAct agent:
Follow the ReAct pattern:
- First, reason about what data you need.
- Then, call the relevant tool(s).
- Observe tool outputs before deciding next steps.
- Never make up data — always use real numbers from tool outputs.
- When confident, give exactly 4 concise, actionable insights in JSON.
- Its better to make use of as much numerical data as possible

IMPORTANT: 
- Use the tools to gather comprehensive information
- Call multiple tools if needed to provide complete analysis
- After receiving tool results, synthesize them into a coherent response
- Be thorough and use all relevant tools for the query
- Make sure you use numerical vaues to support but dont hallucinate ,use the numerical values you get from tools 


Analyze the query and decide which tools to use."""
        
        messages = [SystemMessage(content=system_prompt)] + messages

    # Call LLM with tools bound
    result = llm_generator_with_tools.invoke(messages)
    
    # Check if the model wants to use tools
    tool_calls = getattr(result, 'tool_calls', [])
    
    if tool_calls:
        print(f"[GENERATOR NODE] Model requested {len(tool_calls)} tool calls")
        
        # Add the AI message with tool calls to history
        messages.append(result)
        
        # Execute each tool call
        for tool_call in tool_calls:
            tool_name = tool_call.get('name')
            tool_args = tool_call.get('args', {})
            tool_id = tool_call.get('id', 'unknown')
            
            print(f"[GENERATOR NODE] Executing tool: {tool_name} with args: {tool_args}")
            
            try:
                # Execute the tool with proper arguments
                if tool_name in TOOL_MAP:
                    # All tools now accept a query string
                    tool_fn = TOOL_MAP[tool_name]
                    # Call the actual function, not the tool wrapper
                    tool_output = tool_fn.func(user_query) if hasattr(tool_fn, 'func') else tool_fn(user_query)
                    print(f"[GENERATOR NODE] Tool {tool_name} executed successfully")
                else:
                    tool_output = {"error": f"Unknown tool: {tool_name}"}
                    print(f"[GENERATOR NODE] Unknown tool: {tool_name}")
                
                # Add tool result as ToolMessage
                messages.append(
                    ToolMessage(
                        content=json.dumps(tool_output),
                        tool_call_id=tool_id,
                        name=tool_name
                    )
                )
            except Exception as e:
                error_msg = f"Error executing {tool_name}: {str(e)}"
                print(f"[GENERATOR NODE] {error_msg}")
                messages.append(
                    ToolMessage(
                        content=json.dumps({"error": error_msg}),
                        tool_call_id=tool_id,
                        name=tool_name
                    )
                )
        
        # Call LLM again with tool results
        print("[GENERATOR NODE] Calling LLM with tool results...")
        final_result = llm_generator_with_tools.invoke(messages)
        final_content = getattr(final_result, "content", "")
        messages.append(final_result)
        
    else:
        # No tool calls, just use the content
        print("[GENERATOR NODE] No tool calls requested by model")
        final_content = getattr(result, "content", "")
        messages.append(result)

    print(f"[GENERATOR NODE] Final output:\n{final_content}")

    return {
        "iteration": iteration,
        "last_generator_text": final_content,
        "history": [{"iteration": iteration, "generator_text": final_content}],
        "messages": messages
    }

# ---------------- ReAct reflector node ----------------
def node_reflector(state: PipelineState) -> dict:
    """Reflector: checks if generator output meets requirements, advises revision if needed."""
    sys_msg = """You are a quality control analyst reviewing market insights.

Evaluate the analysis based on:
1. Completeness - Does it address all aspects of the query?
2. Tool Usage - Were appropriate tools used?
3. Depth - Is the analysis thorough and insightful?
4. Coherence - Is the information well-organized?

If the analysis is satisfactory, respond with "CONVERGE - Analysis is complete."
If improvements are needed, specify what's missing and suggest which tools to use."""

    system_msg = SystemMessage(content=sys_msg)
    messages = [system_msg, HumanMessage(content=state.get("last_generator_text", ""))]

    result = llm_reflector.invoke(messages)
    critique_text = getattr(result, "content", getattr(result, "text", str(result)))
    
    print(f"[REFLECTOR NODE] Reflector output:\n{critique_text}")

    return {
        "last_reflector_text": critique_text,
        "history": [{"iteration": state.get("iteration", 0), "reflector_text": critique_text}],
        "messages": [AIMessage(content=critique_text)]
    }

# ---------------- Stop criteria ----------------
MAX_ITERS = 3
SIM_THRESHOLD = 0.86

def should_continue(state: PipelineState) -> Literal["reflector", "end"]:
    it = state.get("iteration", 0)
    if it >= MAX_ITERS:
        print("[SHOULD_CONTINUE] Max iterations reached")
        return END
    
    last_ref = (state.get("last_reflector_text") or "").lower()
    if "converge" in last_ref:
        print("[SHOULD_CONTINUE] Reflector signaled convergence")
        return END
    
    hist = state.get("history", []) or []
    gen_texts = [h.get("generator_text") for h in hist if h.get("generator_text")]
    if len(gen_texts) >= 2:
        similarity = difflib.SequenceMatcher(None, gen_texts[-2], gen_texts[-1]).ratio()
        if similarity >= SIM_THRESHOLD:
            print(f"[SHOULD_CONTINUE] High similarity detected: {similarity:.2f}")
            return END
    
    print("[SHOULD_CONTINUE] Continuing to reflector")
    return "reflector"

# ---------------- Pipeline builder ----------------
def build_pipeline():
    graph = StateGraph(PipelineState)

    # Add nodes
    graph.add_node("input_node", input_node)
    graph.add_node("intent_node", intent_node)
    graph.add_node("sqlgen_node", sqlgen_node)
    graph.add_node("validation_node", validation_node)
    graph.add_node("insert_node", insert_node)
    graph.add_node("generator", node_generator)
    graph.add_node("reflector", node_reflector)

    # Connect nodes
    graph.add_edge(START, "input_node")
    graph.add_edge("input_node", "intent_node")
    graph.add_edge("intent_node", "sqlgen_node")
    graph.add_edge("sqlgen_node", "validation_node")

    # Conditional loop-back after validation
    graph.add_conditional_edges(
        "validation_node",
        lambda state: "input_node" if state.get("loop_back") else "insert_node",
        {"input_node": "input_node", "insert_node": "insert_node"},
    )

    # After insertion, go to generator
    graph.add_edge("insert_node", "generator")

    # ReAct agent loop
    graph.add_conditional_edges("generator", should_continue, ["reflector", END])
    graph.add_edge("reflector", "generator")

    return graph.compile()

# ---------------- Pipeline runner ----------------
def run_data_fetching(user_query: str = ""):
    print("\n🚀 Running data fetching pipeline...")
    graph = build_pipeline()
    initial_state = PipelineState(
        user_query=user_query,
        filters=None,
        sql_query=None,
        result=None,
        error=None,
        inserted=False,
        loop_back=False,
        retry_count=0,
        messages=[HumanMessage(content=user_query)],
        history=[],
        iteration=0,
        last_generator_text="",
        last_reflector_text="",
        final_analysis=""
    )
    temp_query_holder = {"user_query": user_query, "retry_count": 0, "loop_back": False, "filters": None, "error": None}

    try:
        final_state = graph.invoke(initial_state, config={"configurable": {"temp_query_holder": temp_query_holder}})
    except Exception as e:
        return {"retry": False, "message": f"Pipeline error: {str(e)}"}

    if final_state.get("loop_back"):
        return {"retry": True, "message": final_state.get("error", "No data found."), "previous_filters": final_state.get("filters", {})}
    if final_state.get("error"):
        return {"retry": False, "message": final_state["error"]}
    if not final_state.get("result"):
        return {"retry": True, "message": "No data found. Please refine your filters."}
    
    return {
        "retry": False, 
        "data": final_state["result"],
        "analysis": final_state.get("last_generator_text", ""),
        "iterations": final_state.get("iteration", 0)
    }




# Nodes.py

from typing import Any, Dict, List, Optional
import os
import json
import re
import traceback

import pandas as pd
import matplotlib.pyplot as plt
from google import genai
from state_manager import StateManager
from dotenv import load_dotenv

env_path = '.env'
load_dotenv(dotenv_path=env_path)

class GeminiClient:
     
    def _init_(self, model: str = "gemini-2.5-flash"):
        self.model_name = model
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY not found in environment variables.")
        self.client = genai.Client(api_key=api_key)

    def generate(self, prompt: str, max_output_tokens: int = 512) -> str:
        response = self.client.models.generate_content(
        model=self.model_name,
        contents=prompt,
        config={"max_output_tokens": max_output_tokens}
    )
        return response.text or ""



# Instantiate Gemini client at module load (so errors are immediate)
_gemini_client: Optional[GeminiClient] = None
try:
    _gemini_client = GeminiClient()
except Exception as e:
    # Do not swallow — leave None; llm node will raise a clear error when called
    _gemini_client = None
    # It's fine to continue: other nodes (parsing/visualization) can be tested without LLM.
    # But we do not provide any fallback LLM in this file by your instruction (Option 1).

# ---------- Helpers ----------
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def _parse_swot_text(text: str) -> Dict[str, Any]:
    """Parse raw SWOT plain text into structured dict with metrics & bullets."""
    lines = text.splitlines()
    metrics: Dict[str, Dict[str, float]] = {}
    strengths: List[str] = []
    weaknesses: List[str] = []
    threats: List[str] = []
    section: Optional[str] = None

    metric_re = re.compile(r'([A-Za-z0-9_ \-]+):\s*([-+]?\d*\.?\d+)\s*vs\s*avg\s*([-+]?\d*\.?\d+)', re.IGNORECASE)
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        low = ln.lower()
        if low.startswith("strengths"):
            section = "strengths"; continue
        if low.startswith("weaknesses"):
            section = "weaknesses"; continue
        if low.startswith("threats"):
            section = "threats"; continue
        cleaned = re.sub(r'^[\-\u2022\]\s', '', ln).strip()
        m = metric_re.search(cleaned)
        if m:
            name = m.group(1).strip().replace(" ", "_")
            try:
                pivot = float(m.group(2))
                avg = float(m.group(3))
            except Exception:
                continue
            metrics[name] = {"pivot": pivot, "avg": avg}
        else:
            if section == "strengths":
                strengths.append(cleaned)
            elif section == "weaknesses":
                weaknesses.append(cleaned)
            elif section == "threats":
                threats.append(cleaned)
    return {"metrics": metrics, "strengths": strengths, "weaknesses": weaknesses, "threats": threats}


def _market_json_to_extracted(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize Market Analyzer JSON into DataFrames and meta dict."""
    extracted: Dict[str, Any] = {}
    prods = obj.get("product_summary_topN", []) or []
    brands = obj.get("brand_market_share_topN", []) or obj.get("brand_summary_topN", []) or []
    corr = obj.get("correlation_matrix", {}) or {}
    try:
        extracted["product_summary_df"] = pd.DataFrame(prods)
    except Exception:
        extracted["product_summary_df"] = pd.DataFrame()
    try:
        extracted["brand_market_share_df"] = pd.DataFrame(brands)
    except Exception:
        extracted["brand_market_share_df"] = pd.DataFrame()
    try:
        if isinstance(corr, dict) and corr:
            corr_df = pd.DataFrame(corr)
        else:
            corr_df = pd.DataFrame()
        extracted["correlation_matrix_df"] = corr_df
    except Exception:
        extracted["correlation_matrix_df"] = pd.DataFrame()
    extracted["meta"] = obj.get("global_summary", {})
    return extracted


def _reviews_json_to_extracted(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize Reviews JSON into reviews DataFrame and meta."""
    reviews = obj.get("reviews_summary", []) or []
    rows: List[Dict[str, Any]] = []
    for r in reviews:
        rows.append({
            "product_id": r.get("product_id"),
            "product_name": r.get("product_name"),
            "brand_name": r.get("brand_name"),
            "avg_rating": r.get("avg_rating"),
            "avg_returns_rate": r.get("avg_returns_rate"),
            "total_num_reviews": r.get("total_num_reviews"),
            "top_common_keywords": r.get("top_common_keywords"),
            "sample_reviews": (r.get("review") or [])[:3]
        })
    try:
        reviews_df = pd.DataFrame(rows)
    except Exception:
        reviews_df = pd.DataFrame()
    return {"reviews_df": reviews_df, "meta": {"meta_rows_fetched": obj.get("meta_rows_fetched", None)}}


def _extract_captions_from_text(text: str) -> List[str]:
    """Heuristic: extract up to two short lines from text as captions."""
    if not text:
        return ["", ""]
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    caps: List[str] = []
    for ln in lines:
        if len(caps) >= 2:
            break
        if len(ln) <= 200:
            caps.append(ln)
    while len(caps) < 2:
        caps.append("")
    return caps


def _save_fig(fig, basename: str) -> str:
    path = os.path.join(OUTPUT_DIR, basename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path

# ----------------------------
# Node: input_node
# ----------------------------
def input_node(state: StateManager) -> None:
    """
    Interactive input: asks which tool and reads payload from stdin.
    Saves into state:
      - 'tool_name' (str)
      - 'raw_payload' (str) OR 'analysis_text' for SWOT
    Clears downstream keys.
    """
    print("=== Input Node ===")
    print('Which tool is sending data? Options:')
    print('  "SWOT Analysis Tool", "Market Analysis Tool", "Reviews Analysis Tool"')
    tool = input("Tool name: ").strip()
    print("Paste the payload (JSON or SWOT text). Finish input with EOF (Ctrl+D or Ctrl+Z then Enter).")
    lines: List[str] = []
    try:
        while True:
            ln = input()
            lines.append(ln)
    except EOFError:
        pass
    payload = "\n".join(lines).strip()
    state.set("tool_name", tool)
    if "swot" in (tool or "").lower():
        state.set("analysis_text", payload)
        state.set("raw_payload", None)
    else:
        state.set("raw_payload", payload)
        state.set("analysis_text", None)
    # clear downstream
    state.set("extracted_df", None)
    state.set("llm_prompt", None)
    state.set("llm_output", None)
    state.set("charts", [])
    state.set("summary", None)
    print(f"Saved tool_name='{tool}' with payload length {len(payload)} characters.")


def insight_extractor_node(state: StateManager) -> None:
    """
    Parse raw input into structured extracted_df in state.
    Handles SWOT (text), Market JSON, Reviews JSON.
    """
    tool = (state.get("tool_name") or "").lower()
    raw_payload = state.get("raw_payload")
    analysis_text = state.get("analysis_text")

    if "swot" in tool:
        if not analysis_text:
            raise ValueError("insight_extractor_node: missing analysis_text for SWOT tool")
        parsed = _parse_swot_text(analysis_text)
        state.set("extracted_df", parsed)
        print("insight_extractor_node: parsed SWOT into extracted_df.")
        return

    if raw_payload is None:
        raise ValueError("insight_extractor_node: missing raw_payload for non-SWOT tool")

    try:
        obj = json.loads(raw_payload)
    except Exception as exc:
        raise ValueError(f"insight_extractor_node: invalid JSON payload: {exc}")

    if "global_summary" in obj or "product_summary_topN" in obj or "brand_market_share_topN" in obj:
        extracted = _market_json_to_extracted(obj)
        state.set("extracted_df", extracted)
        print("insight_extractor_node: parsed Market JSON into extracted_df.")
        return

    if "reviews_summary" in obj or "reviews" in obj:
        extracted = _reviews_json_to_extracted(obj)
        state.set("extracted_df", extracted)
        print("insight_extractor_node: parsed Reviews JSON into extracted_df.")
        return

    # unknown JSON structure -> store raw
    state.set("extracted_df", {"raw_json": obj})
    print("insight_extractor_node: unknown JSON structure, saved raw_json in extracted_df.")


# ----------------------------
# Node: llm_insight_analyzer_node
# ----------------------------
def llm_insight_analyzer_node(state: StateManager) -> None:
    """
    Prepare tool-specific prompt from extracted_df and call Gemini to generate:
     - two chart captions
     - 3 actionable insights (or other requested outputs)
    Saves:
      - state['llm_prompt']
      - state['llm_output']
      - state['summary'] (short human summary)
    """
    extracted = state.get("extracted_df")
    tool = (state.get("tool_name") or "").lower()

    if not extracted:
        raise ValueError("llm_insight_analyzer_node: missing extracted_df in state")

    # Build prompt
    if "metrics" in extracted:  # SWOT
        metrics = extracted.get("metrics", {})
        # pick top 6 by diff
        items = sorted(metrics.items(), key=lambda kv: abs(kv[1]["pivot"] - kv[1]["avg"]), reverse=True)[:6]
        chosen = {k: v for k, v in items}
        bullets = {
            "strengths": extracted.get("strengths", [])[:6],
            "weaknesses": extracted.get("weaknesses", [])[:6],
            "threats": extracted.get("threats", [])[:6]
        }
        prompt = (
            "You are an expert market analyst for makeup & skincare.\n\n"
            "Input (SWOT):\n"
            f"Selected metrics (pivot vs avg): {json.dumps(chosen, indent=2)}\n"
            f"Bullets (sample): {json.dumps(bullets, indent=2)}\n\n"
            "Tasks:\n"
            "1) Provide two short captions for charts: (A) Pivot vs competitor avg; (B) SWOT counts chart.\n"
            "2) Provide three actionable insights (1-3 sentences each) focused on product/marketing actions.\n"
            "Return plain text."
        )

    elif "product_summary_df" in extracted or "brand_market_share_df" in extracted:  # Market
        prod_df: pd.DataFrame = extracted.get("product_summary_df", pd.DataFrame())
        brand_df: pd.DataFrame = extracted.get("brand_market_share_df", pd.DataFrame())
        meta = extracted.get("meta", {})
        pd_sample = prod_df.head(5).to_dict(orient="records") if not prod_df.empty else []
        bd_sample = brand_df.head(5).to_dict(orient="records") if not brand_df.empty else []
        prompt = (
            "You are a market analyst for cosmetics (makeup & skincare).\n\n"
            "Input (market summary):\n"
            f"Meta: {json.dumps(meta)}\n"
            f"Top products sample: {json.dumps(pd_sample, indent=2)}\n"
            f"Top brands sample: {json.dumps(bd_sample, indent=2)}\n\n"
            "Tasks:\n"
            "1) Provide two short captions: (A) Brand market share bar chart; (B) Product revenue vs units scatter.\n"
            "2) Provide five concise takeaways and three recommended next steps for a product pivot.\n"
            "Return plain text."
        )

    elif "reviews_df" in extracted:  # Reviews
        reviews_df: pd.DataFrame = extracted.get("reviews_df", pd.DataFrame())
        sample = reviews_df.head(6).to_dict(orient="records") if not reviews_df.empty else []
        prompt = (
            "You are a product reviews analyst.\n\n"
            "Input (reviews sample):\n"
            f"{json.dumps(sample, indent=2)}\n\n"
            "Tasks:\n"
            "1) Provide two short captions: (A) Average rating per product; (B) Rating vs returns rate.\n"
            "2) List top 5 sentiment-driven issues and 3 product improvement suggestions.\n"
            "Return plain text."
        )
    else:
        prompt = "Please summarize the following data and suggest two chart captions and three insights:\n\n" + json.dumps(extracted, default=str)[:4000]

    # Save prompt
    state.set("llm_prompt", prompt)

    # Ensure gemini client is available
    if _gemini_client is None:
        raise RuntimeError("Gemini client not initialized. Ensure google-genai is installed and GEMINI_API_KEY is set.")

    # Call Gemini
    try:
        print("llm_insight_analyzer_node: calling Gemini model...")
        out = _gemini_client.generate(prompt, max_output_tokens=800)
    except Exception as exc:
        traceback.print_exc()
        out = f"[Gemini call failed: {exc}]"

    state.set("llm_output", out)
    state.set("summary", out if len(out) <= 1200 else out[:1200] + " ...[truncated]")
    print("llm_insight_analyzer_node: saved llm_output and summary.")


# ----------------------------
# Node: visualization_output_node
# ----------------------------
def visualization_output_node(state: StateManager) -> None:
    """
    Generate two charts according to the extracted_df and store them in state['charts'].
    Attaches captions extracted heuristically from LLM output if available.
    """
    extracted = state.get("extracted_df")
    llm_out = state.get("llm_output") or ""
    captions = _extract_captions_from_text(llm_out)
    charts: List[Dict[str, str]] = []

    def register(fig, filename: str, caption: str):
        path = _save_fig(fig, filename)
        charts.append({"path": path, "caption": caption})

    try:
        # SWOT visuals
        if isinstance(extracted, dict) and "metrics" in extracted:
            metrics = extracted.get("metrics", {})
            if metrics:
                items = sorted(metrics.items(), key=lambda kv: abs(kv[1]["pivot"] - kv[1]["avg"]), reverse=True)[:6]
                names = [k for k, v in items]
                pivots = [v["pivot"] for k, v in items]
                avgs = [v["avg"] for k, v in items]
                fig, ax = plt.subplots(figsize=(8,4))
                x = range(len(names))
                width = 0.35
                ax.bar([i - width/2 for i in x], pivots, width=width, label="pivot")
                ax.bar([i + width/2 for i in x], avgs, width=width, label="avg_competitor")
                ax.set_xticks(list(x)); ax.set_xticklabels(names, rotation=35, ha="right")
                ax.set_ylabel("Value"); ax.set_title("Pivot vs Competitor Average (selected metrics)"); ax.legend()
                register(fig, "swot_pivot_vs_avg.png", captions[0] if captions else "Pivot vs Avg")

            counts = {
                "strengths": len(extracted.get("strengths", [])),
                "weaknesses": len(extracted.get("weaknesses", [])),
                "threats": len(extracted.get("threats", []))
            }
            fig, ax = plt.subplots(figsize=(6,3))
            ax.bar(list(counts.keys()), list(counts.values()))
            ax.set_title("SWOT bullet counts")
            register(fig, "swot_counts.png", captions[1] if len(captions) > 1 else "SWOT counts")

        # Market visuals
        elif isinstance(extracted, dict) and ("product_summary_df" in extracted or "brand_market_share_df" in extracted):
            brand_df: pd.DataFrame = extracted.get("brand_market_share_df", pd.DataFrame())
            prod_df: pd.DataFrame = extracted.get("product_summary_df", pd.DataFrame())

            if isinstance(brand_df, pd.DataFrame) and not brand_df.empty:
                if "share" in brand_df.columns:
                    x = brand_df["brand_name"].astype(str)
                    y = brand_df["share"].astype(float)
                else:
                    x = brand_df["brand_name"].astype(str)
                    y = brand_df.get("total_revenue_usd", pd.Series([0]*len(brand_df))).astype(float)
                fig, ax = plt.subplots(figsize=(8,4))
                ax.bar(x, y)
                ax.set_xticklabels(x, rotation=35, ha="right")
                ax.set_title("Brand Market Share (topN)")
                register(fig, "market_brand_share.png", captions[0] if captions else "Brand Market Share")

            if isinstance(prod_df, pd.DataFrame) and not prod_df.empty:
                if "total_units_sold" in prod_df.columns and "total_revenue_usd" in prod_df.columns:
                    fig, ax = plt.subplots(figsize=(7,5))
                    ax.scatter(prod_df["total_units_sold"].astype(float), prod_df["total_revenue_usd"].astype(float))
                    for i, row in prod_df.head(10).iterrows():
                        name = str(row.get("product_name", ""))[:20]
                        ax.annotate(name, (row["total_units_sold"], row["total_revenue_usd"]), fontsize=7)
                    ax.set_xlabel("Units sold"); ax.set_ylabel("Total revenue (USD)"); ax.set_title("Product revenue vs units")
                    register(fig, "market_product_revenue_vs_units.png", captions[1] if len(captions) > 1 else "Revenue vs Units")
                else:
                    # fallback bar by revenue
                    if "total_revenue_usd" in prod_df.columns:
                        prod_sorted = prod_df.sort_values("total_revenue_usd", ascending=False).head(15)
                        fig, ax = plt.subplots(figsize=(9,4))
                        ax.bar(prod_sorted["product_name"].astype(str), prod_sorted["total_revenue_usd"].astype(float))
                        ax.set_xticklabels(prod_sorted["product_name"].astype(str), rotation=60, ha="right")
                        ax.set_title("Top products by revenue")
                        register(fig, "market_top_products_revenue.png", captions[1] if len(captions) > 1 else "Top product revenue")

        # Reviews visuals
        elif isinstance(extracted, dict) and "reviews_df" in extracted:
            df: pd.DataFrame = extracted.get("reviews_df", pd.DataFrame())
            if isinstance(df, pd.DataFrame) and not df.empty:
                df_sorted = df.sort_values("avg_rating", ascending=False).head(15)
                fig, ax = plt.subplots(figsize=(9,4))
                ax.bar(df_sorted["product_name"].astype(str), df_sorted["avg_rating"].astype(float))
                ax.set_xticklabels(df_sorted["product_name"].astype(str), rotation=60, ha="right")
                ax.set_ylabel("Avg rating"); ax.set_title("Average rating per product (top)")
                register(fig, "reviews_avg_rating.png", captions[0] if captions else "Avg rating per product")

                if "avg_rating" in df.columns and "avg_returns_rate" in df.columns:
                    fig, ax = plt.subplots(figsize=(7,5))
                    ax.scatter(df["avg_rating"].astype(float), df["avg_returns_rate"].astype(float))
                    for i, r in df.iterrows():
                        if pd.notna(r.get("avg_rating")) and pd.notna(r.get("avg_returns_rate")):
                            ax.annotate(str(r.get("product_id") or r.get("product_name", ""))[:8], (r["avg_rating"], r["avg_returns_rate"]), fontsize=7)
                    ax.set_xlabel("Avg rating"); ax.set_ylabel("Avg returns rate"); ax.set_title("Rating vs Returns Rate")
                    register(fig, "reviews_rating_vs_returns.png", captions[1] if len(captions) > 1 else "Rating vs Returns")
            else:
                print("visualization_output_node: reviews_df empty or missing columns; no charts created.")

        else:
            print("visualization_output_node: extracted_df format not recognized; no charts generated.")

    except Exception:
        traceback.print_exc()
        print("visualization_output_node: error while creating charts.")

    state.set("charts", charts)
    print(f"visualization_output_node: saved {len(charts)} chart(s) to '{OUTPUT_DIR}'.")


# End of Nodes.py


# ////////////////////////////////////////////////////////////////////////////////////////////////////
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict, Literal, Any, Dict, Optional
import operator
import difflib
import json
import os
from dotenv import load_dotenv
from supabase import create_client

# LangChain Groq import
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Custom tools
from app.analysis.tool1 import swot_analysis as swot_tool
from app.analysis.market_tool import deep_market_analysis
from app.analysis.reviews_summary_tool import fetch_reviews_summary

# Agents
from app.data_fetching.agents.intent_agent import detect_intent
from app.data_fetching.agents.sql_agent import generate_sql
from app.data_fetching.agents.validator_agent import validate_sql_and_execute
from langchain_core.tools import tool

# --- Config / load env ---
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Please set SUPABASE_URL and SUPABASE_KEY in your environment or .env file")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# -----------------------------Model loading--------------------
llm_generator = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.7)
llm_reflector = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.7)

TOOLS = [swot_tool, deep_market_analysis, fetch_reviews_summary]
llm_generator_with_tool=llm_generator.bind_tools(TOOLS)
# ---------------- PIPELINE STATE ----------------
class PipelineState(TypedDict):
    user_query: Optional[str]
    filters: Optional[Dict[str, Any]]
    sql_query: Optional[str]
    result: Optional[Any]
    error: Optional[str]
    inserted: Optional[bool]
    loop_back: Optional[bool]
    retry_count: Optional[int]
    messages: Annotated[list, add_messages]
    history: Annotated[list, operator.add]
    iteration: int
    last_generator_text: str
    last_reflector_text: str
    final_analysis: str

# ---------------- NODES ----------------
# ---------------- NODES ----------------
def input_node(state: PipelineState, **kwargs):
    """Prompt user for input or refined query in case of errors or loopbacks."""
    temp_query_holder = kwargs.setdefault("temp_query_holder", {})
    print("\n[INPUT NODE] Awaiting user input...")
    MAX_RETRIES = 5

    current_retry = temp_query_holder.get("retry_count", 0)
    if current_retry >= MAX_RETRIES:
        print("[INPUT NODE] Maximum retries reached.")
        return {**state, "error": "Maximum retries reached.", "loop_back": False}

    if temp_query_holder.get("loop_back") or temp_query_holder.get("error"):
        temp_query_holder["retry_count"] = current_retry + 1
        temp_query_holder["user_query"] = input("Please enter a refined query: ")
        print(f"[INPUT NODE] Refined query received: {temp_query_holder['user_query']}")
    elif "user_query" not in temp_query_holder or not temp_query_holder["user_query"]:
        temp_query_holder["user_query"] = input("Please enter your query: ")
        print(f"[INPUT NODE] Initial query received: {temp_query_holder['user_query']}")

    temp_query_holder["error"] = None
    temp_query_holder["loop_back"] = False

    return {**state, "user_query": temp_query_holder["user_query"], "error": None, "loop_back": False,
            "retry_count": temp_query_holder.get("retry_count", 0)}

def intent_node(state: PipelineState, **kwargs):
    """Detect user intent from query and extract filters for SQL."""
    temp_query_holder = kwargs.setdefault("temp_query_holder", {})
    user_query = state.get("user_query")
    print("\n[INTENT NODE] Step 1: Detecting intent from query...")
    print(f"[INTENT NODE] User query: {user_query}")
    try:
        intent = detect_intent(user_query)
        temp_query_holder["filters"] = intent.get("filters", {})
        print(f"[INTENT NODE] Detected filters: {temp_query_holder['filters']}")
        return {**state, "filters": intent.get("filters", {}), "loop_back": False, "error": None}
    except Exception as e:
        print(f"[INTENT NODE] Error in detecting intent: {str(e)}")
        temp_query_holder["error"] = str(e)
        temp_query_holder["loop_back"] = True
        return {**state, "error": str(e), "loop_back": True}

def sqlgen_node(state: PipelineState, **kwargs):
    """Generate SQL query from detected filters."""
    temp_query_holder = kwargs.setdefault("temp_query_holder", {})
    if state.get("error"):
        return state

    print("\n[SQL NODE] Step 2: Generating SQL query...")
    try:
        filters = state.get("filters", {})
        sql_query = generate_sql(filters)
        print(f"[SQL NODE] Generated SQL: {sql_query}")
        return {**state, "sql_query": sql_query}
    except Exception as e:
        print(f"[SQL NODE] Error generating SQL: {str(e)}")
        temp_query_holder["error"] = str(e)
        temp_query_holder["loop_back"] = True
        return {**state, "error": str(e), "loop_back": True}

def validation_node(state: PipelineState, **kwargs):
    """Validate and execute SQL, handle user-correctable errors."""
    temp_query_holder = kwargs.setdefault("temp_query_holder", {})
    if state.get("error"):
        return state

    print("\n[VALIDATION NODE] Step 3: Validating and executing SQL...")
    validation_output = validate_sql_and_execute(state["sql_query"])

    if "error" in validation_output:
        msg = validation_output["error"]
        user_correctable = validation_output.get("user_correctable", False)
        print(f"[VALIDATION NODE] SQL Error: {msg}, user_correctable={user_correctable}")
        if user_correctable:
            temp_query_holder["loop_back"] = True
            temp_query_holder["retry_count"] = temp_query_holder.get("retry_count", 0) + 1
        temp_query_holder["error"] = msg
        return {**state, "error": msg, "loop_back": user_correctable}

    print(f"[VALIDATION NODE] SQL executed successfully, rows returned: {len(validation_output.get('data', []))}")
    return {**state, "result": validation_output.get("data", []), "loop_back": False, "error": None}

def insert_node(state: PipelineState):
    """Insert fetched data into Supabase table."""
    if state.get("error") or not state.get("result"):
        return state
    table_name = "product_flat_table"
    try:
        print(f"[INSERT NODE] Clearing old data in table {table_name}...")
        del_response = supabase.table(table_name).delete().neq("product_id", "").execute()
        if hasattr(del_response, "error") and del_response.error:
            print(f"[INSERT NODE] Error clearing old data: {del_response.error}")
            return {**state, "error": f"Failed to clear old data: {del_response.error}"}

        print(f"[INSERT NODE] Inserting {len(state['result'])} rows into table {table_name}...")
        response = supabase.table(table_name).insert(state["result"]).execute()
        if hasattr(response, "error") and response.error:
            print(f"[INSERT NODE] Error inserting data: {response.error}")
            return {**state, "error": f"Insert failed: {response.error}"}
        print("[INSERT NODE] Data insertion successful.")
        return {**state, "inserted": True}
    except Exception as e:
        print(f"[INSERT NODE] Exception during insert: {str(e)}")
        return {**state, "error": f"Insert exception: {str(e)}"}

# ---------------- ReAct generator node ----------------
def node_generator(state: PipelineState) -> dict:
    """ReAct-style generator: call tools iteratively and produce 4 actionable insights."""
    user_query = state.get("user_query")
    iteration = state.get("iteration", 0) + 1
    messages = list(state.get("messages", []))

    print(f"\n[GENERATOR NODE] Iteration {iteration} - Reasoning with LLM...")

    # --- system message for ReAct agent ---
    sys_msg = SystemMessage(content="""You are a strategic market analyst with access to tools and must reason carefully before giving insights.""")
    messages.append(sys_msg)
    messages.append(HumanMessage(content=user_query))

    # --- Bind tools to LLM ---
    TOOLS = [swot_tool, deep_market_analysis, fetch_reviews_summary]
    llm_with_tools = llm_generator.bind_tools(TOOLS)

    # --- Invoke LLM ---
    result = llm_with_tools.invoke(messages)
    content = getattr(result, "content", getattr(result, "text", str(result)))
    messages.append(AIMessage(content=content))

    print(f"[GENERATOR NODE] LLM output:\n{content}")

    return {
        "iteration": iteration,
        "last_generator_text": content,
        "history": [{"iteration": iteration, "generator_text": content}],
        "messages": messages
    }

# ---------------- ReAct reflector node ----------------
def node_reflector(state: PipelineState) -> dict:
    """Reflector: checks if generator output meets requirements, advises revision if needed."""
    sys_msg= """You are a quality control analyst reviewing market insights..."""
    system_msg = SystemMessage(content=sys_msg)
    messages = list(state.get("messages", []))
    messages.append(system_msg)
    messages.append(HumanMessage(content=state.get("last_generator_text", "")))

    result = llm_reflector.invoke(messages)
    critique_text = getattr(result, "content", getattr(result, "text", str(result)))
    messages.append(AIMessage(content=critique_text))
    print(f"[REFLECTOR NODE] Reflector output:\n{critique_text}")

    return {
        "last_reflector_text": critique_text,
        "history": [{"iteration": state.get("iteration", 0), "reflector_text": critique_text}],
        "messages": messages
    }

# ---------------- Stop criteria ----------------
MAX_ITERS = 3
SIM_THRESHOLD = 0.86

def should_continue(state: PipelineState) -> Literal["reflector", "end"]:
    it = state.get("iteration", 0)
    if it >= MAX_ITERS:
        return END
    last_ref = (state.get("last_reflector_text") or "").lower()
    if "converge" in last_ref:
        return END
    hist = state.get("history", []) or []
    gen_texts = [h.get("generator_text") for h in hist if h.get("generator_text")]
    if len(gen_texts) >= 2 and difflib.SequenceMatcher(None, gen_texts[-2], gen_texts[-1]).ratio() >= SIM_THRESHOLD:
        return END
    return "reflector"

# ---------------- Pipeline builder ----------------
def build_pipeline():
    graph = StateGraph(PipelineState)

    # ----------------- Add nodes manually -----------------
    graph.add_node("input_node", input_node)
    graph.add_node("intent_node", intent_node)
    graph.add_node("sqlgen_node", sqlgen_node)
    graph.add_node("validation_node", validation_node)
    graph.add_node("insert_node", insert_node)
    graph.add_node("generator", node_generator)
    graph.add_node("reflector", node_reflector)

    # ----------------- Connect nodes -----------------
    graph.add_edge(START, "input_node")
    graph.add_edge("input_node", "intent_node")
    graph.add_edge("intent_node", "sqlgen_node")
    graph.add_edge("sqlgen_node", "validation_node")

    # Conditional loop-back after validation
    graph.add_conditional_edges(
        "validation_node",
        lambda state: "input_node" if state.get("loop_back") else "insert_node",
        {"input_node": "input_node", "insert_node": "insert_node"},
    )

    # After insertion, go to generator
    graph.add_edge("insert_node", "generator")

    # ReAct agent loop
    graph.add_conditional_edges("generator", should_continue, ["reflector", END])
    graph.add_edge("reflector", "generator")

    # End edges
    graph.add_edge("insert_node", END)

    # Set entry point
    graph.set_entry_point("input_node")

    return graph.compile()


# ---------------- Pipeline runner ----------------
def run_data_fetching(user_query: str = ""):
    print("\n🚀 Running data fetching pipeline...")
    graph = build_pipeline()
    initial_state = PipelineState(
        user_query=user_query,
        filters=None,
        sql_query=None,
        result=None,
        error=None,
        inserted=False,
        loop_back=False,
        retry_count=0,
        messages=[HumanMessage(content=user_query)],
        history=[],
        iteration=0,
        last_generator_text="",
        last_reflector_text="",
        final_analysis=""
    )
    temp_query_holder = {"user_query": user_query, "retry_count": 0, "loop_back": False, "filters": None, "error": None}

    try:
        final_state = graph.invoke(initial_state, config={"configurable": {"temp_query_holder": temp_query_holder}})
    except Exception as e:
        return {"retry": False, "message": f"Pipeline error: {str(e)}"}

    if final_state.get("loop_back"):
        return {"retry": True, "message": final_state.get("error", "No data found."), "previous_filters": final_state.get("filters", {})}
    if final_state.get("error"):
        return {"retry": False, "message": final_state["error"]}
    if not final_state.get("result"):
        return {"retry": True, "message": "No data found. Please refine your filters."}
    return {"retry": False, "data": final_state["result"]}
# /////////////////////////////////////////////////////////////////////////////////////////////////////////
# ---------------- ReAct generator node ----------------
def node_generator(state: PipelineState) -> dict:
    """
    ReAct-style generator node.
    The LLM will:
    - Reason about what data is needed
    - Call tools iteratively
    - Observe tool outputs
    - Generate exactly 4 actionable insights referencing real numbers
    """
    user_query = state.get("user_query")
    iteration = state.get("iteration", 0) + 1
    messages = list(state.get("messages", []))

    # --- system message for ReAct agent ---
    sys_msg = SystemMessage(content="""
You are a strategic market analyst with access to the following tools:

TOOLS:
1. deep_market_analysis(): returns market aggregates, brand rankings, market shares, competition metrics
2. swot_tool(): returns competitive SWOT analysis with numeric comparisons
3. fetch_reviews_summary(): returns customer sentiment, keywords, return rates, sample reviews

GUIDELINES (ReAct loop):
- Before giving final insights, ALWAYS reason about what data you need.
- Call the appropriate tool using: call tool_name()
- Observe tool outputs before making decisions.
- You may call multiple tools iteratively if new data is needed.
- NEVER invent numbers; only use actual tool outputs.
- Generate exactly 4 actionable insights referencing numbers from tools.

OUTPUT FORMAT:
```json
{
  "insights": [
    {"insight": "...", "action": "..."},
    {"insight": "...", "action": "..."},
    {"insight": "...", "action": "..."},
    {"insight": "...", "action": "..."}
  ],
  "tools_used": ["tool1", "tool2"],
  "provenance": {
    "tool1": {"key_metrics": "values used"},
    "tool2": {"key_metrics": "values used"}
  }
}
""")
messages.append(sys_msg)
messages.append(HumanMessage(content=user_query))

# --- Bind tools to LLM ---
TOOLS = [swot_tool, deep_market_analysis, fetch_reviews_summary]
llm_with_tools = llm_generator.bind_tools(TOOLS)

# --- Invoke LLM, allowing multiple tool calls ---
result = llm_with_tools.invoke(messages)
content = getattr(result, "content", getattr(result, "text", str(result)))
messages.append(AIMessage(content=content))

print(f"\n[GENERATOR NODE] Iteration {iteration} - Output:\n{content}")

return {
    "iteration": iteration,
    "last_generator_text": content,
    "history": [{"iteration": iteration, "generator_text": content}],
    "messages": messages
}
# /////////////////////////////////////////////////////

# def node_generator(state: PipelineState) -> dict:
#     """ReAct-style generator that properly handles tool calls from Groq."""
#     user_query = state.get("user_query")
#     filters = state.get("filters", {})
#     iteration = state.get("iteration", 0) + 1
#     messages = list(state.get("messages", []))

#     print(f"\n[GENERATOR NODE] Iteration {iteration} - Reasoning with LLM...")

#     # Add system message with clear instructions about tools
#     if iteration == 1:
#         system_prompt = f"""You are a market analysis assistant ReAct Agent with access to specialized tools.

# User Query: {user_query}
# Detected Filters: {json.dumps(filters, indent=2)}

# You have access to these tools:
# 1. swot_analysis - Performs SWOT analysis on products
# 2. deep_market_analysis - Provides deep market insights
# 3. fetch_reviews_summary - Fetches and summarizes product reviews

# ReAct agent:
# Follow the ReAct pattern:
# - First, reason about what data you need.
# - Then, call the relevant tool(s).
# - Observe tool outputs before deciding next steps.
# - Never make up data — always use real numbers from tool outputs.
# - When confident, give exactly 4 concise, actionable insights in JSON.
# - Its better to make use of as much numerical data as possible

# IMPORTANT: 
# - Use the tools to gather comprehensive information
# - Call multiple tools if needed to provide complete analysis
# - After receiving tool results, synthesize them into a coherent response
# - Be thorough and use all relevant tools for the query
# - Make sure you use numerical vaues to support but dont hallucinate ,use the numerical values you get from tools 


# Analyze the query and decide which tools to use."""
        
#         messages = [SystemMessage(content=system_prompt)] + messages

#     # Call LLM with tools bound
#     result = llm_generator_with_tools.invoke(messages)
    
#     # Check if the model wants to use tools
#     tool_calls = getattr(result, 'tool_calls', [])
    
#     if tool_calls:
#         print(f"[GENERATOR NODE] Model requested {len(tool_calls)} tool calls")
        
#         # Add the AI message with tool calls to history
#         messages.append(result)
        
#         # Execute each tool call
#         for tool_call in tool_calls:
#             tool_name = tool_call.get('name')
#             tool_args = tool_call.get('args', {})
#             tool_id = tool_call.get('id', 'unknown')
            
#             print(f"[GENERATOR NODE] Executing tool: {tool_name} with args: {tool_args}")
            
#             try:
#                 # Execute the tool with proper arguments
#                 if tool_name in TOOL_MAP:
#                     # All tools now accept a query string
#                     tool_fn = TOOL_MAP[tool_name]
#                     # Call the actual function, not the tool wrapper
#                     tool_output = tool_fn.func(user_query) if hasattr(tool_fn, 'func') else tool_fn(user_query)
#                     print(f"[GENERATOR NODE] Tool {tool_name} executed successfully")
#                 else:
#                     tool_output = {"error": f"Unknown tool: {tool_name}"}
#                     print(f"[GENERATOR NODE] Unknown tool: {tool_name}")
                
#                 # Add tool result as ToolMessage
#                 messages.append(
#                     ToolMessage(
#                         content=json.dumps(tool_output),
#                         tool_call_id=tool_id,
#                         name=tool_name
#                     )
#                 )
#             except Exception as e:
#                 error_msg = f"Error executing {tool_name}: {str(e)}"
#                 print(f"[GENERATOR NODE] {error_msg}")
#                 messages.append(
#                     ToolMessage(
#                         content=json.dumps({"error": error_msg}),
#                         tool_call_id=tool_id,
#                         name=tool_name
#                     )
#                 )
        
#         # Call LLM again with tool results
#         print("[GENERATOR NODE] Calling LLM with tool results...")
#         final_result = llm_generator_with_tools.invoke(messages)
#         final_content = getattr(final_result, "content", "")
#         messages.append(final_result)
        
#     else:
#         # No tool calls, just use the content
#         print("[GENERATOR NODE] No tool calls requested by model")
#         final_content = getattr(result, "content", "")
#         messages.append(result)

#     print(f"[GENERATOR NODE] Final output:\n{final_content}")

#     return {
#         "iteration": iteration,
#         "last_generator_text": final_content,
#         "history": [{"iteration": iteration, "generator_text": final_content}],
#         "messages": messages
#     }
