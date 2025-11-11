"""
langgraph_swot_graph.py

LangGraph (Graph API) pipeline:
  START -> node_fetch_top5 (A)
        -> node_normalize (B)
        -> node_generator (C)
           conditional -> node_reflector (E)  --loop--> node_generator (C)
                           (stop criteria node D implemented as conditional)
        -> END (finalization performed in generator when loop ends)

Requirements:
  - langgraph==1.0.1 (Graph API)
  - langchain (for init_chat_model, @tool)
  - pandas
  - user_functions.py containing:
      fetch_top5_brands(product: str, country: str) -> List[Dict[str,Any]]
      normalize_and_validate(list_of_dicts: List[Dict[str,Any]]) -> pd.DataFrame
      df_to_json(df: pd.DataFrame) -> List[Dict[str,Any]]
      swot_analysis(df: pd.DataFrame, pivot_brand: str) -> Dict[str,Any]
"""

from typing import Annotated, TypedDict, Literal, Any, List, Dict
import operator
import difflib
import json
import pandas as pd

# LangGraph imports (Graph API)
from langgraph.graph import StateGraph, START, END

# Import Google's Generative AI library
import google.generativeai as genai
from langchain.tools import tool

# Import user-provided domain functions (put these in user_functions.py)
try:
    from user_functions import (
        fetch_top5_brands_by_revenue as fetch_top5_brands,
        normalize_and_validate,
        rows_to_json_safe_all_columns as df_to_json,
        swot_analysis,
    )
except Exception as e:
    raise ImportError(
        "Couldn't import required user functions. Provide them in user_functions.py "
        "or update the import path. Required functions:\n"
        "- fetch_top5_brands(product, country) -> List[Dict]\n"
        "- normalize_and_validate(list_of_dicts) -> pd.DataFrame\n"
        "- df_to_json(df) -> List[Dict]\n"
        "- swot_analysis(df, pivot_brand) -> Dict\n    "
    ) from e

# -------------------------
# Configuration: 
# -------------------------

from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# --- API Key ---
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")  # Using standard Google API key name

# --- Validate ---
if not GEMINI_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY. Please check your .env file.")

# Initialize Gemini model directly
genai.configure(api_key=GEMINI_API_KEY)
generator_model = genai.GenerativeModel('gemini-2.0-flash')
reflector_model = genai.GenerativeModel('gemini-2.0-flash')  # Using same model for both

# -------------------------
# Tools: wrap the user's SWOT function as a LangChain tool
# (The model can call this tool by name; the tool will reconstruct a DataFrame
# from the provided JSON and call the user's swot_analysis.)
# -------------------------
@tool
def swot_tool(df_json: List[Dict[str, Any]], pivot_brand: str) -> Dict[str, Any]:
    """
    Tool signature exposed to the LLM:
      swot_tool(df_json, pivot_brand) -> returns Python dict (the structured swot)

    The LLM will be able to call this tool by name.
    """
    # convert json -> DataFrame
    df = pd.DataFrame(df_json)
    return swot_analysis(df, pivot_brand=df.iloc[0]["brand_name"])


# Create tools list (for use in generator node)
tools = [swot_tool]
tools_by_name = {t.name: t for t in tools}
# -------------------------
# State definition
# -------------------------
class AgentState(TypedDict):
    """
    Fields:
      - product, country : initial placeholders (strings)
      - raw_top5 : Annotated list, replaced not appended
      - last_df_json : latest df as JSON list (replace)
      - history : Annotated append list for generator/reflector iterations
      - iteration: int (current iteration)
      - last_generator_text: str (previous generator textual output)
      - last_reflector_text: str (previous reflector textual output)
      - final_analysis: str (final summary stored when done)
    """
    product: str
    country: str
    raw_top5: List[Dict[str, Any]]
    last_df_json: List[Dict[str, Any]]
    history: Annotated[list, operator.add]  # append semantics
    iteration: int
    last_generator_text: str
    last_reflector_text: str
    final_analysis: str


# -------------------------
# Node implementations (functions that accept and return partial state dicts)
# Each node returns a dict representing updates to the State.
# -------------------------
def node_fetch_top5(state: AgentState) -> dict:
    # Node A: call user fetch_top5_brands using placeholders in state
    product = state["product"]
    country = state["country"]
    raw_list = fetch_top5_brands(product, country)
    # Keep only top5 from user function (it should already be top5)
    return {"raw_top5": raw_list}


def node_normalize(state: AgentState) -> dict:
    # Node B: normalize & validate -> produce DataFrame then JSON snapshot
    raw = state.get("raw_top5", [])
    df = normalize_and_validate(raw)  # user function returns pandas DataFrame
    df_json = df_to_json(df)  # user conversion function
    return {"last_df_json": df_json}


def node_generator(state: AgentState) -> dict:
    """
    Node C: generator (Gemini) that uses bound tools (swot_tool).
    We:
      - increment iteration
      - build a compact prompt using last_df_json
      - invoke generator_model_with_tools.invoke([...]) per quickstart pattern
      - store generator textual output in last_generator_text and append to history
    """
    # increment iteration
    iteration = state.get("iteration", 0) + 1

    # compact preview of dataframe (top rows)
    df_preview = json.dumps(state.get("last_df_json", [])[:5], default=str)

    system_msg = (
        "You are a strategic insights generator. You have access to a tool `swot_tool(df_json, pivot_brand)` "
        "that returns a structured SWOT for a given pivot brand. Use the tool when needed to obtain "
        "structured SWOT before producing insights."
    )

    # Build the conversation/messages for this generator model call.
    # LangGraph quickstart shows calling model_with_tools.invoke([...messages...])
    from langchain.messages import SystemMessage, HumanMessage

    messages = [
        SystemMessage(content=system_msg),
        HumanMessage(content=(
            "Context: produce 4 concise insights for the pivot brand (1-2 sentences each) "
            "plus a suggested action per insight, and return as JSON. "
            f"DF preview: {df_preview}\n\n"
            "You may call the tool swot_tool by issuing a tool call. "
            "If you call the tool, the tool will return a JSON object you can use.\n"
            "Return a single assistant message containing either the insights JSON or tool calls."
        ))
    ]

    # Start a chat session with Gemini
    chat = generator_model.start_chat(history=[])
    
    # First message: system prompt
    system_response = chat.send_message(system_msg)
    
    # Second message: include data and tool instructions
    message = (
        f"Context: Here is the data preview and tools available:\n"
        f"Data preview: {df_preview}\n\n"
        f"You can request a SWOT analysis by responding with a JSON object like:\n"
        f'{{"tool_call": "swot_tool", "df_json": [...your data...], "pivot_brand": "brand name"}}\n\n'
        f"After receiving the SWOT analysis, provide 4 concise insights (1-2 sentences each) "
        f"plus a suggested action per insight, all in JSON format."
    )

    # Send the message and get response
    response = chat.send_message(message)
    content = response.text
    
    # Check if response is a tool call request. The model may embed JSON inside
    # markdown code fences or surrounding text, so use a robust extractor that
    # finds the JSON object which contains the key "tool_call" and parses it.
    def _extract_json_with_key(s: str, key: str = "tool_call"):
        # Fast path: try parsing whole content
        try:
            obj = json.loads(s)
            if isinstance(obj, dict) and key in obj:
                return obj
        except Exception:
            pass

        # Search for the key and then find a balanced JSON object around it
        k = f'"{key}"'
        ki = s.find(k)
        if ki == -1:
            return None

        # find an opening brace before the key
        open_idx = s.rfind('{', 0, ki)
        if open_idx == -1:
            open_idx = s.find('{', ki)
            if open_idx == -1:
                return None

        # scan forward to find the matching closing brace
        depth = 0
        i = open_idx
        end_idx = -1
        while i < len(s):
            if s[i] == '{':
                depth += 1
            elif s[i] == '}':
                depth -= 1
                if depth == 0:
                    end_idx = i
                    break
            i += 1

        if end_idx == -1:
            return None

        candidate = s[open_idx:end_idx+1]
        try:
            return json.loads(candidate)
        except Exception:
            return None

    parsed = _extract_json_with_key(content, "tool_call")


    if isinstance(parsed, dict) and parsed.get("tool_call") == "swot_tool":
        # Debug: show that a tool call was detected
        print("DEBUG: detected tool_call request from model; executing swot_tool...")
        try:
            df_json_arg = parsed.get("df_json", state.get("last_df_json", []))
            pivot_brand_arg = parsed.get("pivot_brand") or (df_json_arg[0].get("brand_name") if df_json_arg else "")
            print(f"DEBUG: pivot_brand_arg={pivot_brand_arg}; rows_for_swot={len(df_json_arg)}")
            tool_result = swot_analysis(pd.DataFrame(df_json_arg), pivot_brand=pivot_brand_arg)
            print("DEBUG: swot_tool executed; returning result to model")

            # Ensure tool_result is a dict with all SWOT keys
            swot_keys = ["Strengths", "Weaknesses", "Opportunities", "Threats"]
            if not isinstance(tool_result, dict):
                tool_result = {k: [] for k in swot_keys}
            else:
                for k in swot_keys:
                    if k not in tool_result:
                        tool_result[k] = []

            # Send the tool result back to continue the conversation
            followup = (
                f"Here is the SWOT analysis result (always with all keys):\n{json.dumps(tool_result, indent=2)}\n\n"
                f"Please provide your final insights and action items based on this SWOT analysis seeing the result stored in tool_result.  "
                f"Format as JSON with 'insights' array containing objects with 'insight' and 'action' fields."
            )
            final_response = chat.send_message(followup)
            content = final_response.text
        except Exception as e:
            print(f"DEBUG: swot_tool execution failed: {e}")
            content = json.dumps({"error": f"SWOT analysis failed: {str(e)}"})

    # Append to history (append semantics handled by StateGraph)
    history_entry = {
        "iteration": iteration,
        "generator_text": content,
    }

    return {"iteration": iteration, "last_generator_text": content, "history": [history_entry]}


def node_reflector(state: AgentState) -> dict:
    """
    Node E: reflector (Groq) that critiques the generator's latest output.
    It reads last_generator_text and returns critique text.
    """
    from langchain.messages import SystemMessage, HumanMessage

    system_msg = "You are a critical analyst. Given generator output, point out missing evidence, contradictions, and say whether to 'CONVERGE' or 'REVISE'."
    generator_text = state.get("last_generator_text", "")

    messages = [
        SystemMessage(content=system_msg),
        HumanMessage(content=f"Generator output:\n{generator_text}\n\nPlease respond with: 'decision: CONVERGED' or 'decision: REVISE' followed by a short critique and a list of up to 3 checks.")
    ]

    # Build the prompt for Gemini
    prompt = f"{system_msg}\n\n{messages[1].content}"
    
    # Generate response
    response = reflector_model.generate_content(prompt)
    critique_text = response.text

    # Append critique to history entry corresponding to latest iteration
    history_entry = {
        "iteration": state.get("iteration", 0),
        "reflector_text": critique_text,
    }

    return {"last_reflector_text": critique_text, "history": [history_entry]}


# -------------------------
# Conditional edge function = Node D (stop criteria)
# This function receives the whole State and must return either "generator" or END or "reflector".
# We'll implement: if iteration >= max_iters OR 'converge' token appears in last_reflector_text OR
# generator convergence via textual similarity -> stop (END). Otherwise go to reflector.
# -------------------------
MAX_ITERS = 3
SIM_THRESHOLD = 0.86  # textual similarity threshold for generator outputs convergence


def should_continue(state: AgentState) -> Literal["reflector", "end"]:
    it = state.get("iteration", 0)
    if it >= MAX_ITERS:
        return END

    # If reflector asked to CONVERGE, stop
    last_ref = (state.get("last_reflector_text") or "").lower()
    if "converge" in last_ref or "converged" in last_ref:
        return END

    # textual similarity check between last two generator outputs in history
    hist = state.get("history", [])
    # extract generator_texts from history
    gen_texts = [h.get("generator_text") for h in hist if h.get("generator_text")]
    if len(gen_texts) >= 2:
        a, b = gen_texts[-2], gen_texts[-1]
        sim = difflib.SequenceMatcher(None, a or "", b or "").ratio()
        if sim >= SIM_THRESHOLD:
            return END

    # otherwise continue to reflector (so loop will go: generator -> reflector -> generator -> ...)
    return "reflector"


# -------------------------
# Build the StateGraph
# -------------------------
def build_agency_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # add nodes (names correspond to the functions added)
    graph.add_node("fetch_top5", node_fetch_top5)
    graph.add_node("normalize", node_normalize)
    graph.add_node("generator", node_generator)
    graph.add_node("reflector", node_reflector)

    # connect nodes:
    graph.add_edge(START, "fetch_top5")
    graph.add_edge("fetch_top5", "normalize")
    graph.add_edge("normalize", "generator")

    # conditional: after generator, either go to reflector (continue loop) or END (stop)
    graph.add_conditional_edges("generator", should_continue, ["reflector", END])

    # after reflector, go back to generator (loop)
    graph.add_edge("reflector", "generator")

    return graph

import ast
from typing import Any

def compose_final_analysis(state: AgentState) -> Any:
    """
    Build a final analysis from the state's history and last outputs.
    Strategy:
      - Look for structured JSON in the last generator_text (try json.loads)
      - If JSON parse fails, fall back to concatenating recent generator_text entries
      - Return a dict if parsed JSON found, else a single string
    """
    hist = state.get("history", []) or []
    # collect generator_text entries (most recent last)
    gen_texts = [h.get("generator_text") for h in hist if h.get("generator_text")]
    if not gen_texts:
        # fallback to any last_generator_text
        gen_texts = [state.get("last_generator_text", "")] if state.get("last_generator_text") else []

    last = gen_texts[-1] if gen_texts else ""

    # Try JSON first
    try:
        parsed = json.loads(last)
        return parsed
    except Exception:
        # Try ast.literal_eval as a more permissive fallback for Python-style dicts/lists
        try:
            parsed = ast.literal_eval(last)
            return parsed
        except Exception:
            # Otherwise, join the last few generator outputs into a summary string
            preview = " ".join(gen_texts[-3:])  # last 3 entries
            # Shorten to a reasonable length
            return preview.strip()
# -------------------------
# Invocation / entrypoint
# -------------------------
if __name__ == "__main__":
    # Hard-coded placeholders for product & country per your request:
    initial_state: AgentState = {
        "product": "lipstick",
        "country": "india",
        "raw_top5": [],
        "last_df_json": [],
        "history": [],    # append-mode
        "iteration": 0,
        "last_generator_text": "",
        "last_reflector_text": "",
        "final_analysis": "",
    }

    graph = build_agency_graph()
    compiled_agent = graph.compile()

    # Invoke the graph. The compiled agent will run nodes per the graph edges/conditionals.
    final_state = compiled_agent.invoke(initial_state)
# -----------------
    # # final_state contains the updated state after run. The final_analysis can be composed
    # # from the state's history or generator model as needed (here we just print history).
    # print("=== HISTORY ===")
    # for entry in final_state["history"]:
    #     print(entry)
    # print("=== FINAL ===")
    # print(final_state.get("final_analysis", "(no final analysis produced)"))
    # Compose final analysis and attach to state
    final_analysis = compose_final_analysis(final_state)
    final_state["final_analysis"] = final_analysis

    # Print the history and final analysis (pretty)
    print("=== HISTORY ===")
    for entry in final_state.get("history", []):
        print(entry)
    print("=== FINAL ANALYSIS ===")
    if isinstance(final_analysis, (dict, list)):
        print(json.dumps(final_analysis, indent=2, ensure_ascii=False))
    else:
        print(final_analysis)

    from IPython.display import display,Image
    display(Image(compiled_agent.get_graph().draw_mermaid_png()))
