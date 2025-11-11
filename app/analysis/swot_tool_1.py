# this file has swot logic whenuser inputs g, price, ingredients, country --> its swot func being used in test_swot.py
"""
swot_agent.py

Single-file, copy-pasteable implementation:
- Uses langchain/langgraph imports you asked for
- @tool swot_tool implements the SWOT logic you defined:
    * price comparison uses avg_revenue_usd
    * size uses g_size
    * banned ingredients fetched from Supabase Countries.banned_ingredients
    * opportunities/threats use <6 => opportunity else threat
    * recommendation uses mean(avg_marketing_spend_usd)
- Wires a small LangGraph StateGraph:
    START -> LLM node -> (tools_condition) -> ToolNode -> END
- Example usage at bottom demonstrates calling the graph programmatically.

Requirements (install as needed):
    pip install python-dotenv pandas numpy supabase py-httpx langchain langgraph
    # package names and versions vary; adapt to your environment
"""

import os
import json
from typing import Any, Dict, List, TypedDict, Optional

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from supabase import create_client

# LangChain / LangGraph imports (modern pattern)
from langchain.chat_models import init_chat_model
from typing_extensions import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition

# -------------------------
# Load environment & clients
# -------------------------
load_dotenv()  # expects .env in working directory with SUPABASE_URL, SUPABASE_KEY, GEMINI_API_KEY

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise EnvironmentError("Missing SUPABASE_URL or SUPABASE_KEY in .env")

# create supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# -------------------------
# Small helpers
# -------------------------
def safe_mean(series) -> Optional[float]:
    arr = pd.to_numeric(series, errors="coerce")
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return None
    return float(arr.mean())


def normalize_ingredients(ings: List[str]) -> List[str]:
    return [str(i).lower().strip() for i in (ings or [])]


def fetch_banned_ingredients_from_supabase(country_id: Any) -> List[str]:
    """
    Fetch banned_ingredients from 'country' table in Supabase.
    Accepts JSON array or comma-separated string stored in the column.
    Returns normalized list of strings (lowercase).
    """
    if country_id is None:
        return []
    resp = supabase.table("Countries").select("banned_ingredients").eq("country_id", country_id).limit(1).execute()
    # supabase-py shape: resp.data
    if not resp or not hasattr(resp, "data") or not resp.data:
        return []
    raw = resp.data[0].get("banned_ingredients")
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x).lower().strip() for x in raw]
    if isinstance(raw, str):
        s = raw.strip()
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(x).lower().strip() for x in parsed]
        except Exception:
            return [p.strip().lower() for p in s.split(",") if p.strip()]
    return []


# -------------------------
# State type for LangGraph
# -------------------------
class SWOTState(TypedDict):
    messages: Annotated[list, add_messages]
    # df: we will pass it in runtime as a pandas.DataFrame
    df: Any
    # json_result is the top5 brands JSON (or a record containing country_id)
    json_result: Any
    user_query: str


# -------------------------
# The SWOT tool 
# -------------------------
@tool
def swot_tool(user_query: str, df: List[Dict[str, Any]], json_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inputs:
      - user_query: the raw user prompt (e.g. "i want to launch 10g lipstick for $20, its ingredients include beeswax etc")
      - df: top brands data, passed as list-of-dicts (json) or pandas DataFrame-compatible structure
      - json_result: additional json record (may include country_id). This mirrors your 'json_result' input.

    Output:
      - JSON-serializable dict with schema_version "swot_v1" and fields:
        computed_means, strengths, weaknesses, opportunities, threats, suggestion, metadata
    """
    # Accept df as either list-of-dicts or pandas DataFrame
    try:
        if isinstance(df, list):
            top_brands = pd.DataFrame(df)
        elif isinstance(df, pd.DataFrame):
            top_brands = df.copy()
        else:
            # try parse JSON string
            top_brands = pd.DataFrame(json.loads(df))
    except Exception as e:
        return {"error": f"failed to parse df/top_brands: {str(e)}"}

    # --- Parse user query (simple heuristics as before) ---
    uq = (user_query or "").lower()
    user_price = None
    user_size = None
    user_ingredients: List[str] = []

    # quick parse for "$" and "g"
    for token in uq.replace(",", " ").split():
        if token.startswith("$"):
            try:
                user_price = float(token.replace("$", ""))
            except Exception:
                pass
        if token.endswith("g"):
            try:
                user_size = float(token.replace("g", ""))
            except Exception:
                pass

    # parse ingredients phrase "include" or "ingredients"
    if "include" in uq:
        part = uq.split("include", 1)[1]
        # stop at "for" or end
        part = part.split(" for ")[0]
        user_ingredients = [x.strip() for x in part.replace("etc", "").split(",") if x.strip()]
    elif "ingredients" in uq:
        part = uq.split("ingredients", 1)[1]
        user_ingredients = [x.strip() for x in part.replace("etc", "").split(",") if x.strip()]

    # fallbacks
    user_price = float(user_price) if user_price is not None else 0.0
    user_size = float(user_size) if user_size is not None else 0.0
    user_ingredients = normalize_ingredients(user_ingredients)

    # --- Compute means (explicit columns as you requested) ---
    mean_price = safe_mean(top_brands.get("avg_revenue_usd", pd.Series(dtype=float)))
    mean_g_size = safe_mean(top_brands.get("g_size", pd.Series(dtype=float)))
    mean_avg_rating = safe_mean(top_brands.get("avg_rating", pd.Series(dtype=float)))
    mean_avg_sentiment = safe_mean(top_brands.get("avg_sentiment_score", pd.Series(dtype=float)))
    mean_avg_returns_rate = safe_mean(top_brands.get("avg_returns_rate", pd.Series(dtype=float)))
    mean_marketing_spend = safe_mean(top_brands.get("avg_marketing_spend_usd", pd.Series(dtype=float)))

    strengths: List[str] = []
    weaknesses: List[str] = []
    opportunities: List[str] = []
    threats: List[str] = []

    # --- Strengths / Weaknesses rules (exact as specified) ---
    if mean_price is None:
        weaknesses.append("mean(avg_revenue_usd) not available in top brands.")
    else:
        if mean_price > user_price:
            strengths.append(f"user price (${user_price:.2f}) is lower than market mean (${mean_price:.2f})")
        else:
            weaknesses.append(f"user price (${user_price:.2f}) is NOT lower than market mean (${mean_price:.2f})")

    if mean_g_size is None:
        weaknesses.append("mean(g_size) not available in top brands.")
    else:
        if mean_g_size > user_size:
            weaknesses.append(f"user size ({user_size}g) is smaller than market mean size ({mean_g_size:.2f}g)")
        else:
            strengths.append(f"user size ({user_size}g) is >= market mean size ({mean_g_size:.2f}g)")

    # --- Banned ingredients: determine country_id then fetch list from Supabase ---
    country_id = None
   
    # Step 1: Try to get country_id from JSON result
    if isinstance(json_result, dict) and "country_id" in json_result:
        country_id = str(json_result.get("country_id")).strip()  # ensure string type

    # Step 2: Fallback — check the DataFrame (top_brands)
    elif "country_id" in top_brands.columns and len(top_brands) > 0:
        country_id = str(top_brands["country_id"].iloc[0]).strip()  # also ensure string type

    # Step 3: Initialize banned ingredient list
    banned_list: List[str] = []

    # Step 4: Handle if no country_id found
    if not country_id:
        weaknesses.append("country_id not provided; cannot check banned ingredients.")
    else:
        try:
            # country_id is a text key in your Supabase 'countries' table
            banned_list = fetch_banned_ingredients_from_supabase(country_id)
        except Exception as e:
            banned_list = []
            weaknesses.append(f"Error fetching banned ingredients for {country_id}: {str(e)}")

        # Step 5: Check user ingredients vs banned list
        if banned_list:
            matches = [ui for ui in user_ingredients for b in banned_list if b and b.lower() in ui.lower()]
            if matches:
                weaknesses.append(
                    f"User product contains banned ingredients for country {country_id}: {sorted(set(matches))}"
                )
            else:
                strengths.append(f"No banned ingredients detected for country {country_id}.")
        else:
            # Empty banned list means no banned ingredients defined in DB
            strengths.append(f"No banned ingredients listed for country {country_id} (or list empty).")

    # --- Opportunities / Threats rules (mean < 6 => opportunity else threat) ---
    def classify_mean(v: Optional[float], label: str):
        if v is None:
            return ("unknown", f"{label} mean not available")
        if v < 6:
            return ("opportunity", f"mean {label} = {v:.2f}")
        return ("threat", f"mean {label} = {v:.2f}")

    c_rating, msg_rating = classify_mean(mean_avg_rating, "avg_rating")
    if c_rating == "opportunity":
        opportunities.append(msg_rating)
    elif c_rating == "threat":
        threats.append(msg_rating)
    else:
        opportunities.append(msg_rating)

    c_sent, msg_sent = classify_mean(mean_avg_sentiment, "avg_sentiment_score")
    if c_sent == "opportunity":
        opportunities.append(msg_sent)
    elif c_sent == "threat":
        threats.append(msg_sent)
    else:
        opportunities.append(msg_sent)

    c_ret, msg_ret = classify_mean(mean_avg_returns_rate, "avg_returns_rate")
    if c_ret == "opportunity":
        opportunities.append(msg_ret)
    elif c_ret == "threat":
        threats.append(msg_ret)
    else:
        opportunities.append(msg_ret)

    # --- Suggestion: marketing spend ---
    suggestion = {}
    if mean_marketing_spend is None:
        suggestion["recommended_min_marketing_budget_usd"] = None
        suggestion["note"] = "avg_marketing_spend_usd not available in top brands data."
    else:
        suggestion["recommended_min_marketing_budget_usd"] = round(mean_marketing_spend, 2)
        suggestion["note"] = "Mean avg_marketing_spend_usd across top brands — use as minimum benchmark."

    # --- Final structured result (JSON-ready) ---
    result = {
        "schema_version": "swot_v1",
        "user_product": {
            "product_name": None,
            "price_usd": round(user_price, 2),
            "g_size": round(user_size, 2),
            "ingredients": user_ingredients,
            "country_id": country_id,
        },
        "computed_means": {
            "mean_avg_revenue_usd": round(mean_price, 2) if mean_price is not None else None,
            "mean_g_size": round(mean_g_size, 2) if mean_g_size is not None else None,
            "mean_avg_rating": round(mean_avg_rating, 2) if mean_avg_rating is not None else None,
            "mean_avg_sentiment_score": round(mean_avg_sentiment, 2) if mean_avg_sentiment is not None else None,
            "mean_avg_returns_rate": round(mean_avg_returns_rate, 2) if mean_avg_returns_rate is not None else None,
            "mean_marketing_spend_usd": round(mean_marketing_spend, 2) if mean_marketing_spend is not None else None,
            "banned_ingredients_checked": banned_list,
        },
        "strengths": strengths,
        "weaknesses": weaknesses,
        "opportunities": opportunities,
        "threats": threats,
        "suggestion": suggestion,
        "metadata": {"notes": "Output intended for downstream reflector LLM. schema_version=swot_v1"}
    }

    return result


# -------------------------
# Register tools list (LangGraph will pick this up via ToolNode)
# -------------------------
tools = [swot_tool]


# -------------------------
# Build a minimal LangGraph flow (LLM -> Tools -> END)
# -------------------------
def build_swot_stategraph() -> StateGraph:
    """
    Returns a compiled StateGraph that:
      START -> LLM node -> (tools_condition) -> ToolNode(tools) -> END
    The LLM node uses init_chat_model(...).bind_tools under the hood (langgraph handles this).
    """
    # create a StateGraph typed by SWOTState
    graph = StateGraph(SWOTState)

    # Create an LLM wrapper using the same init_chat_model API you use.
    # Note: langgraph will expect the model variable to be an LLM object that has .bind_tools(...) internally.
    llm = init_chat_model("google_genai:gemini-2.0-flash")  # gemini key taken from environment by your configured provider

    # In many setups langgraph expects the LLM instance that already supports tool binding.
    # ToolNode is constructed with the tool list we provided.
    tool_node = ToolNode(tools)

    # Add nodes to the graph
    graph.add_node("llm", llm)
    graph.add_node("tools", tool_node)

    # wire edges: START -> llm
    graph.add_edge(START, "llm")
    # When LLM decides a tool should run, tools_condition routes to tool node
    graph.add_conditional_edges("llm", tools_condition)
    # When tools finish, go to END
    graph.add_edge("tools", END)

    return graph


# -------------------------
# Example usage: call the graph programmatically
# -------------------------
if __name__ == "__main__":
    # sample minimal top_brands df
    sample_df = pd.DataFrame(
        [
            {"product_id": 1, "product_name": "A", "brand_id": 1, "brand_name": "A", "category": "lipstick",
             "avg_revenue_usd": 22.5, "g_size": 12.0, "avg_rating": 5.0, "avg_sentiment_score": 5.2,
             "avg_returns_rate": 4.5, "avg_marketing_spend_usd": 40000, "country_id": 1},
            {"product_id": 2, "product_name": "B", "brand_id": 2, "brand_name": "B", "category": "lipstick",
             "avg_revenue_usd": 24.0, "g_size": 11.0, "avg_rating": 6.5, "avg_sentiment_score": 6.2,
             "avg_returns_rate": 7.0, "avg_marketing_spend_usd": 60000, "country_id": 1},
            {"product_id": 1, "product_name": "C", "brand_id": 32, "brand_name": "Z", "category": "lipstick",
             "avg_revenue_usd": 25, "g_size": 14.0, "avg_rating": 7.0, "avg_sentiment_score": 7.2,
             "avg_returns_rate": 3.5, "avg_marketing_spend_usd": 45000, "country_id": 1},
            {"product_id": 2, "product_name": "D", "brand_id": 21, "brand_name": "X", "category": "lipstick",
             "avg_revenue_usd": 23.0, "g_size": 11.0, "avg_rating": 4.5, "avg_sentiment_score": 8.2,
             "avg_returns_rate": 6.0, "avg_marketing_spend_usd": 50000, "country_id": 1},
            {"product_id": 1, "product_name": "E", "brand_id": 11, "brand_name": "Y", "category": "lipstick",
             "avg_revenue_usd": 20.0, "g_size": 13.0, "avg_rating": 4.0, "avg_sentiment_score": 3.2,
             "avg_returns_rate": 8.5, "avg_marketing_spend_usd": 35000, "country_id": 1},
        ]
    )

    # Example user prompt
    user_prompt = "i want to launch 10g lipstick for $20, its ingredients include beeswax, castor oil"
    df_json = sample_df.to_dict(orient="records")
    json_result = {"country_id": 1}

    # Build graph and compile (langgraph compile step depends on your setup; here we use run_single for demonstration)
    sg = build_swot_stategraph()
    compiled = sg.compile()

    # Execute: many langgraph setups accept compiled.run(...) or compiled.start(...)
    # The exact call depends on langgraph version. Common pattern:
    try:
        # A basic run where we set the initial state
        initial_state = SWOTState(messages=[], df=df_json, json_result=json_result, user_query=user_prompt)
        # run the compiled graph; different runtimes may differ in API
        output = compiled.run(initial_state)  # if your langgraph uses .run or .start, adapt accordingly
    except Exception as e:
        # fallback: directly call the tool programmatically (this always works)
        print(f"Graph execution failed: {str(e)}. Falling back to direct tool call.")
        # output = swot_tool(user_query=user_prompt, df=df_json, json_result=json_result)

    # output is either the graph result or the tool result dict
    print(json.dumps(output, indent=2))
