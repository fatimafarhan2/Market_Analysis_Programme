"""
top5_brands_by_revenue.py

Fetch top 5 brands (rows) from product_flat_table by total_revenue_usd.
Assumes one product type per brand (no aggregation needed).

Requirements:
  pip install supabase python-dotenv pandas
Optional (only if you want LLM summary):
  pip install google-ai-generativelanguage

Usage:
  - Put SUPABASE_URL and SUPABASE_KEY in a .env file or in environment variables.
  - Run: python top5_brands_by_revenue.py
"""

import os
import json
from dotenv import load_dotenv
from supabase import create_client
import pandas as pd
from typing import List, Dict, Any
from google import genai

# --- Config / load env ---
load_dotenv()  # reads .env into environment
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# ✅ Get API key from .env
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError("Please set GOOGLE_API_KEY in your environment or .env file")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Please set SUPABASE_URL and SUPABASE_KEY in your environment or .env file")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# --- Columns we will fetch (only what's needed) ---
COLUMNS = ",".join([
    "product_id","product_name","brand_id","brand_name","category",
    "key_function","price_range","total_units_sold","total_revenue_usd",
    "avg_revenue_usd","total_net_units_sold","total_net_revenue_usd",
    "avg_returns_rate","avg_marketing_spend_usd",
    "avg_marketing_efficiency_ratio","avg_return_on_marketing_spend",
    "avg_online_sales_ratio","avg_rating","avg_sentiment_score",
    "total_num_reviews","avg_population_millions","avg_income_usd",
    "avg_urbanization_rate","avg_online_shopping_penetration",
    "top_common_keywords","dominant_distribution_channel",
    "country_id","country_name","region","currency"
])


def fetch_top5_by_revenue(limit: int = 5) -> List[Dict[str, Any]]:
    """
    Query Supabase product_flat_table ordering by total_revenue_usd desc and return top `limit` rows.
    """
    # Use select with explicit columns, order and limit to let DB do the sorting
    resp = supabase.table("product_flat_table") \
                   .select(COLUMNS) \
                   .order("total_revenue_usd", desc=True) \
                   .limit(limit) \
                   .execute()

    # Basic error handling (supabase-py returns a response-like object)
    # Different versions may return different shapes: handle common ones
    data = None
    if isinstance(resp, dict):
        # older/newer client shapes
        data = resp.get("data")
        status = resp.get("status")
        error = resp.get("error")
        if error:
            raise RuntimeError(f"Supabase error: {error}")
    else:
        # when resp has .data attribute (supabase-py)
        data = getattr(resp, "data", None)
        status = getattr(resp, "status_code", None)

    if data is None:
        raise RuntimeError(f"Failed to fetch data from Supabase. Raw response: {resp}")

    return data


def normalize_and_validate(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert to DataFrame, coerce numerics, trim whitespace on strings, and add simple validation.
    Returns a pandas DataFrame with cleaned values.
    """
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Trim string columns that we will display
    for col in ["product_id","product_name","brand_id","brand_name","country_name","currency"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Coerce numeric columns
    numeric_cols = [
        "total_units_sold","total_revenue_usd","avg_revenue_usd",
        "total_net_units_sold","total_net_revenue_usd","avg_returns_rate",
        "avg_marketing_spend_usd","avg_marketing_efficiency_ratio",
        "avg_return_on_marketing_spend","avg_online_sales_ratio","avg_rating",
        "avg_sentiment_score","total_num_reviews","avg_population_millions",
        "avg_income_usd","avg_urbanization_rate","avg_online_shopping_penetration"
    ]
    for n in numeric_cols:
        if n in df.columns:
            df[n] = pd.to_numeric(df[n], errors="coerce")

    # Replace NaN in revenue/units with 0 to make sorting safe
    if "total_revenue_usd" in df.columns:
        df["total_revenue_usd"] = df["total_revenue_usd"].fillna(0.0)
    if "total_units_sold" in df.columns:
        df["total_units_sold"] = df["total_units_sold"].fillna(0).astype(int)

    return df


def pretty_print_top5(df: pd.DataFrame):
    """
    Prints a small human-readable table for the top 5.
    """
    if df.empty:
        print("No rows returned.")
        return

    display_cols = [
        "brand_id","brand_name","product_id","product_name",
        "country_name","currency","total_units_sold","total_revenue_usd"
    ]
    # keep only columns that exist
    display_cols = [c for c in display_cols if c in df.columns]

    df_display = df[display_cols].copy()
    # format revenue nicely
    if "total_revenue_usd" in df_display.columns:
        df_display["total_revenue_usd"] = df_display["total_revenue_usd"].map(lambda v: f"{v:,.2f}")

    print("\nTop {} brands by total_revenue_usd:\n".format(len(df_display)))
    print(df_display.to_string(index=False))


# Updated rows_to_json_safe + Gemini caller
def rows_to_json_safe_all_columns(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Convert the DataFrame rows into a JSON-serializable list of dicts containing ALL columns.
    - Keeps df unchanged (reads only).
    - Converts pandas/numpy types to plain Python types.
    - Replaces NaN with None (so JSON will use null), or empty string for string fields if preferred.
    """
    out: List[Dict[str, Any]] = []
    if df is None or df.empty:
        return out

    # iterate rows safely
    for _, row in df.iterrows():
        raw = row.to_dict()
        safe_row: Dict[str, Any] = {}
        for col, val in raw.items():
            # handle pandas / numpy NA
            if pd.isna(val):
                # Use None (-> null in JSON). If you prefer "", change to "".
                safe_row[col] = None
                continue

            # basic type normalization to ensure JSON compatibility
            if isinstance(val, (pd.Timestamp,)):
                # isoformat for timestamps
                safe_row[col] = val.isoformat()
            elif isinstance(val, (pd.Int16Dtype, pd.Int32Dtype, pd.Int64Dtype)) or (isinstance(val, (int,)) and not isinstance(val, bool)):
                safe_row[col] = int(val)
            elif isinstance(val, (float,)) and not pd.isna(val):
                # round floats to 6 decimals to avoid very long reprs (adjust if needed)
                safe_row[col] = float(round(val, 6))
            elif isinstance(val, (list, dict)):
                # if already a container, try to json-serialize as-is
                safe_row[col] = val
            else:
                # fallback: convert to str for other types (e.g., numpy.str_, objects)
                safe_row[col] = str(val)
        out.append(safe_row)

    return out


def call_gemini_list_top5(json_result):
    api_key = os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)

    prompt = f"""
    You are a data explainer. Given this JSON of top 5 brands and their product details,
    write a natural, human-readable paragraph listing each brand, its revenue, and key highlights. Relate columns
    to each other where relevant (e.g., revenue vs units sold, rating vs sentiment).
    []
    Do NOT analyze or interpret — just list clearly.

    JSON data:
    {json_result}
    """

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt]
    )

    # Extract LLM text
    return response.text


def main():
    raw_rows = fetch_top5_by_revenue(limit=5)
    df = normalize_and_validate(raw_rows)
    pretty_print_top5(df)
    json_result = rows_to_json_safe_all_columns(df)
    # Save json_result to CSV
    # pd.DataFrame(json_result).to_csv("top5_brands_by_revenue.csv", index=False)
    print("Saved top 5 brands to top5_brands_by_revenue.csv")
    # # Print JSON for downstream consumption
    # print("\nCalling llm on JSON:")
    listing_text = call_gemini_list_top5(json_result)
    print(listing_text)

    return json_result


if __name__ == "__main__":
    top5 = main()
