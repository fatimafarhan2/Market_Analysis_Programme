#reviews_summary_tool.py

import re
import json
import pandas as pd
from typing import Dict, Any, List
from langchain_core.tools import tool
from db_connect import supabase
from typing import Annotated

# -----------------------
# Utility to sanitize SQL
# -----------------------
def _clean_sql_for_rpc(sql: str) -> str:
    """Remove SQL comments and trailing semicolons before sending to Supabase RPC."""
    if not sql:
        return sql
    clean_sql = re.sub(r'(--.*?$)|(/\*.*?\*/)', '', sql, flags=re.MULTILINE | re.DOTALL).strip()
    while clean_sql.endswith(";"):
        clean_sql = clean_sql[:-1].strip()
    return clean_sql

# -----------------------
# Execute SQL via Supabase RPC
# -----------------------
def _run_sql(supabase_client, sql: str) -> List[Dict[str, Any]]:
    sql_clean = _clean_sql_for_rpc(sql)
    try:
        res = supabase_client.rpc("run_sql", {"sql": sql_clean}).execute()
    except Exception as e:
        print("RPC call failed:", e)
        return []
    data = getattr(res, "data", None)
    if data is None:
        return []
    if isinstance(data, str):
        try:
            parsed = json.loads(data)
            return parsed if isinstance(parsed, list) else [parsed]
        except Exception:
            return []
    return data

# -----------------------
# SQL to load reviews flat table
# -----------------------
_REVIEWS_FLAT_SELECT = """
SELECT
    product_id,
    product_name,
    brand_id,
    brand_name,
    category,
    key_function,
    price_range,
    country_name,
    total_units_sold,
    avg_returns_rate,
    avg_rating,
    total_num_reviews,
    top_common_keywords,
    review
FROM reviews_ftable_m
"""

# -----------------------
# Convert to DataFrame
# -----------------------
def _to_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    numeric_cols = ["total_units_sold", "avg_returns_rate", "avg_rating", "total_num_reviews"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# -----------------------
# Core logic function
# -----------------------
def get_reviews_summary(supabase_client=None) -> Dict[str, Any]:
    """
    Core function to fetch and aggregate reviews from 'reviews_ftable_m'.
    Groups identical product records and collects all reviews in a list.
    Returns a structured summary for each unique product.
    """
    sup = supabase_client or supabase
    results: Dict[str, Any] = {"notes": "Reviews summary extracted", "errors": []}

    # Fetch rows from Supabase
    rows = _run_sql(sup, _REVIEWS_FLAT_SELECT)
    df = _to_df(rows)
    results["meta_rows_fetched"] = len(df)

    if df.empty:
        results["errors"].append("No rows returned from reviews_ftable_m.")
        return results

    # Group reviews by unique product attributes
    group_cols = [
        "product_id", "product_name", "brand_id", "brand_name",
        "category", "key_function", "price_range", "country_name",
        "total_units_sold", "avg_returns_rate", "avg_rating",
        "total_num_reviews", "top_common_keywords"
    ]

    grouped = (
        df.groupby(group_cols, dropna=False)
        .agg({"review": lambda x: list(x.dropna().unique())})
        .reset_index()
    )

    results["reviews_summary"] = grouped.to_dict(orient="records")
    return results

# -----------------------
# LangChain-compatible Tool
# -----------------------
# LangChain-compatible Tool (v0.1.x)
@tool(return_direct=False)
def fetch_reviews_summary(supabase_client=None) -> Dict[str, Any]:
    """
    Fetch and aggregate product reviews from 'reviews_ftable_m'.
    LangChain tool wrapper for get_reviews_summary().
    """
    print("Invoking fetch_reviews_summary tool...")
    return get_reviews_summary(supabase_client)


# -----------------------
# CLI local run
# -----------------------
if __name__ == "__main__":
    print("Fetching reviews summary locally...")
    # Directly call the core function for local testing
    output = get_reviews_summary()
    print(json.dumps(output, indent=2, ensure_ascii=False))
