# new implementation of swot so we don't have to pass df in evry nodes' state
"""
top5_brands_by_revenue.py

Fetch top 20 brands from product_flat_table by total_revenue_usd,
filtered to a specific country (provided by user).
"""

import os
from typing import Annotated
import json
from dotenv import load_dotenv
from supabase import create_client
import pandas as pd
from typing import List, Dict, Any, Optional
from google import genai
from langchain_core.tools import tool

# --- Config / load env ---
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Please set SUPABASE_URL and SUPABASE_KEY in your environment or .env file")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Columns to fetch ---
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

PRODUCT = "Shampoo"
COUNTRY = "Japan"

def rows_to_json_safe_all_columns(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert DataFrame rows to a JSON-safe list."""
    if df.empty:
        return []
    return json.loads(df.to_json(orient="records"))

def fetch_top5_brands_by_revenue(
    product: str,
    country: str
) -> List[Dict[str, Any]]:
    """
    Fetch top brands by total revenue for a specific product,
    optionally filtered by country.
    """
    query = supabase.table("product_flat_table").select(COLUMNS)

    # Filter by product (name or ID)
    if product:
        if str(product).isdigit():
            query = query.eq("product_id", int(product))
        else:
            query = query.ilike("product_name", f"%{product}%")

    # Optional country filter
    if country:
        if str(country).isdigit():
            query = query.eq("country_id", int(country))
        else:
            query = query.ilike("country_name", f"%{country}%")

    # Order by revenue and limit results
    query = query.order("total_revenue_usd", desc=True).limit(5)
    resp = query.execute()

    data = getattr(resp, "data", None)
    if not data:
        print(f"No results found for product: {product} in country: {country}")
        return []

    return data


def normalize_and_validate(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Clean and prepare the data."""
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Clean string columns
    for col in ["product_id","product_name","brand_id","brand_name","country_name","currency"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Convert numeric columns
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
            df[n] = pd.to_numeric(df[n], errors="coerce").fillna(0)

    return df
@tool
def swot_analysis(query: str) -> Dict[str, Any]:
    """
    Perform a structured SWOT (Strengths, Weaknesses, Opportunities, Threats) analysis
    for a selected brand within a competitive dataset of cosmetic or product market data.

    This tool compares the specified pivot brand against its competitors using both 
    numeric and categorical data from the provided DataFrame. The goal is to identify 
    areas where the pivot brand performs better (strengths), underperforms (weaknesses), 
    potential areas for market growth (opportunities), and external risks posed by 
    competing brands (threats).

    The analysis dynamically handles numeric columns through mean-based comparisons 
    and categorical columns through mode-based comparisons. It also ensures that 
    insights are data-driven — each observation includes relevant figures (e.g., 
    numeric differences, mean comparisons, or categorical contrasts) to make findings 
    more meaningful and evidence-backed.

    Returns
    -------
    dict
        A dictionary with four keys: "Strengths", "Weaknesses", "Opportunities", and "Threats".
        Each key maps to a list of textual insights containing data-backed observations.
        
        Example:
        {
            "Strengths": [
                "Strong avg_rating: 4.6 vs avg 4.2",
                "High online sales ratio: 0.78 vs avg 0.65"
            ],
            "Weaknesses": [
                "Higher weakness in avg_returns_rate: 0.15 vs avg 0.10"
            ],
            "Opportunities": [
                "Market potential in avg_income_usd: 55000.00 vs avg 48000.00"
            ],
            "Threats": [
                "Competitor 'BrandX' outperforms pivot in avg_sentiment_score: 4.7 vs pivot 4.4"
            ]
        }

    Notes
    -----
    - Strengths and weaknesses are derived from **internal comparisons** (pivot vs. competitor means).
    - Opportunities and threats are derived from **external factors** (market or categorical differences).
    - Non-numeric fields are handled through mode frequency comparisons.
    - Returns are concise and suitable for LLM consumption or downstream visualization.

    Intended Use
    -------------
    This tool is designed to be invoked within a LangChain or LangGraph pipeline
    as part of a multi-step agentic analysis (e.g., used by a generator or reasoning node
    to gather structured SWOT data before generating strategic insights).
    """

    # INTEGRATION POINT: fetch product and country from globals or state
    # Expect product / country to be available in the generator's state as globals.
    product = "shampoo"  # e.g., shampoo
    country = "Japan"
    
    if product is None and country is None:
        # If both are missing, still try an empty fetch (will return nothing) but warn.
        print("Warning: PRODUCT and COUNTRY not found in globals. Attempting fetch without filters.")

    # fetch top rows (this returns a list of dict rows)
    rows = fetch_top5_brands_by_revenue(product=product, country=country)
    pivot_brand = rows[0]["brand_name"] if rows else None

    if not rows:
        return {"swot": {"Strengths": [], "Weaknesses": [], "Opportunities": [], "Threats": []},
                "rows_json": [],
                "pivot_brand": None}

    # Normalize into DataFrame
    df = normalize_and_validate(rows)

    # Provide JSON-safe rows as requested by generator
    rows_json = rows_to_json_safe_all_columns(df)

    pivot = df[df["brand_name"] == pivot_brand].iloc[0]
    competitors = df[df["brand_name"] != pivot_brand]

    # Define column groups
    strengths_cols = [
        "total_revenue_usd", "avg_revenue_usd", "total_units_sold", "avg_rating",
        "avg_sentiment_score", "avg_return_on_marketing_spend", 
        "avg_marketing_efficiency_ratio", "avg_online_sales_ratio"
    ]
    weaknesses_cols = [
        "avg_returns_rate", "avg_marketing_spend_usd", "price_range", 
        "total_net_units_sold", "total_net_revenue_usd"
    ]
    opportunities_cols = [
        "avg_population_millions", "avg_income_usd", "avg_urbanization_rate", 
        "avg_online_shopping_penetration", "region", "category"
    ]
    threats_cols = [
        "avg_rating", "avg_sentiment_score", "avg_marketing_spend_usd", "avg_online_sales_ratio"
        "country_name", "total_revenue_usd" 
    ]

    results = {"Strengths": [], "Weaknesses": [], "Opportunities": [], "Threats": []}

    # Strengths and Weaknesses (internal comparison)
    for col in strengths_cols:
        if col not in df.columns:
            continue
        # try numeric comparison first
        comp_numeric = pd.to_numeric(competitors[col], errors="coerce").dropna()
        if not comp_numeric.empty:
            comp_mean = comp_numeric.mean()
            try:
                pivot_val = float(pd.to_numeric(pivot[col], errors="coerce"))
                if pivot_val > comp_mean:
                    results["Strengths"].append(f"Strong {col}: {pivot_val:.2f} vs avg {comp_mean:.2f}")
                else:
                    results["Weaknesses"].append(f"Weak {col}: {pivot_val:.2f} vs avg {comp_mean:.2f}")
            except Exception:
                # pivot value not numeric despite numeric competitors; skip
                continue
        else:
            # non-numeric/categorical comparison: compare to most common competitor value
            comp_mode = competitors[col].mode(dropna=True)
            if not comp_mode.empty:
                mode_val = comp_mode.iloc[0]
                if pivot[col] == mode_val:
                    results["Strengths"].append(f"{col}: {pivot_brand} matches competitors' common value ({pivot[col]})")
                else:
                    results["Weaknesses"].append(f"{col}: {pivot_brand} differs from competitors' common value ({pivot[col]} vs {mode_val})")

    # Weakness columns (inverted logic)
    for col in weaknesses_cols:
        if col not in df.columns:
            continue
        comp_numeric = pd.to_numeric(competitors[col], errors="coerce").dropna()
        if not comp_numeric.empty:
            comp_mean = comp_numeric.mean()
            try:
                pivot_val = float(pd.to_numeric(pivot[col], errors="coerce"))
                if pivot_val < comp_mean:
                    results["Weaknesses"].append(f"Higher weakness in {col}: {pivot_val:.2f} vs avg {comp_mean:.2f}")
                else:
                    results["Strengths"].append(f"Better performance in {col}: {pivot_val:.2f} vs avg {comp_mean:.2f}")
            except Exception:
                continue
        else:
            # categorical handling (e.g., price_range)
            comp_mode = competitors[col].mode(dropna=True)
            if not comp_mode.empty:
                mode_val = comp_mode.iloc[0]
                if pivot[col] == mode_val:
                    results["Strengths"].append(f"{col}: {pivot_brand} matches competitors' common value ({pivot[col]})")
                else:
                    results["Weaknesses"].append(f"{col}: {pivot_brand} differs from competitors' common value ({pivot[col]} vs {mode_val})")

    # Opportunities (external market)
    for col in opportunities_cols:
        if col not in df.columns:
            continue
        comp_numeric = pd.to_numeric(competitors[col], errors="coerce").dropna()
        if not comp_numeric.empty:
            comp_mean = comp_numeric.mean()
            try:
                pivot_val = float(pd.to_numeric(pivot[col], errors="coerce"))
                if pivot_val > comp_mean:
                    results["Opportunities"].append(f"Market potential in {col}: {pivot_val:.2f} vs avg {comp_mean:.2f}")
            except Exception:
                continue
        else:
            # for categorical market signals, mark opportunity when pivot value is unique or less common
            comp_counts = competitors[col].value_counts(dropna=True)
            pivot_count = comp_counts.get(pivot[col], 0)
            if pivot_count == 0:
                results["Opportunities"].append(f"Market niche in {col}: {pivot_brand} value '{pivot[col]}' not common among competitors")

    # Threats (external risk) — compare pivot to each competitor and list competing brands that outperform pivot
    for col in threats_cols:
        if col not in df.columns:
            continue

        # try numeric comparison first: find competitors with values greater than pivot
        comp_series = pd.to_numeric(competitors[col], errors="coerce")
        try:
            pivot_val = float(pd.to_numeric(pivot[col], errors="coerce"))
        except Exception:
            # pivot value not numeric (or missing) — skip numeric threat checks for this column
            pivot_val = None

        if pivot_val is not None and not pd.isna(pivot_val):
            # find all competitors that outperform pivot in this metric
            outperformers = competitors.loc[comp_series > pivot_val, ["brand_name", col]].copy()
            if not outperformers.empty:
                for _, row in outperformers.iterrows():
                    try: 
                        comp_val = float(pd.to_numeric(row[col], errors="coerce"))
                        results["Threats"].append(
                            f"Competitor '{row['brand_name']}' outperforms {pivot_brand} in {col}: "
                            f"{comp_val:.2f} vs pivot {pivot_val:.2f}"
                        )
                    except Exception:
                        # fallback to string representation if conversion fails
                        results["Threats"].append(
                            f"Competitor '{row['brand_name']}' outperforms {pivot_brand} in {col}: "
                            f"{row[col]} vs pivot {pivot_val}"
                        )
            # if nobody outperforms pivot, optional note (commented out)
            else:
                results["Threats"].append(f"No competitors currently outperform {pivot_brand} in {col}.")
        else:
            # Non-numeric / categorical threat handling
            comp_mode = competitors[col].mode(dropna=True)
            if not comp_mode.empty and comp_mode.iloc[0] != pivot[col]:
                results["Threats"].append(
                    f"Competitors commonly have {col}='{comp_mode.iloc[0]}' while {pivot_brand} has '{pivot[col]}'"
                )

    return results

if __name__ == "__main__":
    print("Running SWOT analysis (internal fetch)...")

    results = swot_analysis()

    print("\n===== SWOT RESULTS =====")
    if not results:
        print("No SWOT results found.")
    else:
        for section, points in results.items():
            print(f"\n{section}:")
            for p in points:
                print(f" - {p}")

