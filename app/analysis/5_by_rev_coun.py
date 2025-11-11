# sorted by product and country and finally by revenue descending
"""
top5_brands_by_revenue.py

Fetch top 20 brands from product_flat_table by total_revenue_usd,
filtered to a specific country (provided by user).
"""

import os
import json
from dotenv import load_dotenv
from supabase import create_client
import pandas as pd
from typing import List, Dict, Any, Optional
from google import genai

# --- Config / load env ---
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
api_key = os.getenv("GOOGLE_API_KEY")

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


def fetch_top5_brands_by_revenue(
    product: str,
    limit: int = 20,
    country: Optional[str] = None
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
    query = query.order("total_revenue_usd", desc=True).limit(limit)
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


def pretty_print_top5(df: pd.DataFrame, country: Optional[str]):
    """Print the top 20 results neatly."""
    if df.empty:
        print(f"No rows found for {country}.")
        return

    display_cols = [
        "brand_name","product_name","country_name","currency",
        "total_units_sold","total_revenue_usd"
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    df_display = df[display_cols].copy()

    df_display["total_revenue_usd"] = df_display["total_revenue_usd"].map(lambda v: f"{v:,.2f}")
    print(f"\nTop {len(df_display)} brands in {country} by total_revenue_usd:\n")
    print(df_display.to_string(index=False))


def rows_to_json_safe_all_columns(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert DataFrame rows to a JSON-safe list."""
    if df.empty:
        return []
    return json.loads(df.to_json(orient="records"))


def call_gemini_list_top5(json_result):
    """Optional Gemini summary."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "(Skipping Gemini summary: GOOGLE_API_KEY not set.)"

    client = genai.Client(api_key=api_key)
    prompt = f"""
    List the top brands from this JSON clearly with their revenue and key highlights.

    JSON:
    {json_result}
    """
    response = client.models.generate_content(model="gemini-2.0-flash", contents=[prompt])
    return response.text

import pandas as pd

def swot_analysis(df, pivot_brand):
    # Separate pivot and competitors
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
                    results["Strengths"].append(f"{col}: pivot matches competitors' common value ({pivot[col]})")
                else:
                    results["Weaknesses"].append(f"{col}: pivot differs from competitors' common value ({pivot[col]} vs {mode_val})")

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
                    results["Strengths"].append(f"{col}: pivot matches competitors' common value ({pivot[col]})")
                else:
                    results["Weaknesses"].append(f"{col}: pivot differs from competitors' common value ({pivot[col]} vs {mode_val})")

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
                results["Opportunities"].append(f"Market niche in {col}: pivot value '{pivot[col]}' not common among competitors")

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
                            f"Competitor '{row['brand_name']}' outperforms pivot in {col}: "
                            f"{comp_val:.2f} vs pivot {pivot_val:.2f}"
                        )
                    except Exception:
                        # fallback to string representation if conversion fails
                        results["Threats"].append(
                            f"Competitor '{row['brand_name']}' outperforms pivot in {col}: "
                            f"{row[col]} vs pivot {pivot_val}"
                        )
            # if nobody outperforms pivot, optional note (commented out)
            # else:
            #     results["Threats"].append(f"No competitors currently outperform pivot in {col}.")
        else:
            # Non-numeric / categorical threat handling
            comp_mode = competitors[col].mode(dropna=True)
            if not comp_mode.empty and comp_mode.iloc[0] != pivot[col]:
                results["Threats"].append(
                    f"Competitors commonly have {col}='{comp_mode.iloc[0]}' while pivot has '{pivot[col]}'"
                )

    return results


def main():
    # 🧩 Placeholder for user input — replace this with dynamic input from your app or LLM
    user_country_input = "india"   # <--- CHANGE THIS dynamically later (e.g., from user query)

    raw_rows = fetch_top5_brands_by_revenue(product="lipstick", country=user_country_input)
    df = normalize_and_validate(raw_rows)
    pretty_print_top5(df, country=user_country_input)

    json_result = rows_to_json_safe_all_columns(df)

    print("\n df going for SWOT...")
    swot_analysis_results = swot_analysis(df, pivot_brand=df.iloc[0]["brand_name"])
    # print(type(swot_analysis_results))
    print("\nSWOT Analysis Results:")   
    print(json.dumps(swot_analysis_results, indent=2))
    # summary = call_gemini_list_top5(json_result)
    # print(summary)

    return json_result

if __name__ == "__main__":
    top5 = main()
    print("\n Program ended.")
    