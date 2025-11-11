# app/shared/market_analysis_tool/market_tool.py
from langchain_core.tools import tool
import re
import json
from typing import Dict, Any, List, Optional
from db_connect import supabase      # if filename is db_connect.py
# from app.data_fetching.services.db_connect import supabase
import numpy as np
import pandas as pd
from scipy import stats
from typing import Annotated

def _clean_sql_for_rpc(sql: str) -> str:
    """Strip comments and trailing semicolons for Supabase run_sql RPC."""
    if not sql:
        return sql
    # remove SQL comments -- and /* */
    clean_sql = re.sub(r'(--.*?$)|(/\*.*?\*/)', '', sql, flags=re.MULTILINE | re.DOTALL).strip()
    # remove trailing semicolon(s)
    while clean_sql.endswith(";"):
        clean_sql = clean_sql[:-1].strip()
    return clean_sql

def _run_sql(supabase_client, sql: str) -> List[Dict[str, Any]]:
    """
    Execute SQL via Supabase RPC `run_sql`.
    Returns list[dict] (rows). Always returns [] rather than None.
    """
    sql_clean = _clean_sql_for_rpc(sql)
    # Debug: print(sql_clean)  # enable if needed
    try:
        res = supabase_client.rpc("run_sql", {"sql": sql_clean}).execute()
    except Exception as e:
        print("RPC call failed:", e)
        return []
    data = getattr(res, "data", None)
    # Some supabase clients return nested mapping; handle common shapes
    if data is None:
        return []
    # If data is a JSON string inside a list, attempt to parse
    if isinstance(data, str):
        try:
            parsed = json.loads(data)
            return parsed if isinstance(parsed, list) else [parsed]
        except Exception:
            return []
    # If data is a list of rows, return
    return data

# -----------------------
# SQL to load the flat table (no filters)
# -----------------------
_PRODUCT_FLAT_SELECT = """
SELECT
    product_id,
    product_name,
    brand_id,
    brand_name,
    category,
    key_function,
    price_range,
    country_id,
    country_name,
    region,
    currency,
    total_units_sold,
    total_revenue_usd,
    avg_revenue_usd,
    total_net_units_sold,
    total_net_revenue_usd,
    avg_returns_rate,
    avg_marketing_spend_usd,
    avg_marketing_efficiency_ratio,
    avg_return_on_marketing_spend,
    avg_online_sales_ratio,
    avg_rating,
    avg_sentiment_score,
    total_num_reviews,
    avg_population_millions,
    avg_income_usd,
    avg_urbanization_rate,
    avg_online_shopping_penetration,
    top_common_keywords,
    dominant_distribution_channel
FROM product_flat_table
"""

# -----------------------
# Dataframe helpers
# -----------------------
def _to_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)

    numeric_cols = [
        "total_units_sold", "total_revenue_usd", "avg_revenue_usd", "total_net_units_sold",
        "total_net_revenue_usd", "avg_returns_rate", "avg_marketing_spend_usd",
        "avg_marketing_efficiency_ratio", "avg_return_on_marketing_spend",
        "avg_online_sales_ratio", "avg_rating", "avg_sentiment_score",
        "total_num_reviews", "avg_population_millions", "avg_income_usd",
        "avg_urbanization_rate", "avg_online_shopping_penetration"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Normalize text columns
    for c in ["country_name", "region", "brand_name", "product_name", "category"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df

# -----------------------
# KPI / statistical helpers
# -----------------------
def _herfindahl_index(series: pd.Series, value_col: str = "total_revenue_usd") -> float:
    """
    Compute HHI (market concentration) on group by series (string labels).
    Expects a DataFrame with grouped sums or a series mapping label->value.
    We'll accept either a DataFrame (with columns [label, value_col]) or a Series.
    """
    if isinstance(series, pd.DataFrame):
        s = series[value_col].fillna(0).astype(float)
    elif isinstance(series, pd.Series):
        s = series.fillna(0).astype(float)
    else:
        return 0.0
    total = s.sum()
    if total <= 0:
        return 0.0
    shares = (s / total) ** 2
    return float((shares.sum()) * 10000)  # HHI scaled (0..10000)

def _market_shares(df: pd.DataFrame, group_by: List[str], value_col: str = "total_revenue_usd"):
    """
    Returns DataFrame with group cols and share and cumulative share.
    """
    grp = df.groupby(group_by)[value_col].sum().reset_index().sort_values(value_col, ascending=False)
    total = grp[value_col].sum()
    if total == 0:
        grp["share"] = 0.0
        grp["cum_share"] = 0.0
    else:
        grp["share"] = grp[value_col] / total
        grp["cum_share"] = grp["share"].cumsum()
    return grp



# -----------------------
# Core deep-analysis function
# -----------------------
@tool
def deep_market_analysis() -> Dict[str, Any]:
    """
    Perform in-depth analysis on product_flat_table (assumes data is already the 'slice' you want).
    - supabase_client: if None, uses imported `supabase` from db_connect
    - run_regressions: whether to attempt regression models
    - per_scope_limit: number of groups shown in top lists
    Returns a JSON-serializable dict of results.
    """
    per_scope_limit = 5
    sup = supabase
    results: Dict[str, Any] = {"notes": "Deep market analysis", "errors": []}

    print("Running deep_market_analysis...")
    rows = _run_sql(sup, _PRODUCT_FLAT_SELECT)
    df = _to_df(rows)
    results["meta_rows_fetched"] = len(df)

    if df.empty:
        results["errors"].append("No rows returned from product_flat_table.")
        return results

   
    total_revenue = float(df["total_revenue_usd"].sum(skipna=True))
    total_units = int(df["total_units_sold"].sum(skipna=True))
    total_brands = int(df["brand_name"].nunique())
    total_products = int(df["product_name"].nunique())
    avg_rating = float(df["avg_rating"].mean(skipna=True))
    avg_sentiment = float(df["avg_sentiment_score"].mean(skipna=True))
    avg_roms = float(df["avg_return_on_marketing_spend"].mean(skipna=True))
    avg_eff = float(df["avg_marketing_efficiency_ratio"].mean(skipna=True))

    results["global_summary"] = {
        "total_revenue_usd": round(total_revenue, 2),
        "total_units_sold": total_units,
        "total_brands": total_brands,
        "total_products": total_products,
        "avg_rating": round(avg_rating, 3),
        "avg_sentiment_score": round(avg_sentiment, 3),
        "avg_roms": round(avg_roms, 3),
        "avg_marketing_efficiency_ratio": round(avg_eff, 3)
    }

    
    region_grp = df.groupby("region").agg(
        total_revenue_usd=("total_revenue_usd", "sum"),
        total_units_sold=("total_units_sold", "sum"),
        avg_rating=("avg_rating", "mean"),
        avg_sentiment=("avg_sentiment_score", "mean"),
        avg_roms=("avg_return_on_marketing_spend", "mean"),
        avg_efficiency=("avg_marketing_efficiency_ratio", "mean"),
        active_brands=("brand_name", lambda s: s.nunique()),
        active_products=("product_name", lambda s: s.nunique())
    ).reset_index().sort_values("total_revenue_usd", ascending=False)

    results["regions"] = region_grp.to_dict(orient="records")

    country_grp = df.groupby(["region", "country_name"]).agg(
        total_revenue_usd=("total_revenue_usd", "sum"),
        total_units_sold=("total_units_sold", "sum"),
        avg_rating=("avg_rating", "mean"),
        avg_sentiment=("avg_sentiment_score", "mean"),
        avg_roms=("avg_return_on_marketing_spend", "mean"),
        avg_efficiency=("avg_marketing_efficiency_ratio", "mean"),
        active_brands=("brand_name", lambda s: s.nunique()),
        active_products=("product_name", lambda s: s.nunique())
    ).reset_index().sort_values("total_revenue_usd", ascending=False)

    results["countries"] = country_grp.to_dict(orient="records")

    
    brand_grp = df.groupby(["brand_name", "region", "country_name"]).agg(
        total_revenue_usd=("total_revenue_usd", "sum"),
        total_units_sold=("total_units_sold", "sum"),
        total_products=("product_name", "nunique"),
        avg_rating=("avg_rating", "mean"),
        avg_sentiment=("avg_sentiment_score", "mean"),
        avg_roms=("avg_return_on_marketing_spend", "mean"),
        avg_efficiency=("avg_marketing_efficiency_ratio", "mean")
    ).reset_index().sort_values("total_revenue_usd", ascending=False)

    product_grp = df.groupby(["product_name", "brand_name", "country_name", "region"]).agg(
        total_revenue_usd=("total_revenue_usd", "sum"),
        total_units_sold=("total_units_sold", "sum"),
        avg_rating=("avg_rating", "mean"),
        avg_sentiment=("avg_sentiment_score", "mean"),
        avg_roms=("avg_return_on_marketing_spend", "mean"),
        avg_efficiency=("avg_marketing_efficiency_ratio", "mean")
    ).reset_index().sort_values("total_revenue_usd", ascending=False)

    results["brand_summary_topN"] = brand_grp.head(per_scope_limit).to_dict(orient="records")
    results["product_summary_topN"] = product_grp.head(per_scope_limit).to_dict(orient="records")

    # -----------------------
    # Compute market shares using the existing _market_shares helper
    # -----------------------
    brand_market_share = _market_shares(df, ["brand_name"], "total_revenue_usd")
    results["brand_market_share_topN"] = brand_market_share.head(per_scope_limit).to_dict(orient="records")

    country_market_share = _market_shares(df, ["country_name"], "total_revenue_usd")
    results["country_market_share_topN"] = country_market_share.head(per_scope_limit).to_dict(orient="records")



    def _top_bottom(df_, val_col="total_revenue_usd", n=10):
        srt = df_.sort_values(val_col, ascending=False)
        top = srt.head(n).to_dict(orient="records")
        bottom = srt.tail(n).sort_values(val_col, ascending=True).to_dict(orient="records")
        return {"top": top, "bottom": bottom}

    results["global_brand_top_bottom"] = _top_bottom(brand_grp, "total_revenue_usd", per_scope_limit)
    results["global_product_top_bottom"] = _top_bottom(product_grp, "total_revenue_usd", per_scope_limit)

   
    brand_revenue = brand_grp.groupby("brand_name")["total_revenue_usd"].sum().reset_index()
    results["hhi_by_brand"] = _herfindahl_index(brand_revenue, "total_revenue_usd")

    country_revenue = country_grp.groupby("country_name")["total_revenue_usd"].sum().reset_index()
    results["hhi_by_country"] = _herfindahl_index(country_revenue, "total_revenue_usd")

   
    corr_cols = [
        "total_revenue_usd", "total_units_sold", "avg_marketing_spend_usd",
        "avg_marketing_efficiency_ratio", "avg_return_on_marketing_spend",
        "avg_online_sales_ratio", "avg_rating", "avg_sentiment_score"
    ]
    corr_df = df[corr_cols].dropna()
    if not corr_df.empty and corr_df.shape[0] > 1:
        corr_mat = corr_df.corr(method="pearson")
        results["correlation_matrix"] = corr_mat.round(3).to_dict()
    else:
        results["correlation_matrix"] = {}
    # -----------------------
    # Strategy insights
    # -----------------------
    strategy_insights = []

   
    if not brand_grp.empty:
        median_rev = brand_grp["total_revenue_usd"].median()
        for _, row in brand_grp.iterrows():
            sentiment = row.get("avg_sentiment", 0)
            revenue = row.get("total_revenue_usd", 0)
            if sentiment and revenue and sentiment > 0.6 and revenue < median_rev:
                strategy_insights.append({
                    "type": "opportunity",
                    "text": f"Brand {row['brand_name']} in {row['country_name']} has above-average sentiment ({round(sentiment, 3)}) "
                            f"but revenue ({round(revenue,2)}) below median ({round(median_rev,2)}).",
                    "brand": row["brand_name"],
                    "country": row["country_name"],
                    "sentiment": round(sentiment,3),
                    "revenue": round(revenue,2)
                })

    for _, r in region_grp.iterrows():
        avg_roms = r.get("avg_roms", 0)
        if avg_roms is not None and avg_roms < 1.0:
            strategy_insights.append({
                "type": "inefficient_marketing",
                "text": f"Region {r['region']} has avg ROMS = {round(avg_roms,3)} < 1. Consider reallocating marketing spend.",
                "region": r["region"],
                "avg_roms": round(avg_roms,3)
            })


    hhi_brand = results.get("hhi_by_brand", 0)
    if hhi_brand and hhi_brand > 2500:
        strategy_insights.append({
            "type": "concentration",
            "text": f"High market concentration by brand (HHI={round(hhi_brand,2)}). Top brands dominate revenue."
        })

    results["strategy_insights"] = strategy_insights

    return results

# -----------------------
# CLI-style local run for quick tests
# -----------------------
if __name__ == "__main__":
    # local test (ensure environment variables or db_connect provides supabase)
    print("Running deep_market_analysis as script...")
    out = deep_market_analysis()
    print(json.dumps(out, indent=2, default=lambda x: (round(x, 3) if isinstance(x, float) else str(x))))
