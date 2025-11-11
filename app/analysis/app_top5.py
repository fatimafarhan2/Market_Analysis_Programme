# app_top5_brands.py
"""
Streamlit UI for top5_brands_by_revenue.

Place your SUPABASE_URL, SUPABASE_KEY and GOOGLE_API_KEY in a .env or environment variables.
Run: streamlit run app_top5_brands.py
"""

import os
import json
from dotenv import load_dotenv
from supabase import create_client
import pandas as pd
import streamlit as st
from typing import List, Dict, Any
from google import genai

# --- Config / load env ---
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# optional LLM key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    # We'll surface an error inside the app rather than raising immediately so the UI can show instructions
    pass
else:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Columns to fetch (same as your script)
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

# ---------- Data fetching / utils ----------
@st.cache_data(ttl=60)  # cache for a minute by default; adjusts as needed
def fetch_top_by_revenue(limit: int = 5) -> List[Dict[str, Any]]:
    """Fetch top brands by total_revenue_usd from Supabase."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError("SUPABASE_URL and SUPABASE_KEY not set. Put them in a .env or environment variables.")
    # run query
    resp = supabase.table("product_flat_table_h") \
                   .select(COLUMNS) \
                   .order("total_revenue_usd", desc=True) \
                   .limit(limit) \
                   .execute()

    data = None
    if isinstance(resp, dict):
        data = resp.get("data")
        error = resp.get("error")
        if error:
            raise RuntimeError(f"Supabase error: {error}")
    else:
        data = getattr(resp, "data", None)
        status = getattr(resp, "status_code", None)

    if data is None:
        raise RuntimeError(f"Failed to fetch data from Supabase. Raw response: {resp}")

    return data

def normalize_and_validate(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Same cleaning logic as original script."""
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    for col in ["product_id","product_name","brand_id","brand_name","country_name","currency"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

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

    if "total_revenue_usd" in df.columns:
        df["total_revenue_usd"] = df["total_revenue_usd"].fillna(0.0)
    if "total_units_sold" in df.columns:
        df["total_units_sold"] = df["total_units_sold"].fillna(0).astype(int)

    return df

def rows_to_json_safe_all_columns(df: pd.DataFrame) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if df is None or df.empty:
        return out

    for _, row in df.iterrows():
        raw = row.to_dict()
        safe_row: Dict[str, Any] = {}
        for col, val in raw.items():
            if pd.isna(val):
                safe_row[col] = None
                continue
            if isinstance(val, (pd.Timestamp,)):
                safe_row[col] = val.isoformat()
            elif isinstance(val, (int,)) and not isinstance(val, bool):
                safe_row[col] = int(val)
            elif isinstance(val, float):
                safe_row[col] = float(round(val, 6))
            elif isinstance(val, (list, dict)):
                safe_row[col] = val
            else:
                safe_row[col] = str(val)
        out.append(safe_row)

    return out

def call_gemini_list_top5(json_result: List[Dict[str, Any]]) -> str:
    """Call Gemini/GenAI to produce an English listing. Requires GOOGLE_API_KEY."""
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY not set. Put it in .env or environment variables to use LLM summary.")
    client = genai.Client(api_key=GOOGLE_API_KEY)

    prompt = f"""
You are a data explainer. Given this JSON of top brands and their product details,
write a natural, human-readable paragraph listing each brand, its revenue, and key highlights. Relate columns
to each other where relevant (e.g., revenue vs units sold, rating vs sentiment). Do NOT analyze heavily — just list clearly.

JSON data:
{json.dumps(json_result, indent=2)}
"""
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt]
    )
    # response.text contains the LLM output in many genai versions
    return getattr(response, "text", str(response))

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Top Brands by Revenue", layout="wide", initial_sidebar_state="expanded")

st.title("Top Brands by Revenue — Explorer")
st.markdown("Fetches top brands from `product_flat_table_h` and visualizes revenue, units, and other details.")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    limit = st.number_input("How many rows to fetch", min_value=1, max_value=20, value=5, step=1)
    refresh = st.button("Refresh data (clear cache)")
    show_llm = st.checkbox("Generate human-readable LLM listing (requires GOOGLE_API_KEY)", value=False)
    show_json_raw = st.checkbox("Show raw JSON result", value=False)
    download_format = st.selectbox("Download format", ["csv", "json"], index=0)
    # st.markdown("---")
    # st.markdown("⚙️ Make sure SUPABASE_* keys are set in environment or .env.")

if refresh:
    # Clear cache by calling the cache clear function or reloading module-level cache
    try:
        fetch_top_by_revenue.clear()
    except Exception:
        # fallback: will naturally re-fetch because cached function key changed on app reload
        pass
    st.experimental_rerun()

# Main area
col1, col2 = st.columns([2, 1], gap = "large")

with col1:
    # st.subheader("Top results")
    st.subheader("📊 Data Overview")
    st.dataframe(df[["brand_name", "product_name", "country_name", "total_revenue_usd", "avg_rating"]].style.format({"total_revenue_usd": "${:,.2f}"}))

    # Fetch data
    try:
        raw_rows = fetch_top_by_revenue(limit=limit)
        df = normalize_and_validate(raw_rows)

        if df.empty:
            st.info("No rows returned from Supabase. Check your table name and keys.")
        else:
            # Format revenue for display copy
            df_display = df.copy()
            if "total_revenue_usd" in df_display.columns:
                df_display["total_revenue_usd_display"] = df_display["total_revenue_usd"].map(lambda v: f"${v:,.2f}")
                # Use the display column for showing but keep numeric one too
                display_cols = ["brand_id","brand_name","product_id","product_name","country_name","total_units_sold","total_revenue_usd_display"]
            else:
                display_cols = list(df_display.columns[:8])

            st.dataframe(df_display[display_cols].rename(columns={"total_revenue_usd_display": "total_revenue_usd"}))

            # # Interactive selection
            # selected_idx = st.selectbox("Select a brand row to inspect", options=list(df.index), format_func=lambda i: f"{df.loc[i,'brand_name']} — {df.loc[i,'country_name'] if 'country_name' in df.columns else ''}")
            # selected_row = df.loc[selected_idx]

            # with st.expander("Selected row details (all available columns)"):
            #     st.json(rows_to_json_safe_all_columns(pd.DataFrame([selected_row]))[0])

            # Simple chart: revenue vs units if both exist
            if "total_revenue_usd" in df.columns and "total_units_sold" in df.columns:
                st.subheader("Revenue vs Units sold (top rows)")
                chart_df = df.set_index("brand_name")[["total_revenue_usd","total_units_sold"]].sort_values("total_revenue_usd", ascending=False)
                # show revenue bar chart and units as line on same chart using altair
                try:
                    import altair as alt
                    chart_df_reset = chart_df.reset_index().melt(id_vars=["brand_name"], value_vars=["total_revenue_usd","total_units_sold"], var_name="metric", value_name="value")
                    chart = alt.Chart(chart_df_reset).mark_bar().encode(
                        x=alt.X("brand_name:N", sort=chart_df.index.tolist(), title="Brand"),
                        y=alt.Y("value:Q"),
                        color="metric:N",
                        column=alt.Column("metric:N", header=alt.Header(title=None))
                    )
                    st.altair_chart(chart.interactive(), use_container_width=True)
                except Exception:
                    # fallback to pandas bar_chart
                    st.bar_chart(chart_df["total_revenue_usd"])

            # Download buttons
            json_result = rows_to_json_safe_all_columns(df)
            if download_format == "csv":
                csv_bytes = pd.DataFrame(json_result).to_csv(index=False).encode("utf-8")
                st.download_button(label="Download CSV", data=csv_bytes, file_name="top_brands.csv", mime="text/csv")
            else:
                json_bytes = json.dumps(json_result, indent=2).encode("utf-8")
                st.download_button(label="Download JSON", data=json_bytes, file_name="top_brands.json", mime="application/json")

    except Exception as e:
        st.error(f"Error fetching data: {e}")
        st.stop()

with col2:
    st.subheader("Overview & Metrics")
    if not df.empty:
        # show a few metrics
        total_rev = float(df["total_revenue_usd"].sum()) if "total_revenue_usd" in df.columns else 0.0
        avg_rev = float(df["total_revenue_usd"].mean()) if "total_revenue_usd" in df.columns else 0.0
        total_units = int(df["total_units_sold"].sum()) if "total_units_sold" in df.columns else 0

        st.metric("Total revenue (sum)", f"${total_rev:,.2f}")
        st.metric("Average revenue (per row)", f"${avg_rev:,.2f}")
        st.metric("Total units (sum)", f"{total_units:,}")

        st.markdown("**Top countries in this set**")
        if "country_name" in df.columns:
            st.table(df["country_name"].value_counts().rename_axis("country").reset_index(name="count").head(10))

    st.markdown("---")
    st.subheader("LLM Summary (optional)")
    if show_llm:
        try:
            listing_text = call_gemini_list_top5(json_result)
            st.markdown("**LLM listing:**")
            st.write(listing_text)
        except Exception as e:
            st.error(f"LLM call failed: {e}")
            st.info("Set GOOGLE_API_KEY and ensure network access to use LLM summarization.")

# Optional raw JSON view
if show_json_raw:
    st.subheader("Raw JSON")
    st.json(json_result)

st.markdown("---")
st.caption("This UI fetches the top rows from the Supabase table `product_flat_table_h`. Adjust the limit in the sidebar. LLM summary requires GOOGLE_API_KEY and will fail gracefully if missing.")
