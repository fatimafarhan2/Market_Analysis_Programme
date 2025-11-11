# streamlit ui for top 5 brands by revenue
import streamlit as st
import pandas as pd
import os
from supabase import create_client
from dotenv import load_dotenv
from google import genai

# --- Config ---
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Functions ---
def fetch_top5_by_revenue(limit: int = 5):
    resp = supabase.table("product_flat_table_h") \
                   .select("*") \
                   .order("total_revenue_usd", desc=True) \
                   .limit(limit) \
                   .execute()
    return getattr(resp, "data", None) or resp.get("data", [])


def normalize_and_validate(rows):
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    num_cols = [c for c in df.columns if "usd" in c or "sold" in c or "rating" in c or "score" in c]
    for n in num_cols:
        df[n] = pd.to_numeric(df[n], errors="coerce")
    df.fillna("", inplace=True)
    return df


def call_gemini_summary(json_data):
    if not GOOGLE_API_KEY:
        return "⚠️ Gemini API key not found. Add GOOGLE_API_KEY in your .env file."
    client = genai.Client(api_key=GOOGLE_API_KEY)
    prompt = f"""
    You are a data analyst. Explain the following JSON listing top 5 brands, focusing on  ratings,
    and performance comparisons. Be concise and factual.

    JSON:
    {json_data}
    """
    response = client.models.generate_content(model="gemini-2.0-flash", contents=[prompt])
    return response.text


# --- Streamlit UI ---
st.set_page_config(page_title="Top 5 Brands by Revenue", layout="wide")

st.title("🏆 Top 5 Brands by Revenue")
st.markdown("This dashboard shows the top 5 brands ranked by **total revenue (USD)**, fetched live from Supabase.")

# Spacing before action button
st.write("")

# Fetch data button
if st.button("Fetch Top 5 Brands"):
    with st.spinner("Fetching data from Supabase..."):
        rows = fetch_top5_by_revenue()
        df = normalize_and_validate(rows)

    if not df.empty:
        # --- Layout section: two clean columns with padding ---
        col1, col2 = st.columns([2, 1], gap="large")

        with col1:
            st.subheader("📊 Data Overview")
            st.dataframe(df[["brand_name", "product_name", "country_name", "total_revenue_usd", "avg_rating"]].style.format({"total_revenue_usd": "${:,.2f}"}))

        with col2:
            st.subheader("💡 Gemini Summary")
            json_result = df.to_dict(orient="records")
            summary = call_gemini_summary(json_result)
            st.write(summary)

        # Chart at bottom
        st.markdown("---")
        st.subheader("📈 Revenue Comparison")
        st.bar_chart(df.set_index("brand_name")["total_revenue_usd"])

    else:
        st.error("No data returned from Supabase. Please check your database connection.")

else:
    st.info("Click the button above to fetch and visualize top 5 brands by revenue.")
