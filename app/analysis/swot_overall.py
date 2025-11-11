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
        "avg_rating", "avg_sentiment_score", "avg_marketing_spend_usd", 
        "country_name", "total_revenue_usd"
    ]

    results = {"Strengths": [], "Weaknesses": [], "Opportunities": [], "Threats": []}

    # Strengths and Weaknesses (internal comparison)
    for col in strengths_cols:
        if pivot[col] > competitors[col].mean():
            results["Strengths"].append(f"Strong {col}: {pivot[col]:.2f} vs avg {competitors[col].mean():.2f}")
        else:
            results["Weaknesses"].append(f"Weak {col}: {pivot[col]:.2f} vs avg {competitors[col].mean():.2f}")

    # Weakness columns (inverted logic)
    for col in weaknesses_cols:
        if pivot[col] < competitors[col].mean():
            results["Weaknesses"].append(f"Higher weakness in {col}: {pivot[col]:.2f}")
        else:
            results["Strengths"].append(f"Better performance in {col}: {pivot[col]:.2f}")

    # Opportunities (external market)
    for col in opportunities_cols:
        if pivot[col] > competitors[col].mean():
            results["Opportunities"].append(f"Market potential in {col}: {pivot[col]:.2f}")

    # Threats (external risk)
    for col in threats_cols:
        if competitors[col].max() > pivot[col]:
            results["Threats"].append(f"Competitor outperforms in {col}: {competitors[col].max():.2f}")

    return results
