"""
Simple test script to run the SWOT analysis directly without LangChain/LangGraph
"""
import json
import pandas as pd
from swot_tool_1 import swot_tool

sample_df = pd.DataFrame(
        [
            {"product_id": 1, "product_name": "A", "brand_id": 1, "brand_name": "A", "category": "lipstick",
             "avg_revenue_usd": 22.5, "g_size": 12.0, "avg_rating": 5.0, "avg_sentiment_score": 5.2,
             "avg_returns_rate": 4.5, "avg_marketing_spend_usd": 40000, "country_id": "C001"},
            {"product_id": 2, "product_name": "B", "brand_id": 2, "brand_name": "B", "category": "lipstick",
             "avg_revenue_usd": 24.0, "g_size": 11.0, "avg_rating": 6.5, "avg_sentiment_score": 6.2,
             "avg_returns_rate": 7.0, "avg_marketing_spend_usd": 60000, "country_id": "C001"},
            {"product_id": 1, "product_name": "C", "brand_id": 32, "brand_name": "Z", "category": "lipstick",
             "avg_revenue_usd": 25, "g_size": 14.0, "avg_rating": 7.0, "avg_sentiment_score": 7.2,
             "avg_returns_rate": 3.5, "avg_marketing_spend_usd": 45000, "country_id": "C001"},
            {"product_id": 2, "product_name": "D", "brand_id": 21, "brand_name": "X", "category": "lipstick",
             "avg_revenue_usd": 23.0, "g_size": 11.0, "avg_rating": 4.5, "avg_sentiment_score": 8.2,
             "avg_returns_rate": 6.0, "avg_marketing_spend_usd": 50000, "country_id": "C001"},
            {"product_id": 1, "product_name": "E", "brand_id": 11, "brand_name": "Y", "category": "lipstick",
             "avg_revenue_usd": 20.0, "g_size": 13.0, "avg_rating": 4.0, "avg_sentiment_score": 3.2,
             "avg_returns_rate": 8.5, "avg_marketing_spend_usd": 35000, "country_id": "C001"},
        ]
)

# Test query
user_prompt = "i want to launch 10g lipstick for $20, its ingredients include beeswax, castor oil,formaldehyde"
df_json = sample_df.to_dict(orient="records")
json_result = {"country_id": "C001"}

# Run the SWOT analysis using the tool's invoke method
output = swot_tool.invoke({
    "user_query": user_prompt,
    "df": df_json,
    "json_result": json_result
})

# Print the result
print(json.dumps(output, indent=2))