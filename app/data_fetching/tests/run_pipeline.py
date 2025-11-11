# app/data_fetching/tests/run_pipeline.py
from app.test import run_data_fetching

if __name__ == "__main__":
    print("Starting interactive data fetching pipeline...")
    
    # Start pipeline with empty query so input_node will ask
    result = run_data_fetching(user_query="")

    # Handle pipeline result
    if result.get("retry"):
        print("\n", result.get("message", "No data found."))
    elif result.get("error") or result.get("message"):
        print("\n Pipeline error:", result.get("error") or result.get("message"))
    else:
        data = result.get("data")
        print("\nPipeline completed successfully!")
        print(f"Sample output: {str(data)[:250]} ...")

# # app/data_fetching/tests/run_pipeline.py
# from app.data_fetching.services.nodes import run_data_fetching

# if __name__ == "__main__":
#     q = "Analyze sales of shampoo in japan and france"
#     res = run_data_fetching(q)
#     # print("\nFinal Pipeline Output:", res[:240])



# if __name__ == "__main__":
#     q = "Show me the analysis for the night creams in the cosmetic industry in Asia"
#     res = run_data_fetching(q)
#     print("\nFinal Pipeline Output:", res)



# # # app/data_fetching/tests/run_pipeline.py
# # from app.data_fetching.services.nodes import run_data_fetching

# # if __name__ == "__main__":
# #     q = "Show me the average revenue earned by night creams in the cosmetic industry in Asia"


# app/data_fetching/tests/test_pipeline.py

import time
from app.data_fetching.services.nodes import run_data_fetching

# -----------------------------
# Comprehensive Test Inputs
# -----------------------------
# test_queries = [
#     # Category-based / Product Type Queries
#     "Show me all shampoos sold in Europe.",
#     "Get sales data for conditioners and hair serums in Asia.",
#     "List all lipsticks and eyeliners with their revenue in North America.",
#     "Display all skincare products available in Australia.",
#     "What are the top-selling perfumes globally?",
#     "Find sunscreen sales data in the Middle East.",
#     "Show all moisturizers and lotions sold in South America.",

#     # Price, Revenue, and Units Filters
#     "Which products generated more than $10,000 revenue last month?",
#     "List items where average price is below $15.",
#     "Find products with total units sold greater than 500.",
#     "Get data for makeup items with revenue between $1000 and $5000.",
#     "Show all skincare items where net_units_sold < 200.",

#     # Region & Country Specific
#     "List all products sold in Pakistan and India.",
#     "Show all European region sales for perfumes.",
#     "Get products from countries in the Americas.",
#     "Display sales data for products in Southeast Asia.",
#     "Show me the list of products sold in France.",

#     # Combined Filters
#     "Find haircare products in Asia with revenue > $2000.",
#     "Show lipsticks under $20 sold in North America.",
#     "List products with sales above $1000 in Europe and Asia.",
#     "Display skincare items in the Middle East priced over $30.",
#     "Show all perfumes sold in France and Germany with net_units_sold > 100.",

#     # Aggregations and Trends
#     "What are the top 5 selling products globally?",
#     "Which category has the highest average revenue in Asia?",
#     "Give me total revenue by region.",
#     "Show total units sold grouped by product category.",
#     "List average revenue per product for Europe.",

#     # Edge & Ambiguous Queries
#     "Show me the stuff sold last year.",
#     "I want to see product data.",
#     "Which things are popular these days?",
#     "Display all details.",
#     "Get me everything.",
#     "Show sales data for random items.",

#     # Complex Logic
#     "Show products in Asia OR Europe with total revenue above 500000 in 2023.",
#     "List items NOT sold in North America.",
#     "Show products sold in Asia but not in Japan.",
#     "Find items priced between $10 and $30 and sold in at least two regions.",

#     # Data Integrity / Validation Edge Cases
#     "Get products with missing region information.",
#     "Show products where price is null or zero.",
#     "Find duplicates by product name.",
#     "Show products where revenue is negative.",

#     # AI Reasoning Stress Tests
#     "Which product performed the best overall?",
#     "Find the least sold product in 2024.",
#     "Which product category grew the most in Asia?",
#     "Compare skincare vs haircare sales in Europe.",
#     "Which product has the highest net revenue margin?",

#     # Natural Language / Human Style Queries
#     "How did shampoos do in Europe this year?",
#     "Tell me about the performance of lipsticks lately.",
#     "I want data for perfumes across all countries.",
#     "Get me some insight into makeup sales.",
#     "How are skincare products performing globally?",

#     # All-in-One Heavy Query
#     "Show all makeup and skincare products sold in Asia and Europe where total revenue > 5000 and units_sold > 100, grouped by category."
# ]

# # -----------------------------
# # Run Test Pipeline
# # -----------------------------
# results = []

# for idx, query in enumerate(test_queries, 1):
#     print(f"\n=== Running Test {idx} ===")
#     print(f"Query: {query}")
#     start_time = time.time()
#     result = run_data_fetching(query)
#     elapsed = time.time() - start_time
#     status = "SUCCESS" if isinstance(result, list) else "ERROR"
#     print(f"Status: {status}")
#     print(f"Time Taken: {elapsed:.2f} sec")
#     # Print a small sample of result if it's a list
#     if isinstance(result, list):
#         print("Sample Result:", result[:3])
#     else:
#         print("Message:", result)
#     results.append({
#         "query": query,
#         "status": status,
#         "time": elapsed,
#         "result": result if isinstance(result, list) else None
#     })

# print("\n✅ All tests completed.")
