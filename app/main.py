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