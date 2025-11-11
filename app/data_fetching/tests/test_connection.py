from app.data_fetching.services.db_connect import supabase
try:
    response = supabase.table("Brands").select("*").limit(1).execute()
    print("✅ Connection successful!")
    print("Fetched sample rows:", response.data)
except Exception as e:
    print("❌ Connection failed:", e)
