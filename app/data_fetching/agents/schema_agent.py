from app.data_fetching.services.db_connect import supabase
from functools import lru_cache

@lru_cache
def get_schema_summary():
    """
    Fetches and caches schema summary from Supabase RPC function.
    """
    response = supabase.rpc("get_schema_summary").execute()
    if not response.data:
        return "No schema Data Returned"

    schema_text = "\n".join([
        f"{r['table_name']}: {r['column_name']} ({r['data_type']})"
        for r in response.data
    ])
    return schema_text


@lru_cache
def get_flat_table_schema():
    """
    Fetch and cache the chema structure for the flat tabes in supabase rpc function
    """
    response=supabase.rpc("get_flat_schema_summary").execute()
    if not response.data:
        return "No schema returned"
    
    schema_text="\n".join([
        f"{r['table_name']}: {r['column_name']} ({r['data_type']})"
        for r in response.data
    ])
    return schema_text