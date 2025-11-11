# app/data_fetching/services/db_connect.py

from supabase import create_client, Client
from dotenv import load_dotenv
import os

def get_supabase_client() -> Client:
    """
    Initializes and returns a Supabase client using credentials from .env.
    """
    load_dotenv()
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        raise ValueError("Supabase credentials not found in environment variables.")
    
    return create_client(supabase_url, supabase_key)


supabase = get_supabase_client()