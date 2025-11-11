# app/data_fetching/agents/validator_agent.py
from app.data_fetching.services.db_connect import supabase

def validate_sql_query(sql: str) -> tuple[bool, str, bool]:
    """
    Validate an SQL query by running a limited test execution via Supabase.
    Returns: (is_valid, message, user_correctable)
    """
    try:
        response = supabase.postgrest.rpc("execute_sql", {"query": sql}).execute()

        # Handle SQL or DB errors
        if hasattr(response, "error") and response.error:
            msg = response.error.get("message", str(response.error))
            return False, f"SQL execution error: {msg}", False

        data = getattr(response, "data", None)

        # Query ran but returned no data → user issue
        if not data:
            return False, "No matching records found. Please refine your query or filters.", True

        # Query successful
        return True, "SQL validated and executed successfully.", False

    except Exception as e:
        return False, f"Validation failed: {str(e)}", False


def safe_execute_sql(sql: str) -> dict:
    """Safely execute the SQL query via Supabase."""
    try:
        response = supabase.postgrest.rpc("execute_sql", {"query": sql}).execute()

        if hasattr(response, "error") and response.error:
            msg = response.error.get("message", str(response.error))
            return {"success": False, "error": msg}

        return {"success": True, "data": getattr(response, "data", None)}

    except Exception as e:
        return {"success": False, "error": f"Execution failed: {str(e)}"}

# validator_agent.py

def validate_sql_and_execute(sql: str) -> dict:
    """Combined validation and execution wrapper for pipeline with safe error handling."""
    try:
        response = supabase.postgrest.rpc("execute_sql", {"query": sql}).execute()
    except Exception as e:
        return {"error": f"Execution failed: {str(e)}", "user_correctable": False}

    # Check if response has proper error
    if hasattr(response, "error") and response.error:
        msg = response.error.get("message") if isinstance(response.error, dict) else str(response.error)
        return {"error": f"SQL execution error: {msg}", "user_correctable": False}

    # Check if response has data
    data = getattr(response, "data", None)

    if not isinstance(data, list):
        # Likely server returned HTML / invalid response
        return {"error": "Server returned invalid response. Please try again later.", "user_correctable": False}

    if not data:
        # Valid query but no results
        return {"error": "No matching records found. Please refine your query or filters.", "user_correctable": True}

    return {"data": data}
