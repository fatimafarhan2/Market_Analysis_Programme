from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional, Dict, Any
from app.data_fetching.agents.intent_agent import detect_intent
from app.data_fetching.agents.sql_agent import generate_sql
from app.data_fetching.agents.validator_agent import validate_sql_and_execute
from app.data_fetching.services.db_connect import supabase

# ---------------- PIPELINE STATE ----------------

class PipelineState(TypedDict):
    user_query: Optional[str]
    filters: Optional[Dict[str, Any]]
    sql_query: Optional[str]
    result: Optional[Any]
    error: Optional[str]
    inserted: Optional[bool]
    loop_back: Optional[bool]
    retry_count: Optional[int]

# ---------------- NODES ----------------

def input_node(state: PipelineState, **kwargs):
    """Handles user input."""
    temp_query_holder = kwargs.setdefault("temp_query_holder", {})

    print("\nAwaiting user input...")
    MAX_RETRIES = 5
    
    # Check retry count
    current_retry = temp_query_holder.get("retry_count", 0)
    if current_retry >= MAX_RETRIES:
        print(f"Max retries reached ({MAX_RETRIES}). Exiting pipeline.")
        return {
            **state,
            "error": "Maximum retries reached.",
            "loop_back": False
        }

    # Handle loopback or initial query
    if temp_query_holder.get("loop_back") or temp_query_holder.get("error"):
        temp_query_holder["retry_count"] = current_retry + 1
        print("Previous attempt failed or returned no results.")
        temp_query_holder["user_query"] = input("Please enter a refined query: ")
    elif "user_query" not in temp_query_holder or not temp_query_holder["user_query"]:
        temp_query_holder["user_query"] = input("Please enter your query: ")

    # Reset error and loop_back flags
    temp_query_holder["error"] = None
    temp_query_holder["loop_back"] = False

    print(f"✓ Received query: {temp_query_holder['user_query']}")
    
    # CRITICAL FIX: Copy query to state so other nodes can access it
    return {
        **state,
        "user_query": temp_query_holder["user_query"],
        "error": None,
        "loop_back": False,
        "retry_count": temp_query_holder.get("retry_count", 0)
    }

def intent_node(state: PipelineState, **kwargs):
    temp_query_holder = kwargs.setdefault("temp_query_holder", {})
    
    # Use state's user_query (which was set by input_node)
    user_query = state.get("user_query")
    print(f"[DEBUG] user_query in intent_node: {user_query!r}")
    print("\n Step 1: Detecting intent...")
    
    try:
        intent = detect_intent(user_query)
        print("✓ Intent detected successfully:")
        print(f"   Filters: {intent.get('filters', {})}")
        
        # Store filters in temp_query_holder for reference
        temp_query_holder["filters"] = intent.get("filters", {})
        
        return {
            **state,
            "filters": intent.get("filters", {}),
            "loop_back": False,
            "error": None
        }
    except Exception as e:
        print(f"✗ Intent detection failed: {e}")
        temp_query_holder["error"] = str(e)
        temp_query_holder["loop_back"] = True
        
        return {
            **state,
            "error": str(e),
            "loop_back": True
        }

def sqlgen_node(state: PipelineState, **kwargs):
    temp_query_holder = kwargs.setdefault("temp_query_holder", {})

    if state.get("error"):
        print(" Skipping SQL generation due to previous error.")
        return state

    print("\n🔧 Step 2: Generating SQL query...")
    try:
        filters = state.get("filters", {})
        sql_query = generate_sql(filters)
        print("✓ SQL generated successfully:")
        print(f"--- SQL START ---\n{sql_query}\n--- SQL END ---")
        
        return {
            **state,
            "sql_query": sql_query
        }
    except Exception as e:
        print(f"✗ SQL generation failed: {e}")
        temp_query_holder["error"] = str(e)
        temp_query_holder["loop_back"] = True
        
        return {
            **state,
            "error": str(e),
            "loop_back": True
        }

def validation_node(state: PipelineState, **kwargs):
    temp_query_holder = kwargs.setdefault("temp_query_holder", {})

    if state.get("error"):
        print(" Skipping validation due to previous error.")
        return state

    print("\n Step 3: Validating and executing SQL...")
    validation_output = validate_sql_and_execute(state["sql_query"])

    if "error" in validation_output:
        msg = validation_output["error"]
        user_correctable = validation_output.get("user_correctable", False)

        if user_correctable:
            temp_query_holder["loop_back"] = True
            temp_query_holder["retry_count"] = temp_query_holder.get("retry_count", 0) + 1

        print(f"✗ Validation issue: {msg}")
        temp_query_holder["error"] = msg
        
        return {
            **state,
            "error": msg,
            "loop_back": user_correctable
        }

    # Success - update state with results
    result_data = validation_output.get("data", [])
    print("✓ Validation and execution successful!")
    print(f"📊 Result sample: {str(result_data)[:250]} ...")
    
    return {
        **state,
        "result": result_data,
        "loop_back": False,
        "error": None
    }

def insert_node(state: PipelineState):
    if state.get("error"):
        print("⊘ Skipping insert due to previous error.")
        return state

    data_to_insert = state.get("result")
    if not data_to_insert:
        print("⊘ No data to insert.")
        return state

    table_name = "product_flat_table"
    print(f"\n💾 Step 4: Inserting data into {table_name}...")

    try:
        # Clear old data
        del_response = supabase.table(table_name).delete().neq("product_id", "").execute()
        if hasattr(del_response, "error") and del_response.error:
            error_msg = f"Failed to clear old data: {del_response.error}"
            print(f"✗ {error_msg}")
            return {**state, "error": error_msg}
        print("✓ Old data cleared.")

        # Insert new data
        response = supabase.table(table_name).insert(data_to_insert).execute()
        if hasattr(response, "error") and response.error:
            error_msg = f"Insert failed: {response.error}"
            print(f"✗ {error_msg}")
            return {**state, "error": error_msg}
        if getattr(response, "status_code", 0) >= 400:
            error_msg = f"Insert failed with status {response.status_code}"
            print(f"✗ {error_msg}")
            return {**state, "error": error_msg}

        print(f"✓ Insert successful! {len(data_to_insert)} rows inserted.")
        return {**state, "inserted": True}
        
    except Exception as e:
        error_msg = f"Insert exception: {str(e)}"
        print(f"✗ {error_msg}")
        return {**state, "error": error_msg}

# ---------------- PIPELINE BUILDER ----------------

def build_pipeline():
    graph = StateGraph(PipelineState)

    graph.add_node("input_node", input_node)
    graph.add_node("intent_node", intent_node)
    graph.add_node("sqlgen_node", sqlgen_node)
    graph.add_node("validation_node", validation_node)
    graph.add_node("insert_node", insert_node)

    graph.add_edge("input_node", "intent_node")
    graph.add_edge("intent_node", "sqlgen_node")
    graph.add_edge("sqlgen_node", "validation_node")
    
    # Conditional loop-back after validation
    graph.add_conditional_edges(
        "validation_node",
        lambda state: "input_node" if state.get("loop_back") else "insert_node",
        {"input_node": "input_node", "insert_node": "insert_node"},
    )
    
    graph.add_edge("insert_node", END)

    graph.set_entry_point("input_node")
    return graph.compile()

# ---------------- PIPELINE RUNNER ----------------

def run_data_fetching(user_query: str = ""):
    print("\n🚀 Running data fetching pipeline...")
    graph = build_pipeline()

    initial_state = PipelineState(
        user_query=None,
        filters=None,
        sql_query=None,
        result=None,
        error=None,
        inserted=False,
        loop_back=False,
        retry_count=0
    )

    # Temporary holder for interactive query, shared across nodes
    temp_query_holder = {
        "user_query": user_query,
        "retry_count": 0,
        "loop_back": False,
        "filters": None,
        "error": None
    }

    try:
        final_state = graph.invoke(initial_state, config={"configurable": {"temp_query_holder": temp_query_holder}})
    except Exception as e:
        print(f"\nPipeline execution error: {e}")
        return {"retry": False, "message": f"Pipeline error: {str(e)}"}

    print("\nPipeline completed!")

    if final_state.get("loop_back"):
        print("\n No matching data found. Asking user to refine query...")
        return {
            "retry": True,
            "message": final_state.get("error", "No data found. Please try again."),
            "previous_filters": final_state.get("filters", {})
        }

    if final_state.get("error"):
        print(f"\nPipeline error: {final_state['error']}")
        return {"retry": False, "message": final_state["error"]}

    if not final_state.get("result"):
        return {"retry": True, "message": "No data found. Please refine your filters."}

    return {"retry": False, "data": final_state["result"]}