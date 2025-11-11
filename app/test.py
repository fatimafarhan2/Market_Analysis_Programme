from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict, Literal, Any, Dict, Optional
import operator
import difflib
import json
import os
from dotenv import load_dotenv
from supabase import create_client

# LangChain Groq import
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

# Custom tools
from app.analysis.tool1 import swot_analysis as swot_tool
from app.analysis.market_tool import deep_market_analysis
from app.analysis.reviews_summary_tool import fetch_reviews_summary

# Agents
from app.data_fetching.agents.intent_agent import detect_intent
from app.data_fetching.agents.sql_agent import generate_sql
from app.data_fetching.agents.validator_agent import validate_sql_and_execute

# --- Config / load env ---
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Please set SUPABASE_URL and SUPABASE_KEY in your environment or .env file")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# -----------------------------Model loading--------------------
llm_generator = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.7)
llm_reflector = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.7)

TOOLS = [swot_tool, deep_market_analysis, fetch_reviews_summary]

# Create tool map for execution
TOOL_MAP = {
    "swot_analysis": swot_tool,
    "deep_market_analysis": deep_market_analysis,
    "fetch_reviews_summary": fetch_reviews_summary
}

# Bind tools to the model
llm_generator_with_tools = llm_generator.bind_tools(TOOLS)

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
    messages: Annotated[list, add_messages]
    history: Annotated[list, operator.add]
    iteration: int
    last_generator_text: str
    last_reflector_text: str
    final_analysis: str

# ---------------- NODES ----------------
def input_node(state: PipelineState, **kwargs):
    """Prompt user for input or refined query in case of errors or loopbacks."""
    temp_query_holder = kwargs.setdefault("temp_query_holder", {})
    print("\n[INPUT NODE] Awaiting user input...")
    MAX_RETRIES = 5

    current_retry = temp_query_holder.get("retry_count", 0)
    if current_retry >= MAX_RETRIES:
        print("[INPUT NODE] Maximum retries reached.")
        return {**state, "error": "Maximum retries reached.", "loop_back": False}

    if temp_query_holder.get("loop_back") or temp_query_holder.get("error"):
        temp_query_holder["retry_count"] = current_retry + 1
        temp_query_holder["user_query"] = input("Please enter a refined query: ")
        print(f"[INPUT NODE] Refined query received: {temp_query_holder['user_query']}")
    elif "user_query" not in temp_query_holder or not temp_query_holder["user_query"]:
        temp_query_holder["user_query"] = input("Please enter your query: ")
        print(f"[INPUT NODE] Initial query received: {temp_query_holder['user_query']}")

    temp_query_holder["error"] = None
    temp_query_holder["loop_back"] = False

    return {**state, "user_query": temp_query_holder["user_query"], "error": None, "loop_back": False,
            "retry_count": temp_query_holder.get("retry_count", 0)}

def intent_node(state: PipelineState, **kwargs):
    """Detect user intent from query and extract filters for SQL."""
    temp_query_holder = kwargs.setdefault("temp_query_holder", {})
    user_query = state.get("user_query")
    print("\n[INTENT NODE] Step 1: Detecting intent from query...")
    print(f"[INTENT NODE] User query: {user_query}")
    try:
        intent = detect_intent(user_query)
        temp_query_holder["filters"] = intent.get("filters", {})
        print(f"[INTENT NODE] Detected filters: {temp_query_holder['filters']}")
        return {**state, "filters": intent.get("filters", {}), "loop_back": False, "error": None}
    except Exception as e:
        print(f"[INTENT NODE] Error in detecting intent: {str(e)}")
        temp_query_holder["error"] = str(e)
        temp_query_holder["loop_back"] = True
        return {**state, "error": str(e), "loop_back": True}

def sqlgen_node(state: PipelineState, **kwargs):
    """Generate SQL query from detected filters."""
    temp_query_holder = kwargs.setdefault("temp_query_holder", {})
    if state.get("error"):
        return state

    print("\n[SQL NODE] Step 2: Generating SQL query...")
    try:
        filters = state.get("filters", {})
        sql_query = generate_sql(filters)
        print(f"[SQL NODE] Generated SQL: {sql_query}")
        return {**state, "sql_query": sql_query}
    except Exception as e:
        print(f"[SQL NODE] Error generating SQL: {str(e)}")
        temp_query_holder["error"] = str(e)
        temp_query_holder["loop_back"] = True
        return {**state, "error": str(e), "loop_back": True}

def validation_node(state: PipelineState, **kwargs):
    """Validate and execute SQL, handle user-correctable errors."""
    temp_query_holder = kwargs.setdefault("temp_query_holder", {})
    if state.get("error"):
        return state

    print("\n[VALIDATION NODE] Step 3: Validating and executing SQL...")
    validation_output = validate_sql_and_execute(state["sql_query"])

    if "error" in validation_output:
        msg = validation_output["error"]
        user_correctable = validation_output.get("user_correctable", False)
        print(f"[VALIDATION NODE] SQL Error: {msg}, user_correctable={user_correctable}")
        if user_correctable:
            temp_query_holder["loop_back"] = True
            temp_query_holder["retry_count"] = temp_query_holder.get("retry_count", 0) + 1
        temp_query_holder["error"] = msg
        return {**state, "error": msg, "loop_back": user_correctable}

    print(f"[VALIDATION NODE] SQL executed successfully, rows returned: {len(validation_output.get('data', []))}")
    return {**state, "result": validation_output.get("data", []), "loop_back": False, "error": None}

def insert_node(state: PipelineState):
    """Insert fetched data into Supabase table."""
    if state.get("error") or not state.get("result"):
        return state
    table_name = "product_flat_table"
    try:
        print(f"[INSERT NODE] Clearing old data in table {table_name}...")
        del_response = supabase.table(table_name).delete().neq("product_id", "").execute()
        if hasattr(del_response, "error") and del_response.error:
            print(f"[INSERT NODE] Error clearing old data: {del_response.error}")
            return {**state, "error": f"Failed to clear old data: {del_response.error}"}

        print(f"[INSERT NODE] Inserting {len(state['result'])} rows into table {table_name}...")
        response = supabase.table(table_name).insert(state["result"]).execute()
        if hasattr(response, "error") and response.error:
            print(f"[INSERT NODE] Error inserting data: {response.error}")
            return {**state, "error": f"Insert failed: {response.error}"}
        print("[INSERT NODE] Data insertion successful.")
        return {**state, "inserted": True}
    except Exception as e:
        print(f"[INSERT NODE] Exception during insert: {str(e)}")
        return {**state, "error": f"Insert exception: {str(e)}"}

# ---------------- FIXED ReAct generator node ----------------
def node_generator(state: PipelineState) -> dict:
    """ReAct-style generator that properly handles tool calls from Groq."""
    user_query = state.get("user_query")
    filters = state.get("filters", {})
    iteration = state.get("iteration", 0) + 1
    messages = list(state.get("messages", []))

    print(f"\n[GENERATOR NODE] Iteration {iteration} - Reasoning with LLM...")

    # Add system message with clear instructions about tools
    if iteration == 1:
        system_prompt = f"""You are a market analysis assistant ReAct Agent with access to specialized tools.

User Query: {user_query}
Detected Filters: {json.dumps(filters, indent=2)}

You have access to these tools:
1. swot_analysis - Performs SWOT analysis on products
2. deep_market_analysis - Provides deep market insights
3. fetch_reviews_summary - Fetches and summarizes product reviews

ReAct agent:
Follow the ReAct pattern:
- First, reason about what data you need.
- Then, call the relevant tool(s).
- Observe tool outputs before deciding next steps.
- Never make up data — always use real numbers from tool outputs.
- When confident, give exactly 4 concise, actionable insights in JSON.
- Its better to make use of as much numerical data as possible

IMPORTANT: 
- Use the tools to gather comprehensive information
- Call multiple tools if needed to provide complete analysis
- After receiving tool results, synthesize them into a coherent response
- Be thorough and use all relevant tools for the query
- Make sure you use numerical vaues to support but dont hallucinate ,use the numerical values you get from tools 


Analyze the query and decide which tools to use."""
        
        messages = [SystemMessage(content=system_prompt)] + messages

    # Call LLM with tools bound
    result = llm_generator_with_tools.invoke(messages)
    
    # Check if the model wants to use tools
    tool_calls = getattr(result, 'tool_calls', [])
    
    if tool_calls:
        print(f"[GENERATOR NODE] Model requested {len(tool_calls)} tool calls")
        
        # Add the AI message with tool calls to history
        messages.append(result)
        
        # Execute each tool call
        for tool_call in tool_calls:
            tool_name = tool_call.get('name')
            tool_args = tool_call.get('args', {})
            tool_id = tool_call.get('id', 'unknown')
            
            print(f"[GENERATOR NODE] Executing tool: {tool_name} with args: {tool_args}")
            
            try:
                # Execute the tool with proper arguments
                if tool_name in TOOL_MAP:
                    # All tools now accept a query string
                    tool_fn = TOOL_MAP[tool_name]
                    # Call the actual function, not the tool wrapper
                    tool_output = tool_fn.func(user_query) if hasattr(tool_fn, 'func') else tool_fn(user_query)
                    print(f"[GENERATOR NODE] Tool {tool_name} executed successfully")
                else:
                    tool_output = {"error": f"Unknown tool: {tool_name}"}
                    print(f"[GENERATOR NODE] Unknown tool: {tool_name}")
                
                # Add tool result as ToolMessage
                messages.append(
                    ToolMessage(
                        content=json.dumps(tool_output),
                        tool_call_id=tool_id,
                        name=tool_name
                    )
                )
            except Exception as e:
                error_msg = f"Error executing {tool_name}: {str(e)}"
                print(f"[GENERATOR NODE] {error_msg}")
                messages.append(
                    ToolMessage(
                        content=json.dumps({"error": error_msg}),
                        tool_call_id=tool_id,
                        name=tool_name
                    )
                )
        
        # Call LLM again with tool results
        print("[GENERATOR NODE] Calling LLM with tool results...")
        final_result = llm_generator_with_tools.invoke(messages)
        final_content = getattr(final_result, "content", "")
        messages.append(final_result)
        
    else:
        # No tool calls, just use the content
        print("[GENERATOR NODE] No tool calls requested by model")
        final_content = getattr(result, "content", "")
        messages.append(result)

    print(f"[GENERATOR NODE] Final output:\n{final_content}")

    return {
        "iteration": iteration,
        "last_generator_text": final_content,
        "history": [{"iteration": iteration, "generator_text": final_content}],
        "messages": messages
    }

# ---------------- ReAct reflector node ----------------
def node_reflector(state: PipelineState) -> dict:
    """Reflector: checks if generator output meets requirements, advises revision if needed."""
    sys_msg = """You are a quality control analyst reviewing market insights.

Evaluate the analysis based on:
1. Completeness - Does it address all aspects of the query?
2. Tool Usage - Were appropriate tools used?
3. Depth - Is the analysis thorough and insightful?
4. Coherence - Is the information well-organized?

If the analysis is satisfactory, respond with "CONVERGE - Analysis is complete."
If improvements are needed, specify what's missing and suggest which tools to use."""

    system_msg = SystemMessage(content=sys_msg)
    messages = [system_msg, HumanMessage(content=state.get("last_generator_text", ""))]

    result = llm_reflector.invoke(messages)
    critique_text = getattr(result, "content", getattr(result, "text", str(result)))
    
    print(f"[REFLECTOR NODE] Reflector output:\n{critique_text}")

    return {
        "last_reflector_text": critique_text,
        "history": [{"iteration": state.get("iteration", 0), "reflector_text": critique_text}],
        "messages": [AIMessage(content=critique_text)]
    }

# ---------------- Stop criteria ----------------
MAX_ITERS = 3
SIM_THRESHOLD = 0.86

def should_continue(state: PipelineState) -> Literal["reflector", "end"]:
    it = state.get("iteration", 0)
    if it >= MAX_ITERS:
        print("[SHOULD_CONTINUE] Max iterations reached")
        return END
    
    last_ref = (state.get("last_reflector_text") or "").lower()
    if "converge" in last_ref:
        print("[SHOULD_CONTINUE] Reflector signaled convergence")
        return END
    
    hist = state.get("history", []) or []
    gen_texts = [h.get("generator_text") for h in hist if h.get("generator_text")]
    if len(gen_texts) >= 2:
        similarity = difflib.SequenceMatcher(None, gen_texts[-2], gen_texts[-1]).ratio()
        if similarity >= SIM_THRESHOLD:
            print(f"[SHOULD_CONTINUE] High similarity detected: {similarity:.2f}")
            return END
    
    print("[SHOULD_CONTINUE] Continuing to reflector")
    return "reflector"

# ---------------- Pipeline builder ----------------
def build_pipeline():
    graph = StateGraph(PipelineState)

    # Add nodes
    graph.add_node("input_node", input_node)
    graph.add_node("intent_node", intent_node)
    graph.add_node("sqlgen_node", sqlgen_node)
    graph.add_node("validation_node", validation_node)
    graph.add_node("insert_node", insert_node)
    graph.add_node("generator", node_generator)
    graph.add_node("reflector", node_reflector)

    # Connect nodes
    graph.add_edge(START, "input_node")
    graph.add_edge("input_node", "intent_node")
    graph.add_edge("intent_node", "sqlgen_node")
    graph.add_edge("sqlgen_node", "validation_node")

    # Conditional loop-back after validation
    graph.add_conditional_edges(
        "validation_node",
        lambda state: "input_node" if state.get("loop_back") else "insert_node",
        {"input_node": "input_node", "insert_node": "insert_node"},
    )

    # After insertion, go to generator
    graph.add_edge("insert_node", "generator")

    # ReAct agent loop
    graph.add_conditional_edges("generator", should_continue, ["reflector", END])
    graph.add_edge("reflector", "generator")

    return graph.compile()

# ---------------- Pipeline runner ----------------
def run_data_fetching(user_query: str = ""):
    print("\n🚀 Running data fetching pipeline...")
    graph = build_pipeline()
    initial_state = PipelineState(
        user_query=user_query,
        filters=None,
        sql_query=None,
        result=None,
        error=None,
        inserted=False,
        loop_back=False,
        retry_count=0,
        messages=[HumanMessage(content=user_query)],
        history=[],
        iteration=0,
        last_generator_text="",
        last_reflector_text="",
        final_analysis=""
    )
    temp_query_holder = {"user_query": user_query, "retry_count": 0, "loop_back": False, "filters": None, "error": None}

    try:
        final_state = graph.invoke(initial_state, config={"configurable": {"temp_query_holder": temp_query_holder}})
    except Exception as e:
        return {"retry": False, "message": f"Pipeline error: {str(e)}"}

    if final_state.get("loop_back"):
        return {"retry": True, "message": final_state.get("error", "No data found."), "previous_filters": final_state.get("filters", {})}
    if final_state.get("error"):
        return {"retry": False, "message": final_state["error"]}
    if not final_state.get("result"):
        return {"retry": True, "message": "No data found. Please refine your filters."}
    
    return {
        "retry": False, 
        "data": final_state["result"],
        "analysis": final_state.get("last_generator_text", ""),
        "iterations": final_state.get("iteration", 0)
    }