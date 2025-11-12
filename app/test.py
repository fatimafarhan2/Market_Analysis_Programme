
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict, Literal, Any, Dict, Optional, List
import operator
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
import matplotlib.pyplot as plt
import pandas as pd
from google import genai
from google.genai import types

# Custom tools
from app.analysis.tool1 import swot_analysis as swot_tool
from app.analysis.market_tool import deep_market_analysis
from app.analysis.reviews_summary_tool import fetch_reviews_summary
from app.data_fetching.agents.intent_agent import detect_intent
from app.data_fetching.agents.sql_agent import generate_sql
from app.data_fetching.agents.validator_agent import validate_sql_and_execute
from app.visualization.market_tool import deep_analysis

# --- Config ---
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

llm_generator = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.4)
llm_reflector = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.3)

TOOLS = [swot_tool, deep_market_analysis, fetch_reviews_summary]
TOOL_MAP = {
    "swot_analysis": swot_tool,
    "deep_market_analysis": deep_market_analysis,
    "fetch_reviews_summary": fetch_reviews_summary
}
llm_generator_with_tools = llm_generator.bind_tools(TOOLS)

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ============================================================================
# UPDATED PIPELINE STATE WITH ALL OUTPUTS
# ============================================================================

class PipelineState(TypedDict):
    # Existing fields
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
    tools_used: list
    tool_outputs: dict
    tool_summaries: dict
    
    chart_paths: List[str]  
    chart_insights: Dict[str, str]
    text_summary: Dict[str, Any]  
    final_output: Dict[str, Any]  


# ============================================================================
# EXISTING NODES (keeping your current implementation)
# ============================================================================

def input_node(state: PipelineState, **kwargs):
   
    temp_query_holder = kwargs.setdefault("temp_query_holder", {})
    print("\n[INPUT NODE] Awaiting user input...")
    MAX_RETRIES = 5
    current_retry = temp_query_holder.get("retry_count", 0)
    if current_retry >= MAX_RETRIES:
        return {**state, "error": "Maximum retries reached.", "loop_back": False}
    if temp_query_holder.get("loop_back") or temp_query_holder.get("error"):
        temp_query_holder["retry_count"] = current_retry + 1
        temp_query_holder["user_query"] = input("Please enter a refined query: ")
    elif "user_query" not in temp_query_holder or not temp_query_holder["user_query"]:
        temp_query_holder["user_query"] = input("Please enter your query: ")
    temp_query_holder["error"] = None
    temp_query_holder["loop_back"] = False
    return {**state, "user_query": temp_query_holder["user_query"], "error": None, 
            "loop_back": False, "retry_count": temp_query_holder.get("retry_count", 0)}

def intent_node(state: PipelineState, **kwargs):
   
    temp_query_holder = kwargs.setdefault("temp_query_holder", {})
    user_query = state.get("user_query")
    print("\n[INTENT NODE] Detecting intent...")
    try:
        intent = detect_intent(user_query)
        temp_query_holder["filters"] = intent.get("filters", {})
        return {**state, "filters": intent.get("filters", {}), "loop_back": False, "error": None}
    except Exception as e:
        temp_query_holder["error"] = str(e)
        temp_query_holder["loop_back"] = True
        return {**state, "error": str(e), "loop_back": True}

def sqlgen_node(state: PipelineState, **kwargs):
    
    if state.get("error"):
        return state
    print("\n[SQL NODE] Generating SQL...")
    try:
        filters = state.get("filters", {})
        sql_query = generate_sql(filters)
        print(sql_query)
        return {**state, "sql_query": sql_query}
    except Exception as e:
        return {**state, "error": str(e), "loop_back": True}

def validation_node(state: PipelineState, **kwargs):
   
    if state.get("error"):
        return state
    print("\n[VALIDATION NODE] Validating SQL...")
    validation_output = validate_sql_and_execute(state["sql_query"])
    print("Validation process done")
    if "error" in validation_output:
        return {**state, "error": validation_output["error"], 
                "loop_back": validation_output.get("user_correctable", False)}
    return {**state, "result": validation_output.get("data", []), "loop_back": False, "error": None}

def insert_node(state: PipelineState):
   
    if state.get("error") or not state.get("result"):
        return state
    table_name = "product_flat_table"
    try:
        supabase.table(table_name).delete().neq("product_id", "").execute()
        supabase.table(table_name).insert(state["result"]).execute()
        print("[INSERT NODE] Data insertion successful.")
        return {**state, "inserted": True}
    except Exception as e:
        return {**state, "error": f"Insert exception: {str(e)}"}


# ============================================================================
# TOKEN MANAGEMENT & GENERATOR NODE
# ============================================================================

def count_tokens_approximate(text: str) -> int:
    return len(text) // 4

def truncate_tool_output(tool_output: dict, max_tokens: int = 1000) -> dict:
   
    output_str = json.dumps(tool_output, indent=2)
    if count_tokens_approximate(output_str) <= max_tokens:
        return tool_output
    truncated = {}
    for key, value in tool_output.items():
        if isinstance(value, list) and len(value) > 5:
            truncated[key] = value[:5]
            truncated[f"{key}_count"] = len(value)
        elif isinstance(value, dict) and len(str(value)) > 500:
            truncated[key] = {k: v for k, v in list(value.items())[:3]}
        else:
            truncated[key] = value
    return truncated

def summarize_tool_output_with_llm(tool_name: str, tool_output: dict, llm) -> str:
   
    try:
        prompt = f"Summarize this {tool_name} output in 100-150 words:\n{json.dumps(tool_output, indent=2)[:3000]}"
        result = llm.invoke([HumanMessage(content=prompt)])
        return getattr(result, "content", str(result))
    except Exception as e:
        return json.dumps(tool_output, indent=2)[:500]

def node_generator(state: PipelineState) -> dict:

    user_query = state.get("user_query", "")
    filters = state.get("filters", {})
    iteration = state.get("iteration", 0)
    messages = list(state.get("messages", []))
    tools_used = state.get("tools_used", [])
    tool_outputs = state.get("tool_outputs", {})
    tool_summaries = state.get("tool_summaries", {})

    print(f"\n[GENERATOR NODE] Iteration {iteration + 1}")

    if iteration == 0:
        system_prompt = f"""You are a market analysis assistant  ReAct agent.
User Query: {user_query}
Filters: {json.dumps(filters, indent=2)}

ReAct Process: - Think: Decide which tool to use next - Act: Call ONE tool at a time - Observe: Review the tool output - Repeat until all relevant tools are used
You cannot output any result without using tools
You are not supposed to use all tools , its not necessary, use it ccording to the need like think using the query what is required 
Available tools:
1. swot_analysis - SWOT analysis , only use this if in the query the Product and the country is mentioned in user query, any one missing we cannot use this tool
2. deep_market_analysis - Market insights  , for rich numerical insights
3. fetch_reviews_summary - Reviews summary
Process: Think → Act (one tool) → Observe → Repeat
After all tools: Create JSON report with executive_summary, detailed_analysis, key_insights, recommendations."""
        messages = [SystemMessage(content=system_prompt)]

    available_tools = [t for t in TOOL_MAP.keys() if t not in tools_used]
    
    if available_tools and iteration < 5:
        summaries_text = "\n".join([f"- {name}: {summary[:200]}..." 
                                    for name, summary in tool_summaries.items()])
        reasoning_prompt = f"Tools used: {tools_used}\nAvailable: {available_tools}\n{summaries_text}\nChoose ONE tool."
        messages.append(HumanMessage(content=reasoning_prompt))
        
        try:
            result = llm_generator_with_tools.invoke(messages)
        except Exception as e:
            if "rate_limit" in str(e).lower():
                messages = [messages[0]] + messages[-3:]
                result = llm_generator_with_tools.invoke(messages)
            else:
                raise
        
        tool_calls = getattr(result, 'tool_calls', [])
        
        if tool_calls:
            tool_call = tool_calls[0]
            tool_name = tool_call.get("name")
            tool_id = tool_call.get("id", f"tool_{iteration}")

            if tool_name in tools_used:
                messages.append(result)
                return {**state, "iteration": iteration + 1, "messages": messages, 
                        "tools_used": tools_used, "tool_outputs": tool_outputs, 
                        "tool_summaries": tool_summaries}

            print(f"[GENERATOR NODE] Executing tool: {tool_name}")
            try:
                tool_fn = TOOL_MAP[tool_name]
                raw_output = tool_fn.func(user_query) if hasattr(tool_fn, "func") else tool_fn(user_query)
                tool_outputs[tool_name] = raw_output
                summary = summarize_tool_output_with_llm(tool_name, raw_output, llm_generator)
                tool_summaries[tool_name] = summary
                tools_used.append(tool_name)
                messages.append(result)
                messages.append(ToolMessage(content=f"Tool: {tool_name}\n{summary}", 
                                           tool_call_id=tool_id, name=tool_name))
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                tool_outputs[tool_name] = {"error": error_msg}
                tool_summaries[tool_name] = error_msg
                tools_used.append(tool_name)

            return {**state, "iteration": iteration + 1, "messages": messages, 
                    "tools_used": tools_used, "tool_outputs": tool_outputs, 
                    "tool_summaries": tool_summaries}

    # Generate final report
    print("[GENERATOR NODE] Generating final report...")
    truncated_outputs = {k: truncate_tool_output(v, 2000) for k, v in tool_outputs.items()}
    
    synthesis_prompt = f"""Generate JSON report:
{json.dumps(truncated_outputs, indent=2)}

Structure:
{{
    "user_query": "{user_query}",
    "tools_used": {json.dumps(tools_used)},
    "executive_summary": "2-3 sentences with key numbers",
    "detailed_analysis": "300+ words with metrics and trends",
    "key_insights": ["insight1", "insight2", "insight3", "insight4"],
    "recommendations": ["rec1", "rec2", "rec3"],
    "metrics_summary": {{"metric1": "value1", "metric2": "value2"}}
}}
Output ONLY valid JSON."""

    final_messages = [SystemMessage(content="You are a data analyst."), 
                     HumanMessage(content=synthesis_prompt)]
    final_result = llm_generator.invoke(final_messages)
    final_content = getattr(final_result, "content", str(final_result))

    # Clean JSON
    if "```json" in final_content:
        final_content = final_content.split("```json")[1].split("```")[0].strip()
    elif "```" in final_content:
        final_content = final_content.split("```")[1].split("```")[0].strip()

    return {**state, "iteration": iteration + 1, "last_generator_text": final_content, 
            "messages": [], "tools_used": tools_used, "tool_outputs": tool_outputs, 
            "tool_summaries": tool_summaries, "final_analysis": final_content}

def node_reflector(state: PipelineState) -> dict:
    """[Your existing reflector]"""
    last_gen = state.get("last_generator_text", "")
    if not last_gen or len(last_gen) < 50:
        return {**state, "last_reflector_text": "CONTINUE - Still gathering data"}
    try:
        if "detailed_analysis" in json.loads(last_gen):
            return {**state, "last_reflector_text": "CONVERGE - Report complete"}
    except:
        pass
    return {**state, "last_reflector_text": "CONTINUE"}

def should_continue(state: PipelineState) -> str:
    """[Your existing logic]"""
    iteration = state.get("iteration", 0)
    last_ref = (state.get("last_reflector_text") or "").lower()
    if "converge" in last_ref or iteration >= 5:
        return "visualization"
    return "reflector"


# ============================================================================
# ENHANCED VISUALIZATION NODE (creates charts)
# ============================================================================

def visualization_node(state: PipelineState, **kwargs):
    """Create visualizations and store chart paths"""
    print("\n[VISUALIZATION NODE] Creating charts...")
    
    analysis = deep_analysis()
    os.makedirs('charts', exist_ok=True)
    chart_paths = []
    
    # Chart 1: Country Revenue
    try:
        country_data = pd.DataFrame(analysis['country_market_share_topN'])
        plt.figure(figsize=(10, 6))
        plt.bar(country_data['country_name'], country_data['total_revenue_usd'], 
                color='steelblue', edgecolor='black')
        plt.title('Revenue by Country', fontsize=16, fontweight='bold')
        plt.ylabel('Revenue (USD)', fontsize=12)
        plt.xlabel('Country', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        chart_path = 'charts/country_revenue.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        chart_paths.append(chart_path)
        print(f" Created: {chart_path}")
    except Exception as e:
        print(f"Error creating country chart: {e}")

    # Chart 2: Brand Market Share
    try:
        brand_data = pd.DataFrame(analysis['brand_market_share_topN'][:8])
        plt.figure(figsize=(10, 8))
        colors = plt.cm.Set3(range(len(brand_data)))
        plt.pie(brand_data['share'], labels=brand_data['brand_name'], 
                autopct='%1.1f%%', startangle=140, colors=colors)
        plt.title('Top Brands Market Share', fontsize=16, fontweight='bold')
        plt.tight_layout()
        chart_path = 'charts/brand_market_share.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        chart_paths.append(chart_path)
        print(f" Created: {chart_path}")
    except Exception as e:
        print(f"Error creating brand chart: {e}")

    # Chart 3: Top Products Revenue
    try:
        product_data = pd.DataFrame(analysis['product_summary_topN'][:10])
        plt.figure(figsize=(12, 6))
        plt.barh(product_data['product_name'], product_data['total_revenue_usd'], 
                color='coral', edgecolor='black')
        plt.title('Top 10 Products by Revenue', fontsize=16, fontweight='bold')
        plt.xlabel('Revenue (USD)', fontsize=12)
        plt.ylabel('Product', fontsize=12)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        chart_path = 'charts/top_products_revenue.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        chart_paths.append(chart_path)
        print(f" Created: {chart_path}")
    except Exception as e:
        print(f"Error creating products chart: {e}")

    # Chart 4: Correlation Heatmap
    try:
        if 'correlation_matrix' in analysis:
            corr_data = pd.DataFrame(analysis['correlation_matrix'])
            plt.figure(figsize=(10, 8))
            plt.imshow(corr_data, cmap='coolwarm', interpolation='nearest', vmin=-1, vmax=1)
            plt.colorbar(label='Correlation Coefficient')
            plt.xticks(range(len(corr_data.columns)), corr_data.columns, rotation=45, ha='right')
            plt.yticks(range(len(corr_data.columns)), corr_data.columns)
            plt.title('Correlation Heatmap of Metrics', fontsize=16, fontweight='bold')
            plt.tight_layout()
            chart_path = 'charts/correlation_heatmap.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            chart_paths.append(chart_path)
            print(f" Created: {chart_path}")
    except Exception as e:
        print(f"Error creating correlation chart: {e}")

    print(f"[VISUALIZATION NODE]  Created {len(chart_paths)} charts")
    return {**state, "chart_paths": chart_paths}


# ============================================================================
# NEW: GEMINI CHART ANALYSIS NODE
# ============================================================================


from pathlib import Path
import pandas as pd
import numpy as np

import pandas as pd
from pathlib import Path

import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def local_chart_analysis_node(state):
    """Analyze charts locally using the underlying data used to create them"""
    print("\n[LOCAL CHART ANALYSIS NODE] Extracting insights from chart data...")

    analysis = deep_analysis()  # Your existing data source
    chart_insights = {}

  
    try:
        country_data = pd.DataFrame(analysis['country_market_share_topN'])
        if not country_data.empty:
            total_rev = country_data['total_revenue_usd'].sum()
            max_country = country_data.loc[country_data['total_revenue_usd'].idxmax()]
            min_country = country_data.loc[country_data['total_revenue_usd'].idxmin()]

            chart_insights['country_revenue'] = (
                f"Total revenue: ${total_rev:,.2f}\n"
                f"Highest revenue: {max_country['country_name']} (${max_country['total_revenue_usd']:,.2f})\n"
                f"Lowest revenue: {min_country['country_name']} (${min_country['total_revenue_usd']:,.2f})"
            )
        else:
            chart_insights['country_revenue'] = "No data available"
    except Exception as e:
        chart_insights['country_revenue'] = f"Analysis failed: {str(e)}"

    try:
        brand_data = pd.DataFrame(analysis['brand_market_share_topN'][:8])
        if not brand_data.empty:
            top_brand = brand_data.loc[brand_data['share'].idxmax()]
            chart_insights['brand_market_share'] = (
                f"Top brand: {top_brand['brand_name']} ({top_brand['share']}% share)\n"
                f"Brands included: {', '.join(brand_data['brand_name'].tolist())}"
            )
        else:
            chart_insights['brand_market_share'] = "No data available"
    except Exception as e:
        chart_insights['brand_market_share'] = f"Analysis failed: {str(e)}"


    try:
        product_data = pd.DataFrame(analysis['product_summary_topN'][:10])
        if not product_data.empty:
            top_product = product_data.loc[product_data['total_revenue_usd'].idxmax()]
            chart_insights['top_products_revenue'] = (
                f"Top product: {top_product['product_name']} (${top_product['total_revenue_usd']:,.2f})\n"
                f"Products included: {', '.join(product_data['product_name'].tolist())}"
            )
        else:
            chart_insights['top_products_revenue'] = "No data available"
    except Exception as e:
        chart_insights['top_products_revenue'] = f"Analysis failed: {str(e)}"

    
    try:
        if 'correlation_matrix' in analysis:
            corr_data = pd.DataFrame(analysis['correlation_matrix'])
            if not corr_data.empty:
                corr_summary = corr_data.abs().unstack().sort_values(ascending=False)
                top_corr = corr_summary[corr_summary < 1].head(5)
                chart_insights['correlation_heatmap'] = "Top correlations:\n" + top_corr.to_string()
            else:
                chart_insights['correlation_heatmap'] = "No data available"
        else:
            chart_insights['correlation_heatmap'] = "No correlation data available"
    except Exception as e:
        chart_insights['correlation_heatmap'] = f"Analysis failed: {str(e)}"

    print("[LOCAL CHART ANALYSIS NODE] Completed analysis for all charts")
    return {**state, "chart_insights": chart_insights}

# ============================================================================
# NEW: FINAL OUTPUT FORMATTING NODE
# ============================================================================

import json
import pandas as pd
from pathlib import Path
from google import genai
from google.genai import types


gemini_client = genai.Client()

def format_final_output_node(state, **kwargs):
    """Extract and format all outputs for frontend consumption, including a Gemini beautified report"""
    print("\n[FORMAT OUTPUT NODE] Preparing final output...")


    final_analysis = state.get("final_analysis", "{}")
    try:
        report = json.loads(final_analysis)
    except json.JSONDecodeError:
        report = {"error": "Failed to parse report", "raw": final_analysis}

    text_summary = {
        "user_query": report.get("user_query", state.get("user_query", "")),
        "executive_summary": report.get("executive_summary", ""),
        "detailed_analysis": report.get("detailed_analysis", ""),
        "key_insights": report.get("key_insights", []),
        "recommendations": report.get("recommendations", []),
    }


    chart_paths = state.get("chart_paths", [])
    chart_insights = state.get("chart_insights", {})

    visualizations = []
    for chart_path in chart_paths:
        chart_name = Path(chart_path).stem
        chart_title = chart_name.replace('_', ' ').title()
        visualizations.append({
            "filename": Path(chart_path).name,
            "path": chart_path,
            "title": chart_title,
            "chart_name": chart_name,
            "insights": chart_insights.get(chart_name, "No insights available"),
            "url": f"/api/charts/{Path(chart_path).name}"  # For API serving
        })


    report_text = f"""
    USER QUERY: {text_summary['user_query']}

    EXECUTIVE SUMMARY:
    {text_summary['executive_summary']}

    DETAILED ANALYSIS:
    {text_summary['detailed_analysis']}

    KEY INSIGHTS:
    {chr(10).join(f"- {k}" for k in text_summary['key_insights'])}

    RECOMMENDATIONS:
    {chr(10).join(f"- {r}" for r in text_summary['recommendations'])}

    CHART INSIGHTS:
    {chr(10).join(f"{v['title']}: {v['insights']}" for v in visualizations)}
    """

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[f"Beautify and format this market analysis report professionally:\n{report_text}"],
            config=types.GenerateContentConfig(temperature=0.2)
        )
        beautified_report = response.text
    except Exception as e:
        print(f"[FORMAT OUTPUT NODE] Gemini beautification failed: {e}")
        beautified_report = report_text

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    json_output_path = output_dir / "final_analysis_output.json"
    report_output_path = output_dir / "final_analysis_report.txt"

    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "text_summary": text_summary,
            "visualizations": visualizations,
            "metadata": {
                "tools_used": state.get("tools_used", []),
                "iterations": state.get("iteration", 0),
                "total_charts": len(visualizations),
                "data_rows": len(state.get("result", []))
            }
        }, f, indent=2, ensure_ascii=False)

    with open(report_output_path, 'w', encoding='utf-8') as f:
        f.write(beautified_report)

    print(f"[FORMAT OUTPUT NODE] JSON output saved to {json_output_path}")
    print(f"[FORMAT OUTPUT NODE] Beautified report saved to {report_output_path}")
    print(f"[FORMAT OUTPUT NODE] {len(visualizations)} charts included")

    return {
        **state,
        "text_summary": text_summary,
        "final_output_json": json_output_path,
        "final_output_report": report_output_path
    }
# ============================================================================
# UPDATED PIPELINE BUILDER
# ============================================================================

def build_pipeline():
    """Build complete pipeline with all nodes"""
    graph = StateGraph(PipelineState)

    # Add all nodes
    graph.add_node("input_node", input_node)
    graph.add_node("intent_node", intent_node)
    graph.add_node("sqlgen_node", sqlgen_node)
    graph.add_node("validation_node", validation_node)
    graph.add_node("insert_node", insert_node)
    graph.add_node("generator", node_generator)
    graph.add_node("reflector", node_reflector)
    graph.add_node("visualization", visualization_node)
    graph.add_node("Chart_Analysis", local_chart_analysis_node)  # NEW
    graph.add_node("format_output", format_final_output_node)  # NEW

    # Connect flow
    graph.add_edge(START, "input_node")
    graph.add_edge("input_node", "intent_node")
    graph.add_edge("intent_node", "sqlgen_node")
    graph.add_edge("sqlgen_node", "validation_node")

    graph.add_conditional_edges(
        "validation_node",
        lambda state: "input_node" if state.get("loop_back") else "insert_node",
        {"input_node": "input_node", "insert_node": "insert_node"}
    )

    graph.add_edge("insert_node", "generator")
    graph.add_conditional_edges("generator", should_continue, ["reflector", "visualization"])
    graph.add_edge("reflector", "generator")
    
    graph.add_edge("visualization", "Chart_Analysis")
    graph.add_edge("Chart_Analysis", "format_output")
    graph.add_edge("format_output", END)

    return graph.compile()


# ============================================================================
# UPDATED PIPELINE RUNNER
# ============================================================================

def run_data_fetching(user_query: str = ""):
    """Run complete pipeline and return frontend-ready output"""
    print("\n" + "="*80)
    print(" MARKET ANALYSIS PIPELINE")
    print("="*80)
    
    graph = build_pipeline()
    
    
    initial_state = PipelineState(
        user_query=None,  
        filters=None,
        sql_query=None,
        result=None,
        error=None,
        inserted=False,
        loop_back=False,
        retry_count=0,
        messages=[],
        history=[],
        iteration=0,
        last_generator_text="",
        last_reflector_text="",
        final_analysis="",
        tools_used=[],
        tool_outputs={},
        tool_summaries={},
        chart_paths=[],
        chart_insights={},
        text_summary={},
        final_output={}
    )
    
    temp_query_holder = {
        "user_query": user_query if user_query else None,  
        "retry_count": 0,
        "loop_back": False,
        "filters": None,
        "error": None
    }

    try:
        final_state = graph.invoke(
            initial_state,
            config={"configurable": {"temp_query_holder": temp_query_holder}}
        )
    except Exception as e:
        print(f"\n [PIPELINE ERROR] {str(e)}")
        return {
            "status": "error",
            "message": f"Pipeline error: {str(e)}",
            "retry": False
        }

    if final_state.get("error"):
        return {
            "status": "error",
            "message": final_state["error"],
            "retry": final_state.get("loop_back", False)
        }
    
    if not final_state.get("result"):
        return {
            "status": "error",
            "message": "No data found",
            "retry": True
        }
    
    final_output = final_state.get("final_output", {})
    
    print("\n" + "="*80)
    print(" PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f" Tools Used: {', '.join(final_output.get('metadata', {}).get('tools_used', []))}")
    print(f" Iterations: {final_output.get('metadata', {}).get('iterations', 0)}")
    print(f" Charts Created: {final_output.get('metadata', {}).get('total_charts', 0)}")
    print(f" Data Rows: {final_output.get('metadata', {}).get('data_rows', 0)}")
    print("\n Output saved to: output/final_analysis_output.json")
    print("="*80 + "\n")
    
    return final_output

