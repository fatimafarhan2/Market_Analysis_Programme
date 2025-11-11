<!-- # # file: react_agent_gemini.py

# import os
# from dotenv import load_dotenv

# from google import genai
# from langchain.agents import create_tool_calling_agent, AgentExecutor
# from langchain.tools import tool
# from langchain.schema import HumanMessage, AIMessage



# # load env
# load_dotenv()
# # initialize Gemini client
# client = genai.Client(api_key=os.getenv())

# # Define your custom tools
# from app.analysis.tool1 import swot_analysis
# from app.analysis.market_tool import deep_market_analysis
# from app.analysis.reviews_summary_tool import fetch_reviews_summary

# @tool("swot_analysis", return_direct=True)
# def swot_tool(company: str):
#     """Perform SWOT analysis for the given company."""
#     return swot_analysis()

# @tool("market_analysis", return_direct=True)
# def market_tool(company: str):
#     """Perform deep market analysis."""
#     return deep_market_analysis()

# @tool("reviews_summary", return_direct=True)
# def reviews_tool(company: str):
#     """Fetch summarized reviews."""
#     return fetch_reviews_summary()

# tools = [swot_tool, market_tool, reviews_tool]

# # Wrap Gemini client in a simple class compatible with LangChain
# class GeminiLLM:
#     def __init__(self, client, model="gemini-2.5-flash", temperature=0.7):
#         self.client = client
#         self.model = model
#         self.temperature = temperature

#     def invoke(self, messages, **kwargs):
#         # combine messages into one prompt
#         prompt = "\n".join(m.content for m in messages if isinstance(m, (HumanMessage, AIMessage)))
#         resp = self.client.models.generate_content(
#             model=self.model,
#             contents=prompt,
#             generation_config={"temperature": self.temperature}
#         )
#         return AIMessage(content=resp.text)

# # instantiate
# llm = GeminiLLM(client)

# # create the agent
# prompt = (
#     "You are a business analysis agent following the ReAct pattern. "
#     "Alternate between reasoning steps and tool calls, then give a final answer."
# )

# agent = create_tool_calling_agent(
#     llm=llm,
#     tools=tools,
#     prompt=prompt
# )

# executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# # Example usage
# if __name__ == "__main__":
#     input_query = "Perform a SWOT and market analysis of Tesla and summarize customer reviews."
#     result = executor.invoke({"input": input_query})
#     print("\nFinal Output:")
#     print(result["output"]) -->
