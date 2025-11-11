from IPython.display import display,Image
display(Image(compiled_agent.get_graph().draw_mermaid_png()))
png_bytes = compiled_agent.get_graph().draw_mermaid_png()  # get PNG bytes
with open("agent_graph.png", "wb") as f:
    f.write(png_bytes)
print("✅ Saved as agent_graph.png")