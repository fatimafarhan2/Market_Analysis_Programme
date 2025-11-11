from app.data_fetching.services.nodes import build_pipeline
graph = build_pipeline()

# save as PNG
png_data = graph.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(png_data)

print("Saved graph as graph.png")
