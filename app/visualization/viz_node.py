# app/shared/market_analysis_tool/visualization_node.py
import matplotlib.pyplot as plt
import pandas as pd
from app.visualization.market_tool import deep_market_analysis
import os
from app.test import PipelineState

def visualization_node(state: PipelineState, **kwargs):
    analysis = deep_market_analysis()
    
    country_data = pd.DataFrame(analysis['country_market_share_topN'])
    plt.figure(figsize=(8, 5))
    plt.bar(country_data['country_name'], country_data['total_revenue_usd'], color='skyblue')
    plt.title('Revenue by Country')
    plt.ylabel('Revenue (USD)')
    plt.xlabel('Country')
    plt.tight_layout()
    os.makedirs('charts', exist_ok=True)
    plt.savefig('charts/country_revenue.png')
    plt.close()


    brand_data = pd.DataFrame(analysis['brand_market_share_topN'])
    plt.figure(figsize=(7, 7))
    plt.pie(brand_data['share'], labels=brand_data['brand_name'], autopct='%1.1f%%', startangle=140)
    plt.title('Top Brands Market Share')
    plt.tight_layout()
    plt.savefig('charts/brand_market_share.png')
    plt.close()

    corr_data = pd.DataFrame(analysis['correlation_matrix'])
    plt.figure(figsize=(8, 6))
    plt.imshow(corr_data, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.xticks(range(len(corr_data.columns)), corr_data.columns, rotation=45, ha='right')
    plt.yticks(range(len(corr_data.columns)), corr_data.columns)
    plt.title('Correlation Heatmap of Metrics')
    plt.tight_layout()
    plt.savefig('charts/correlation_heatmap.png')
    plt.close()

    print("Visualizations saved in 'charts/' folder")

# Run locally
if __name__ == "__main__":
    visualization_node()
