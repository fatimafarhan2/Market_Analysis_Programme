WITH sales_agg AS (
    SELECT * FROM sales_agg_flat
),
reviews_agg AS (
    SELECT * FROM reviews_agg_flat
)
SELECT
    p.product_id,
    p.product_name,
    p.brand_id,
    p.brand_name,
    p.category,
    p.key_function,
    p.price_range,
    c.country_id,
    c.country_name,
    c.region,
    c.currency,
    s.total_units_sold,
    s.total_revenue_usd,
    s.avg_revenue_usd,
    s.total_net_units_sold,
    s.total_net_revenue_usd,
    s.avg_returns_rate,
    s.avg_marketing_spend_usd,
    s.avg_marketing_efficiency_ratio,
    s.avg_return_on_marketing_spend,
    s.avg_online_sales_ratio,
    r.avg_rating,
    r.avg_sentiment_score,
    r.total_num_reviews,
    d.population_millions AS avg_population_millions,
    d.avg_income_usd,
    d.urbanization_rate AS avg_urbanization_rate,
    d.online_shopping_penetration AS avg_online_shopping_penetration,
    r.top_common_keywords,
    s.dominant_distribution_channel
FROM "Products" p
JOIN sales_agg s ON s.product_id = p.product_id
JOIN reviews_agg r ON r.product_id = p.product_id AND r.country_id = s.country_id
JOIN "Countries" c ON c.country_id = s.country_id
JOIN "Demographics_2024" d ON d.country_id = c.country_id

-- filters injected here
