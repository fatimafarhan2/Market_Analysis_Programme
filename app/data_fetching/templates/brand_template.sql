WITH sales_agg AS (
    SELECT
        s.brand_id,
        SUM(s.units_sold) AS total_units_sold,
        SUM(s.revenue_usd) AS total_revenue_usd,
        SUM(s.net_units_sold) AS total_net_units_sold,
        SUM(s.net_revenue_usd) AS total_net_revenue_usd,
        AVG(s.returns_rate) AS avg_returns_rate,
        AVG(s.marketing_efficiency_ratio) AS avg_marketing_efficiency_ratio,
        AVG(s.return_on_marketing_spend) AS avg_return_on_marketing_spend,
        AVG(s.online_sales_ratio) AS avg_online_sales_ratio
    FROM (
        SELECT * FROM "Sales_2023"
        UNION ALL
        SELECT * FROM "Sales_2024"
        UNION ALL
        SELECT * FROM "Sales_2025"
    ) s
    GROUP BY s.brand_id
),

reviews_agg AS (
    SELECT
        r.brand_id,
        AVG(r.avg_rating) AS avg_rating,
        AVG(r.sentiment_score) AS avg_sentiment_score,
        SUM(r.num_reviews) AS total_num_reviews
    FROM (
        SELECT * FROM "Reviews_2023"
        UNION ALL
        SELECT * FROM "Reviews_2024"
        UNION ALL
        SELECT * FROM "Reviews_2025"
    ) r
    GROUP BY r.brand_id
),

demographics_agg AS (
    SELECT
        s.brand_id,
        AVG(d.population_millions) AS avg_population_millions,
        AVG(d.avg_income_usd) AS avg_income_usd,
        AVG(d.urbanization_rate) AS avg_urbanization_rate,
        AVG(d.online_shopping_penetration) AS avg_online_shopping_penetration
    FROM (
        SELECT * FROM "Sales_2023"
        UNION ALL
        SELECT * FROM "Sales_2024"
        UNION ALL
        SELECT * FROM "Sales_2025"
    ) s
    JOIN (
        SELECT * FROM "Demographics_2023"
        UNION ALL
        SELECT * FROM "Demographics_2024"
        UNION ALL
        SELECT * FROM "Demographics_2025"
    ) d ON d.country_id = s.country_id
    GROUP BY s.brand_id
),

country_agg AS (
    SELECT 
        s.brand_id,
        STRING_AGG(DISTINCT c.country_name, ', ') AS country_name,
        STRING_AGG(DISTINCT c.region, ', ') AS region,
        STRING_AGG(DISTINCT c.currency, ', ') AS currency
    FROM (
        SELECT * FROM "Sales_2023"
        UNION ALL
        SELECT * FROM "Sales_2024"
        UNION ALL
        SELECT * FROM "Sales_2025"
    ) s
    JOIN "Countries" c ON c.country_id = s.country_id
    GROUP BY s.brand_id
)

SELECT
    b.brand_id,
    b.brand_name,
    b.headquarters_country,
    b.main_market,
    b.brand_tier,
    b.website,
    s.total_units_sold,
    s.total_revenue_usd,
    s.total_net_units_sold,
    s.total_net_revenue_usd,
    s.avg_returns_rate,
    s.avg_marketing_efficiency_ratio,
    s.avg_return_on_marketing_spend,
    s.avg_online_sales_ratio,
    r.avg_rating,
    r.avg_sentiment_score,
    r.total_num_reviews,
    b.avg_global_rating,
    b.sustainability_index,
    d.avg_population_millions,
    d.avg_income_usd,
    d.avg_urbanization_rate,
    d.avg_online_shopping_penetration,
    c.region,
    c.country_name,
    c.currency
FROM "Brands" b
LEFT JOIN sales_agg s ON s.brand_id = b.brand_id
LEFT JOIN reviews_agg r ON r.brand_id = b.brand_id
LEFT JOIN demographics_agg d ON d.brand_id = b.brand_id
LEFT JOIN country_agg c ON c.brand_id = b.brand_id

-- filters injected here
