import re
from pathlib import Path

TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "templates"

# Product-level filter keys only
PRODUCT_FILTER_KEYS = {
    "region", "brand_name", "country_name", "currency",
    "category", "product_name_contains", "key_function", "price_range",
    "units_sold", "net_units_sold", "total_units_sold", "revenue_usd",
    "net_revenue_usd", "total_revenue_usd", "avg_revenue_usd", "avg_price_usd",
    "marketing_spend_usd", "marketing_spend_per_unit", "avg_marketing_spend_usd",
    "marketing_efficiency_ratio", "avg_marketing_efficiency_ratio",
    "return_on_marketing_spend", "avg_return_on_marketing_spend",
    "returns_rate", "avg_returns_rate", "return_count_estimate",
    "search_to_sales_ratio", "searched_times", "online_sales_ratio",
    "avg_online_sales_ratio", "is_banned", "verified_purchase",
    "avg_rating", "total_num_reviews", "dominant_distribution_channel",
    "avg_sentiment_score", "population_millions", "avg_population_millions",
    "avg_income_usd", "urbanization_rate", "avg_urbanization_rate",
    "online_shopping_penetration", "avg_online_shopping_penetration",
    "female_population_percent", "year"
}

FILTER_METADATA = {
    "category": ("Categorical", "Product category (e.g., haircare, skincare, bodycare, makeup)"),
    "key_function": ("Categorical", "Product function/purpose (e.g., hydration, repair, anti-aging)"),
    "price_range": ("Categorical", "Price bucket or range (low, medium, high)"),
    "brand_name": ("Categorical", "Filter by specific brand (e.g., Estee Lauder)"),
    "product_name": ("Categorical", "Specific product filter (e.g., Matte Lipstick)"),
    "country_name": ("Categorical", "Filter by country (e.g., Japan, Pakistan)"),
    "region": ("Categorical", "Geographical region (e.g., Asia, Europe)"),
    "currency": ("Categorical", "Currency type (e.g., USD, JPY)"),
    "dominant_distribution_channel": ("Categorical", "Main sales channel (Online, Offline)"),
    "is_banned": ("Boolean", "Filter products that are banned or not (true/false)"),
    "verified_purchase": ("Boolean", "Only consider verified reviews (true/false)"),
    "avg_rating": ("Numeric", "Customer rating (1–5)"),
    "avg_sentiment_score": ("Numeric", "Review sentiment score (negative–positive)"),
    "total_num_reviews": ("Numeric", "Total reviews count"),
    "units_sold": ("Numeric", "Filter by sales volume"),
    "net_units_sold": ("Numeric", "Filter by net sales volume"),
    "total_units_sold": ("Numeric", "Filter by total sales volume"),
    "revenue_usd": ("Numeric", "Filter by revenue range"),
    "net_revenue_usd": ("Numeric", "Filter by net revenue"),
    "total_revenue_usd": ("Numeric", "Filter by total revenue"),
    "avg_revenue_usd": ("Numeric", "Filter by average revenue per product"),
    "avg_price_usd": ("Numeric", "Average product price"),
    "marketing_spend_usd": ("Numeric", "Filter by marketing investment"),
    "marketing_spend_per_unit": ("Numeric", "Marketing spend per unit"),
    "avg_marketing_spend_usd": ("Numeric", "Average marketing spend per product"),
    "marketing_efficiency_ratio": ("Numeric", "Marketing efficiency"),
    "avg_marketing_efficiency_ratio": ("Numeric", "Average marketing efficiency"),
    "return_on_marketing_spend": ("Numeric", "Marketing ROI"),
    "avg_return_on_marketing_spend": ("Numeric", "Average marketing ROI"),
    "returns_rate": ("Numeric", "Product return rate"),
    "avg_returns_rate": ("Numeric", "Average return rate"),
    "return_count_estimate": ("Numeric", "Estimated return count"),
    "search_to_sales_ratio": ("Numeric", "Conversion efficiency (searches vs sales)"),
    "searched_times": ("Numeric", "Number of times product searched"),
    "online_sales_ratio": ("Numeric", "Share of online sales"),
    "avg_online_sales_ratio": ("Numeric", "Average online sales share"),
    "population_millions": ("Numeric", "Country market size"),
    "avg_population_millions": ("Numeric", "Average market size across countries"),
    "avg_income_usd": ("Numeric", "Average income in USD"),
    "urbanization_rate": ("Numeric", "Urban population proportion"),
    "avg_urbanization_rate": ("Numeric", "Average urbanization rate"),
    "online_shopping_penetration": ("Numeric", "Online shopping adoption"),
    "avg_online_shopping_penetration": ("Numeric", "Average online shopping adoption"),
    "female_population_percent": ("Numeric", "Percentage of female population"),
    "year": ("Numeric", "Year of sale")
}

numeric_keys = {k for k, v in FILTER_METADATA.items() if v[0] == "Numeric"}

def sanitize(value: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_ %.-]", "", str(value)).strip()
    return clean.replace("'", "''")

def load_template() -> str:
    path = TEMPLATE_DIR / "product_template.sql"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def inject_filters(sql_template: str, filters: dict) -> str:
    clauses = []

    boolean_keys = {"is_banned", "verified_purchase"}
    fuzzy_keys = {"product_name_contains"}
    categorical_keys = PRODUCT_FILTER_KEYS - numeric_keys - boolean_keys - fuzzy_keys

    column_map = {
        "region": "c.region",
        "brand_name": "p.brand_name",
        "category": "p.category",
        "country_name": "c.country_name",
        "currency": "c.currency",
        "product_name_contains": "p.product_name",
        "key_function": "p.key_function",
        "price_range": "p.price_range",
        "dominant_distribution_channel": "s.dominant_distribution_channel",
    }

    operator_map = {
        ">": ">",
        "gt": ">",
        "<": "<",
        "lt": "<",
        ">=": ">=",
        "gte": ">=",
        "<=": "<=",
        "lte": "<=",
        "=": "=",
        "eq": "=",
        "!=": "!=",
        "neq": "!=",
        "between": "BETWEEN"
    }

    for key, raw_value in filters.items():
        if key not in PRODUCT_FILTER_KEYS or raw_value in (None, ""):
            continue

        col = column_map.get(key, f"s.{key}" if key in numeric_keys else key)

        # ---------- Fuzzy Search ----------
        if key in fuzzy_keys:
            values = [sanitize(v.strip().lower()) for v in str(raw_value).split(",") if v.strip()]
            like_clauses = [f"LOWER({col}) LIKE '%{v.replace(' ', '%')}%'" for v in values]
            clauses.append("(" + " OR ".join(like_clauses) + ")")
            continue
        if key in boolean_keys:
            bool_map = {"true": "TRUE", "false": "FALSE"}
            val = bool_map.get(str(raw_value).lower(), "FALSE")
            clauses.append(f"{col} = {val}")
            continue

        if key in numeric_keys:
            if isinstance(raw_value, dict):

                op = raw_value.get("operator")
                print("==="*100)
                print(f"OPERATOR VALUE:{op}")
                print("==="*100)
                value = raw_value.get("value")
                min_v = raw_value.get("min")
                max_v = raw_value.get("max")

                if min_v is not None and max_v is not None:
                    clauses.append(f"{col} BETWEEN {sanitize(min_v)} AND {sanitize(max_v)}")
                elif op and value is not None:
                    sql_op = operator_map.get(op.lower(), "=")
                    clauses.append(f"{col} {sql_op} {sanitize(value)}")
                continue

            values = [sanitize(v.strip().lower()) for v in str(raw_value).split(",") if v.strip()]
            if len(values) == 1:
                clauses.append(f"{col} = {values[0]}")
            else:
                formatted = ", ".join(values)
                clauses.append(f"{col} IN ({formatted})")
            continue

        # ---------- Categorical Filters ----------
        if key in categorical_keys:
            values = [sanitize(v.strip().lower()) for v in str(raw_value).split(",") if v.strip()]
            formatted = ", ".join(f"'{v}'" for v in values)
            clauses.append(f"LOWER({col}) IN ({formatted})")

    where_sql = "WHERE TRUE"
    if clauses:
        where_sql += " AND " + " AND ".join(clauses)

    return sql_template.replace("-- filters injected here", where_sql)


def generate_sql( filters: dict) -> str:
    template = load_template()
    return inject_filters(template, filters).strip()


