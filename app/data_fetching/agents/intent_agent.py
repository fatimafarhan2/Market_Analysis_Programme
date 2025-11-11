import re
import json
from google import genai
from dotenv import load_dotenv
import os
import difflib
from app.data_fetching.agents.sql_agent import FILTER_METADATA, PRODUCT_FILTER_KEYS

load_dotenv()
client = genai.Client()

# Common reference lists for correction
VALID_PRODUCTS = [
    "foundation", "lipstick", "toner", "balm", "shampoo", "conditioner",
    "cleanser", "moisturizer", "lotion", "serum", "micellar water",
    "mask", "face wash", "cream", "mascara", "eyeliner", "eyeshadow",
    "hair serum", "hair mask", "hair oil", "body wash"
]

VALID_BRANDS = [
    "L'Oreal", "Maybelline", "Dior", "Chanel", "Estee Lauder", "Clinique",
    "Shiseido", "Nivea", "Olay", "Neutrogena", "Lancome", "The Body Shop",
    "Revlon", "MAC Cosmetics", "Huda Beauty", "Fenty Beauty", "Garnier",
    "Pond's", "Kiehl's", "Tatcha", "Bioderma", "La Mer", "Innisfree",
    "Etude House", "The Ordinary", "Drunk Elephant", "Lakme", "Sunsilk",
    "Fair & Lovely", "Herbal Essences", "Dove", "Lush", "O.P.I", "Missha",
    "Saeed Ghani", "Hemani"
]

# THESE are the ONLY valid categories in your database schema
# Update this list based on your actual Products table category column values
VALID_CATEGORIES = ["haircare", "skincare", "makeup", "bodycare"]

def correct_spelling(value: str, reference_list):
    match = difflib.get_close_matches(value.lower(), [v.lower() for v in reference_list], n=1, cutoff=0.85)
    return match[0] if match else value

def fix_filter_spellings(filters: dict):
    corrected, corrections = {}, {}
    for key, val in filters.items():
        if not val:
            continue

        if key in ["product_name", "product_name_contains"]:
            items = [v.strip() for v in val.split(",")]
            fixed = []
            for item in items:
                correction = correct_spelling(item, VALID_PRODUCTS)
                fixed.append(correction)
                if correction != item:
                    corrections[item] = correction
            corrected[key] = ", ".join(set(fixed))

        elif key in ["brand_name"]:
            items = [v.strip() for v in val.split(",")]
            fixed = []
            for item in items:
                correction = correct_spelling(item, VALID_BRANDS)
                fixed.append(correction)
                if correction != item:
                    corrections[item] = correction
            corrected[key] = ", ".join(set(fixed))
        else:
            corrected[key] = val

    return corrected, corrections

def detect_intent(user_query: str, previous_filters: dict = None, previous_error: str = None, retry: bool = False):
    """
    Detects product-level intent and filters. 
    Enhanced for loop-back scenarios (when validation or execution fails).
    """

    # Context from previous attempts
    previous_filters_json = json.dumps(previous_filters or {}, indent=2)
    previous_error_text = f"Previous query issue: {previous_error}" if previous_error else ""
    retry_text = (
        "NOTE: This is a retry after a failed attempt. Adjust filters or clarify values intelligently."
        if retry else ""
    )

    prompt = f"""
You are an expert AI data analyst for a cosmetics market opportunity analyzer.

User said: "{user_query}"

{previous_error_text}
{retry_text}
Previous filters (if any): {previous_filters_json}

**CRITICAL: Understanding the Database Schema**

The database has these structures:
1. **Categories** (broad classification): {', '.join(VALID_CATEGORIES)}
   - These are stored in the "category" column of the Products table
   
2. **Product Names** (specific products): shampoo, lipstick, toner, conditioner, etc.
   - These are stored in the "product_name" column as free text
   - Examples: "Total Repair 5 Shampoo - LAN627", "Matte Lipstick Red"

**FILTER SELECTION LOGIC:**

1. **Use "category" filter ONLY when user mentions these exact terms**: {', '.join(VALID_CATEGORIES)}
   Example: "show me haircare products" → {{"category": "haircare"}}

2. **Use "product_name_contains" for ALL specific product types**:
   - User says "shampoo" → {{"product_name_contains": "shampoo"}}
   - User says "lipstick and mascara" → {{"product_name_contains": "lipstick, mascara"}}
   - User says "moisturizing lotion" → {{"product_name_contains": "moisturizing lotion"}}
   
3. **Why this matters**: 
   - "shampoo" is NOT a category value in the database (it would be "haircare")
   - "shampoo" appears in product names like "Total Repair 5 Shampoo"
   - Searching category='shampoo' returns ZERO results
   - Searching product_name LIKE '%shampoo%' returns actual products

**Rule of thumb**: 
- If user mentions a SPECIFIC product type → product_name_contains
- If user mentions a BROAD category from [{', '.join(VALID_CATEGORIES)}] → category

**Other filters:**
- Geographic: Use "country_name" for countries, "region" for regions
- Numeric: Use {{"operator": ">", "value": X}} or {{"min": X, "max": Y}}
- Brands: Use "brand_name"

**Available filter keys**: {list(PRODUCT_FILTER_KEYS)}

Task:
- Extract structured filters following the logic above
- Correct ONLY obvious spelling mistakes (typos), NOT phonetically similar words
- Preserve the user's intended product (e.g., "perfume" should stay "perfume", not become "serum")
- Be consistent: same query should always produce same filters

Response format (strict JSON only):
{{
  "level": "product",
  "filters": {{
      "product_name_contains": "shampoo",
      "country_name": "Japan"
  }},
  "corrections": {{}},
  "message": "Applied product_name_contains for specific product 'shampoo'"
}}

**EXAMPLES:**
Query: "shampoo in Uganda"
✓ CORRECT: {{"product_name_contains": "shampoo", "country_name": "Uganda"}}
✗ WRONG: {{"category": "shampoo", "country_name": "Uganda"}}

Query: "haircare products in Asia"  
✓ CORRECT: {{"category": "haircare", "region": "Asia"}}

Query: "lipstick and toner in France"
✓ CORRECT: {{"product_name_contains": "lipstick, toner", "country_name": "France"}}

Query: "skincare in Japan and France"
✓ CORRECT: {{"category": "skincare", "country_name": "Japan, France"}}
"""

    try:
        response = client.models.generate_content(
            model=os.getenv("MODEL_NAME"),
            contents=prompt
        )
        raw = response.text or ""
        clean = re.sub(r"^```json\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
        parsed = json.loads(clean)

        # SMART POST-PROCESSING: Validate and auto-correct if needed
        filters = parsed.get("filters", {})
        
        if "category" in filters:
            category_values = [v.strip().lower() for v in str(filters["category"]).split(",")]
            
            # Check each value - if it's not a valid category, move it to product_name_contains
            invalid_categories = []
            valid_categories = []
            
            for cat_val in category_values:
                if cat_val not in [c.lower() for c in VALID_CATEGORIES]:
                    invalid_categories.append(cat_val)
                else:
                    valid_categories.append(cat_val)
            
            if invalid_categories:
                print(f"[IntentAgent] Auto-correcting: '{', '.join(invalid_categories)}' are not valid categories")
                print(f"[IntentAgent] Valid categories are: {VALID_CATEGORIES}")
                print(f"[IntentAgent] Moving to product_name_contains instead")
                
                # Move invalid ones to product_name_contains
                existing_products = filters.get("product_name_contains", "")
                if existing_products:
                    invalid_categories.insert(0, existing_products)
                filters["product_name_contains"] = ", ".join(invalid_categories)
                
                # Keep only valid categories
                if valid_categories:
                    filters["category"] = ", ".join(valid_categories)
                else:
                    filters.pop("category")
                
                parsed["message"] = f"Auto-corrected: moved '{', '.join(invalid_categories)}' to product_name_contains (not valid categories)"

        parsed["filters"], corrections = fix_filter_spellings(filters)
        parsed["corrections"] = corrections
        return parsed

    except json.JSONDecodeError:
        print("[IntentAgent] JSON decode failed, retrying fallback...")
        fallback_prompt = f"""
User query: "{user_query}"

CRITICAL RULES:
- Valid categories in database: {', '.join(VALID_CATEGORIES)}
- ONLY use "category" filter for these exact terms
- For any other product type (shampoo, lipstick, etc.) use "product_name_contains"

Previous filters: {previous_filters_json}
Previous error: {previous_error_text}

Output strict JSON with keys: level, filters, corrections, message.

Example: "shampoo in Japan" should give:
{{"level": "product", "filters": {{"product_name_contains": "shampoo", "country_name": "Japan"}}, "corrections": {{}}, "message": ""}}
"""
        try:
            response = client.models.generate_content(
                model=os.getenv("MODEL_NAME"),
                contents=fallback_prompt
            )
            raw = response.text or ""
            clean = re.sub(r"^```json\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
            parsed = json.loads(clean)
            
            # Apply same validation
            filters = parsed.get("filters", {})
            if "category" in filters:
                category_values = [v.strip().lower() for v in str(filters["category"]).split(",")]
                invalid = [c for c in category_values if c not in [cat.lower() for cat in VALID_CATEGORIES]]
                
                if invalid:
                    filters["product_name_contains"] = ", ".join(invalid)
                    valid = [c for c in category_values if c in [cat.lower() for cat in VALID_CATEGORIES]]
                    if valid:
                        filters["category"] = ", ".join(valid)
                    else:
                        filters.pop("category")
            
            parsed["filters"], parsed["corrections"] = fix_filter_spellings(filters)
            return parsed
        except Exception as e:
            print("[IntentAgent] Fallback failed:", str(e))
            return {"level": "product", "filters": {}, "corrections": {}, "message": str(e)}

    except Exception as e:
        print("[IntentAgent] Error:", str(e))
        return {"level": "product", "filters": {}, "corrections": {}, "message": str(e)}

# import re
# import json
# from google import genai
# from dotenv import load_dotenv
# import os
# import difflib
# from app.data_fetching.agents.sql_agent import FILTER_METADATA, PRODUCT_FILTER_KEYS

# load_dotenv()
# client = genai.Client()

# # Common reference lists for correction
# VALID_PRODUCTS = [
#     "foundation", "lipstick", "toner", "balm", "shampoo", "conditioner",
#     "cleanser", "moisturizer", "lotion", "serum", "micellar water",
#     "mask", "face wash", "cream"
# ]

# VALID_BRANDS = [
#     "L'Oreal", "Maybelline", "Dior", "Chanel", "Estee Lauder", "Clinique",
#     "Shiseido", "Nivea", "Olay", "Neutrogena", "Lancome", "The Body Shop",
#     "Revlon", "MAC Cosmetics", "Huda Beauty", "Fenty Beauty", "Garnier",
#     "Pond's", "Kiehl's", "Tatcha", "Bioderma", "La Mer", "Innisfree",
#     "Etude House", "The Ordinary", "Drunk Elephant", "Lakme", "Sunsilk",
#     "Fair & Lovely", "Herbal Essences", "Dove", "Lush", "O.P.I", "Missha",
#     "Saeed Ghani", "Hemani"
# ]

# def correct_spelling(value: str, reference_list):
#     match = difflib.get_close_matches(value.lower(), [v.lower() for v in reference_list], n=1, cutoff=0.6)
#     return match[0] if match else value

# def fix_filter_spellings(filters: dict):
#     corrected, corrections = {}, {}
#     for key, val in filters.items():
#         if not val:
#             continue

#         if key in ["product_name", "product_name_contains"]:
#             items = [v.strip() for v in val.split(",")]
#             fixed = []
#             for item in items:
#                 correction = correct_spelling(item, VALID_PRODUCTS)
#                 fixed.append(correction)
#                 if correction != item:
#                     corrections[item] = correction
#             corrected[key] = ", ".join(set(fixed))

#         elif key in ["brand_name"]:
#             items = [v.strip() for v in val.split(",")]
#             fixed = []
#             for item in items:
#                 correction = correct_spelling(item, VALID_BRANDS)
#                 fixed.append(correction)
#                 if correction != item:
#                     corrections[item] = correction
#             corrected[key] = ", ".join(set(fixed))
#         else:
#             corrected[key] = val

#     return corrected, corrections

# def detect_intent(user_query: str, previous_filters: dict = None, previous_error: str = None, retry: bool = False):
#     """
#     Detects product-level intent and filters. 
#     Enhanced for loop-back scenarios (when validation or execution fails).
#     """

#     # Context from previous attempts
#     previous_filters_json = json.dumps(previous_filters or {}, indent=2)
#     previous_error_text = f"Previous query issue: {previous_error}" if previous_error else ""
#     retry_text = (
#         "NOTE: This is a retry after a failed attempt. Adjust filters or clarify values intelligently."
#         if retry else ""
#     )

#     prompt = f"""
#         You are an expert AI data analyst for a cosmetics market opportunity analyzer.

#         User said: "{user_query}"

#         {previous_error_text}
#         {retry_text}
#         Previous filters (if any): {previous_filters_json}

#         Task:
#         - Extract structured filters (only from {list(PRODUCT_FILTER_KEYS)})
#         - Always treat the query as product-level.
#         - Correct spelling mistakes automatically for known products and brands.
#         - Use JSON-safe filter structures for numeric conditions, for example:
#           "above 5 million" → {{"operator": ">", "value": 5000000}}
#           "between 2 and 5" → {{"min": 2, "max": 5}}

#         If this is a retry (previous query failed or no results):
#         - Re-analyze and refine previous filters.
#         - Suggest valid regions, brands, or product names.
#         - Include a "message" key explaining how filters were adjusted.

#         Response format (strict JSON only):
#         {{
#           "level": "product",
#           "filters": {{
#               "region": "europe",
#               "product_name_contains": "lipstick, toner"
#           }},
#           "corrections": {{
#               "lipstik": "lipstick"
#           }},
#           "message": "Previous query returned no data, so region changed to Europe based on trend data."
#         }}
#     """

#     try:
#         response = client.models.generate_content(
#             model=os.getenv("MODEL_NAME"),
#             contents=prompt
#         )
#         raw = response.text or ""
#         clean = re.sub(r"^```json\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
#         parsed = json.loads(clean)

#         parsed["filters"], parsed["corrections"] = fix_filter_spellings(parsed.get("filters", {}))
#         return parsed

#     except json.JSONDecodeError:
#         print("[IntentAgent] JSON decode failed, retrying fallback...")
#         fallback_prompt = f"""
# User query: "{user_query}"
# Previous filters: {previous_filters_json}
# Previous error: {previous_error_text}
# If retry=True, refine previous filters to valid product/brand names.
# Output strict JSON with keys: level, filters, corrections, message.
# """
#         try:
#             response = client.models.generate_content(
#                 model=os.getenv("MODEL_NAME"),
#                 contents=fallback_prompt
#             )
#             raw = response.text or ""
#             clean = re.sub(r"^```json\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
#             parsed = json.loads(clean)
#             parsed["filters"], parsed["corrections"] = fix_filter_spellings(parsed.get("filters", {}))
#             return parsed
#         except Exception as e:
#             print("[IntentAgent] Fallback failed:", str(e))
#             return {"level": "product", "filters": {}, "corrections": {}, "message": str(e)}

#     except Exception as e:
#         print("[IntentAgent] Error:", str(e))
#         return {"level": "product", "filters": {}, "corrections": {}, "message": str(e)}
