from src.models import llm

def classify_query(state):
    """Classify user query into categories: cars, countries, math, other."""
    
    query = state["query"]
    
    prompt = f"""
    You are a classification expert that can identify the category from a given query.
    There are three four possible categories:
    1. cars - Questions related to cars(manufacturers, Launch year, Engine, ncap rating)
    2. countries - Questions related to countrie(best plays, capitals, languages)
    3. math - Questions related to math(addition, dividion, square root)
    4. other - Questions that do not fit into the above categories
    Classify this query into one of these categories: cars, countries, math, other

    Query: {query}

    Respond with just the category name (cars, countries, math, or other)."""

    response = llm.invoke(prompt)
    category = response.content.strip().lower()
    
    # Ensure valid category
    if category not in ["cars", "countries", "math", "other"]:
        category = "other"
    print(f"Category selected: {category}")
    state["category"] = category
    return state
