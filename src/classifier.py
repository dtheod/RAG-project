from src.models import llm

def classify_query(state):
    """Classify user query into categories: cars, countries, math, other."""
    
    query = state["query"]
    
    prompt = f"""
    You are a classification expert that can identify the category from a given query.
    There are five possible categories:
    1. cars - ALL questions related to cars (features, count, statistics, filtering, etc.)
       Examples: "Tell me about Tesla", "How many cars?", "Which cars have rating > 4?"
    2. countries - Questions related to countries (best plays, capitals, languages)
    3. math - Questions related to math (addition, division, square root)
    4. ragas - Questions related to RAG evaluation, performance, or metrics
    5. other - Questions that do not fit into the above categories
    
    Classify this query into one of these categories: cars, countries, math, ragas, other

    Query: {query}

    Respond with just the category name (cars, countries, math, ragas, or other)."""

    response = llm.invoke(prompt)
    category = response.content.strip().lower()
    
    # Ensure valid category
    if category not in ["cars", "countries", "math", "ragas", "other"]:
        category = "other"
    print(f"Category selected: {category}")
    state["category"] = category
    return state
