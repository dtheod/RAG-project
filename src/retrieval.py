import chromadb
from src.models import embeddings
from src.logger import logger


def search_cars_db(query, k=5):
    """Search the cars database and return formatted results."""
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("cars")
        
    results = collection.query(
        query_texts=[query],
        n_results=k
    )
    logger.info(f"Search results: {results}")

    
    # Return just the documents for the agent to use
    return results['documents'][0] if results['documents'] else []


def search_countries_db(query, k=5):
    """Search the countries database."""
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("countries")
    
    results = collection.query(
        query_texts=[query],
        n_results=k
    )
    logger.info(f"Search results: {results}")
    
    return results['documents'][0] if results['documents'] else []