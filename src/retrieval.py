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

    
    # Return documents and metadata for the agent to use
    if results['documents']:
        return results['documents'][0], results['metadatas'][0]
    return [], []


def search_countries_db(query, k=5):
    """Search the countries database."""
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("countries")
    
    results = collection.query(
        query_texts=[query],
        n_results=k
    )
    logger.info(f"Search results: {results}")
    
    if results['documents']:
        return results['documents'][0], results['metadatas'][0]
    return [], []