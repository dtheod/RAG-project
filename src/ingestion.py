import pandas as pd
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from src.models import embeddings
from src.logger import logger
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction


def create_car_document(row):
    """Create a well-structured document for better semantic search."""
    # Format the document in a more natural, searchable way
    doc_text = f"""
                {row['Car Name']} - {row['Manufacturer']} ({row['Launch Year']})

                {row['Description']}

                Technical Specifications:
                - Engine: {row['Engine Specifications']}
                - Performance: {row['Other Specifications']}
                - User Rating: {row['User Ratings']}/5.0 stars
                - Safety Rating: {row['NCAP Global Rating']}/5 NCAP stars
                - Manufacturer: {row['Manufacturer']}
                - Launch Year: {row['Launch Year']}
                """.strip()
                    
    return doc_text

def ingest_cars(client):
    """Elegant ingestion of cars dataset with improved text formatting."""
    
    # Read the CSV file
    df = pd.read_csv('./data/cars_dataset.csv')
    logger.info(f"Loading {len(df)} cars from CSV")
    
    # Create new collection or getting existing one
    collection = client.get_or_create_collection(
        name = "cars",
        embedding_function=OpenAIEmbeddingFunction(
        model_name="text-embedding-3-small"
        )
    )
       
    # Prepare documents for ingestion
    documents = []
    metadatas = []
    ids = []
    
    for idx, row in df.iterrows():
        # Create well-formatted document
        doc_text = create_car_document(row)
        documents.append(doc_text)
        
        # Store essential metadata
        metadata = {
            'car_name': str(row['Car Name']),
            'manufacturer': str(row['Manufacturer']),
            'launch_year': int(row['Launch Year']),
            'user_rating': float(row['User Ratings']),
            'safety_rating': int(row['NCAP Global Rating']),
            'source': 'cars_dataset.csv'
        }
        metadatas.append(metadata)
        ids.append(f"car_{idx}")
    
    # Generate embeddings and add to collection
    print("Generating embeddings...")
    embeds = embeddings.embed_documents(documents)
    
    print("Adding documents to ChromaDB...")
    collection.upsert(
        embeddings=embeds,
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
   
    logger.info(f"Successfully ingested {len(documents)} cars into ChromaDB!")
    return collection


def ingest_country_documents(client):
    """Ingest country documents into ChromaDB."""

    # Create new collection or getting existing one
    collection = client.get_or_create_collection(
        name = "countries",
        embedding_function=OpenAIEmbeddingFunction(
        model_name="text-embedding-3-small"
        )
    )

    doc = DocumentConverter().convert(source="./data/country_data.md").document
    chunker = HybridChunker()
    chunk_iter = chunker.chunk(dl_doc=doc)

    texts = [chunker.contextualize(chunk=chunk) for chunk in chunk_iter]
    ids = [f"doc_{i}" for i in range(len(texts))]
    collection.upsert(
        documents=texts,
        metadatas=[{"source": "country_data.md"}]*len(texts),
        ids=ids
    )
    logger.info(f"Successfully ingested {len(texts)} country documents into ChromaDB!")
    return collection


def vector_store_setup(client):
    """Set up the vector store by ingesting datasets."""
    collections = client.list_collections()
    print(collections)
    if collections:
        logger.info("Collections are already set")
    else:
        logger.info("Collections are not set and need to be initialised")
        ingest_cars(client)
        ingest_country_documents(client)
    return None

