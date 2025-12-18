import pandas as pd
import duckdb
from pathlib import Path
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
    embeds = embeddings.embed_documents(documents)
    
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


def load_cars_to_duckdb(db_path="duckdb_cars.db", csv_path="./data/cars_dataset.csv"):
    """Load cars dataset into DuckDB database."""
    
    # Check if database already exists and has data
    db_file = Path(db_path)
    if db_file.exists():
        conn = duckdb.connect(db_path)
        try:
            count = conn.execute("SELECT COUNT(*) FROM cars").fetchone()[0]
            if count > 0:
                logger.info(f"DuckDB already exists with {count} rows. Skipping load.")
                conn.close()
                return
        except:
            # Table doesn't exist, continue with loading
            pass
        conn.close()
    
    # Load CSV into pandas
    logger.info(f"Loading cars dataset from {csv_path} into DuckDB")
    df = pd.read_csv(csv_path)
    
    # Connect to DuckDB and create table
    conn = duckdb.connect(db_path)
    
    # Register the dataframe and create table
    conn.execute("DROP TABLE IF EXISTS cars")
    conn.execute("""
        CREATE TABLE cars AS 
        SELECT * FROM df
    """)
    
    # Verify the load
    count = conn.execute("SELECT COUNT(*) FROM cars").fetchone()[0]
    logger.info(f"Successfully loaded {count} cars into DuckDB")
    
    conn.close()


def initialize_databases(client):
    """Initialize both ChromaDB vector store and DuckDB database."""
    logger.info("Starting database initialization...")
    
    # Check and initialize ChromaDB
    collections = client.list_collections()
    if collections:
        logger.info("ChromaDB collections already exist")
    else:
        logger.info("Initializing ChromaDB collections...")
        ingest_cars(client)
        ingest_country_documents(client)
    
    # Check and initialize DuckDB
    db_file = Path("duckdb_cars.db")
    if db_file.exists():
        # Check if it has data
        try:
            conn = duckdb.connect("duckdb_cars.db")
            count = conn.execute("SELECT COUNT(*) FROM cars").fetchone()[0]
            conn.close()
            if count > 0:
                logger.info(f"DuckDB already initialized with {count} rows")
            else:
                load_cars_to_duckdb()
        except:
            # Table doesn't exist
            load_cars_to_duckdb()
    else:
        logger.info("Initializing DuckDB database...")
        load_cars_to_duckdb()
    
    logger.info("Database initialization complete!")

