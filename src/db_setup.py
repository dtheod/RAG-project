import duckdb
import pandas as pd
from pathlib import Path
from src.logger import logger


def load_cars_to_duckdb(db_path="duckdb_cars.db", csv_path="./data/cars_dataset.csv"):
    """Load cars dataset into DuckDB database."""
    
    # Check if database already exists and has data
    db_file = Path(db_path)
    if db_file.exists():
        conn = duckdb.connect(db_path)
        try:
            count = conn.execute("SELECT COUNT(*) FROM cars").fetchone()[0]
            if count > 0:
                logger.info(f"Database already exists with {count} rows. Skipping load.")
                conn.close()
                return
        except:
            # Table doesn't exist, continue with loading
            pass
        conn.close()
    
    # Load CSV into pandas
    logger.info(f"Loading cars dataset from {csv_path}")
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
    
    # Show schema
    schema = conn.execute("DESCRIBE cars").fetchall()
    logger.info(f"Table schema: {schema}")
    
    conn.close()


if __name__ == "__main__":
    load_cars_to_duckdb()
