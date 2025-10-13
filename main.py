from src.ingestion import vector_store_setup
from src.workflow import app, process_query
from langchain.schema.runnable.config import RunnableConfig
import chromadb
import chainlit as cl


def initialize():
    """Ingest documents"""
    client = chromadb.PersistentClient(path="./chroma_db")
    print("passing")
    vector_store_setup(client)
    print("Ingestion complete.")

def run_query(query):
    """Run a single query."""
    result = process_query(query)
    return result["response"]

def evaluation():
    "Run the evaluation queries"

    test_queries = [
        "What is capital of Oslo?"
    ]
    
    for query in test_queries:
        try:
            print("\nQuery: ", query) 
            print("-----------------------")
            response = run_query(query)
            print(f"\nAnswer: {response}\n")
        except Exception as e:
            print(f"Error: {e}\n")


# Chainlit integration
@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming messages from Chainlit UI."""
    # Create a callback handler for LangChain tracing
    cb = cl.LangchainCallbackHandler()
    
    # Configure the runnable with callbacks
    config = RunnableConfig(callbacks=[cb])
    
    # Process the query through the workflow
    result = app.invoke({"query": message.content}, config=config)
    
    # Send the response back to the UI
    await cl.Message(content=result["response"]).send()

 
if __name__ == "__main__":
    initialize()
    evaluation()
