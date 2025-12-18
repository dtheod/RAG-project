from src.ingestion import initialize_databases
from src.workflow import app, process_query
from langchain.schema.runnable.config import RunnableConfig
import chromadb
import chainlit as cl


def initialize():
    """Initialize vector store and DuckDB database"""
    client = chromadb.PersistentClient(path="./chroma_db")
    initialize_databases(client)
    print("Database initialization complete.")

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
    
    
    
    # Get response and sources
    response = result["response"]
    sources = result.get("sources", [])
    contexts = result.get("contexts", [])
    
    
    # Send the response back to the UI
    await cl.Message(content=response).send()
    
    # Display sources in an expandable step
    if sources and contexts:
        async with cl.Step(name="Sources") as step:
            # step.input = "Retrieved Contexts"
            source_text = ""
            for idx, (source, context) in enumerate(zip(sources, contexts)):
                source_text += f"### {idx+1}. {source}\n{context}\n\n"
            step.output = source_text
            await step.send()

 
if __name__ == "__main__":
    initialize()
    query = "List of cars Launched late than 2012?"
    run_query(query)
