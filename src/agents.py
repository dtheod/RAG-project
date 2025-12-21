from src.models import llm
from src.schemas import (
    FormatAgentSchema,
    ClassifierAgentSchema
)
from src.logger import logger
from langchain.chains import LLMMathChain
from langchain.agents import Tool
from src.retrieval import search_cars_db, search_countries_db

from datasets import Dataset
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

import duckdb
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent, SQLDatabaseToolkit
from sqlalchemy import create_engine
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langfuse import observe, Langfuse
from langfuse.langchain import CallbackHandler

# Initialize Langfuse client
langfuse = Langfuse()
langfuse_handler = CallbackHandler()

# ==================== TOOL FUNCTIONS ====================


@tool
def query_cars_database(question: str) -> str:
    """Query the cars database using SQL for counting, filtering, aggregation, and statistical queries.
    Use this tool for questions like: 'How many cars...', 'Count cars by...', 'List all cars where...', 
    'Which cars have rating > X', 'Average rating by manufacturer', etc.
    
    Args:
        question: Natural language question to convert to SQL and execute
        
    Returns:
        Natural language answer based on SQL query results
    """
    
    logger.info(f"SQL Tool executing query: {question}")
    
    try:
        # Create database engine
        engine = create_engine("duckdb:///duckdb_cars.db")
        
        # Define custom table info
        custom_table_info = {
            "cars": (
                "A table of cars with detailed specifications.\n"
                "- \"Car Name\" (VARCHAR): Name of the car model\n"
                "- \"Manufacturer\" (VARCHAR): Car manufacturer/brand\n"
                "- \"Launch Year\" (INTEGER): Year the car was launched\n"
                "- \"Description\" (VARCHAR): Detailed description of the car\n"
                "- \"Engine Specifications\" (VARCHAR): Engine details (type, HP, cc)\n"
                "- \"Other Specifications\" (VARCHAR): Additional specs (type, fuel efficiency, top speed)\n"
                "- \"User Ratings\" (DOUBLE): User rating score (0-5)\n"
                "- \"NCAP Global Rating\" (INTEGER): Safety rating (1-5)\n"
            ),
        }
        
        # Initialize SQLDatabase
        db = SQLDatabase(
            engine=engine,
            include_tables=list(custom_table_info.keys()),
            custom_table_info=custom_table_info,
        )
        
        # Create SQL agent
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        agent = create_sql_agent(
            toolkit=toolkit,
            llm=llm,
            agent_type="tool-calling",
            verbose=False,
            handle_parsing_errors=True,
            return_intermediate_steps=False,
        )
        
        # Execute query without inheriting callbacks
        response_obj = agent.invoke({"input": question}, config={"callbacks": []})
        response = response_obj.get("output", str(response_obj))
        
        logger.info(f"SQL Tool result: {response[:100]}...")
        return response
        
    except Exception as e:
        logger.error(f"SQL Tool error: {e}")
        return f"Error executing SQL query: {str(e)}"


@tool
def search_cars_vector(query: str) -> str:
    """Search for car information using semantic/vector search. Use this for questions about 
    specific car features, descriptions, specifications, or when you need detailed information 
    about particular car models. Examples: 'Tell me about Tesla Model S', 'What are the safety 
    features of BMW X5?', 'Describe electric cars', etc.
    
    Args:
        query: Search query for semantic matching
        
    Returns:
        Relevant car information from vector database
    """
    logger.info(f"Vector search tool executing: {query}")
    
    try:
        # Get relevant context from vector store
        context_docs, context_metadatas = search_cars_db(query, k=3)
        
        if not context_docs:
            return "No specific car information found in the database."
        
        # Format results
        result = "\n\n".join([
            f"**{meta.get('car_name', 'Unknown')}** ({meta.get('manufacturer', 'Unknown')}):\n{doc}"
            for doc, meta in zip(context_docs, context_metadatas)
        ])
        
        logger.info(f"Vector search found {len(context_docs)} results")
        return result
        
    except Exception as e:
        logger.error(f"Vector search tool error: {e}")
        return f"Error searching car database: {str(e)}"


# ==================== AGENT FUNCTIONS ====================


def cars_agent(state):
    """Handle car-related queries by always using both SQL and vector search tools."""
    query = state["query"]
    logger.info(f"Cars agent (Sequential) processing query: {query}")
    
    try:
        # Always call both tools
        logger.info("Calling SQL database tool...")
        sql_result = query_cars_database.invoke(query)
        
        logger.info("Calling vector search tool...")
        vector_result = search_cars_vector.invoke(query)
        
        # Combine results with LLM
        combine_prompt_object = langfuse.get_prompt("combine_prompt")
        combine_prompt = combine_prompt_object.get_langchain_prompt()
        
        response = llm.invoke(combine_prompt, config={"callbacks": []})
        
        # Set state
        state["response"] = response.content
        state["contexts"] = [
            f"SQL Result: {sql_result[:200]}...",
            f"Vector Result: {vector_result[:200]}..."
        ]
        state["sources"] = ["DuckDB SQL + Vector Search"]
        
        logger.info("Cars agent completed successfully using both tools")
        
    except Exception as e:
        logger.error(f"Cars agent error: {e}")
        # Fallback to simple vector search
        try:
            context_docs, context_metadatas = search_cars_db(query)
            context = "\n".join(context_docs) if context_docs else "No information found."
            
            car_prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(
                    "You are a car expert. Answer based on the context provided."
                ),
                HumanMessagePromptTemplate.from_template(
                    "Context: {context}\n\nQuestion: {query}\n\nProvide a helpful answer."
                ),
            ])
            
            formatted_prompt = car_prompt.format_messages(context=context, query=query)
            llm_response = llm.invoke(formatted_prompt, config={"callbacks": []})
            
            state["response"] = llm_response.content
            state["contexts"] = context_docs
            state["sources"] = [meta.get('source', 'Unknown') for meta in context_metadatas]
            
        except Exception as fallback_error:
            logger.error(f"Fallback also failed: {fallback_error}")
            state["response"] = f"Error processing query: {str(e)}"
            state["contexts"] = []
            state["sources"] = []
    
    return state

def format_agent(state):
    """Format the response for better readability."""
    query = state["query"]


    format_prompt_object = langfuse.get_prompt("format_prompt", version=3)
    format_prompt = format_prompt_object.get_langchain_prompt()
    
    format_prompt = ChatPromptTemplate.from_template(format_prompt)

    # Use structured output
    structured_llm = llm.with_structured_output(FormatAgentSchema)
    chain = format_prompt | structured_llm
    
    # Invoke chain
    response = chain.invoke({"query": query}, config={"callbacks": []})
    
    logger.info(f"Formatted response: {response.formatted_query}")
    state["formatted_query"] = response.formatted_query
    return state


def countries_agent(state):
    """Handle country-related queries."""
    query = state["query"]
    
    # Get relevant context from vector store
    context_docs, context_metadatas = search_countries_db(query)
    context = "\n".join(context_docs) if context_docs else "No specific country information found."
    state["contexts"] = context_docs
    state["sources"] = [meta.get('source', 'Unknown') for meta in context_metadatas]
    
    country_prompt_object = langfuse.get_prompt("countries_prompt")
    country_prompt = country_prompt_object.get_langchain_prompt()
    
    # Format the prompt with context and query
    response = llm.invoke(country_prompt, config={"callbacks": []})
    state["response"] = response.content
    return state

def math_agent(state):
    """Handle math-related queries."""
    query = state["query"]
    
    math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
    
    # The LLMMathChain expects a dictionary with a "question" key
    response = math_chain.invoke({"question": query})
    
    # Handle different response formats
    if isinstance(response, dict):
        state["response"] = response.get('answer') or response.get('result', str(response))
    else:
        state["response"] = str(response)
    
    return state

def sql_agent(state):
    """Handle SQL/database queries using DuckDB with LangChain SQL Agent."""

    
    query = state["query"]
    logger.info(f"SQL Agent processing query: {query}")
    
    # Create database engine for DuckDB
    engine = create_engine("duckdb:///duckdb_cars.db")
    
    # Define custom table info for better LLM context
    custom_table_info = {
        "cars": (
            "A table of cars with detailed specifications.\n"
            "- \"Car Name\" (VARCHAR): Name of the car model\n"
            "- \"Manufacturer\" (VARCHAR): Car manufacturer/brand\n"
            "- \"Launch Year\" (INTEGER): Year the car was launched\n"
            "- \"Description\" (VARCHAR): Detailed description of the car\n"
            "- \"Engine Specifications\" (VARCHAR): Engine details (type, HP, cc)\n"
            "- \"Other Specifications\" (VARCHAR): Additional specs (type, fuel efficiency, top speed)\n"
            "- \"User Ratings\" (DOUBLE): User rating score (0-5)\n"
            "- \"NCAP Global Rating\" (INTEGER): Safety rating (1-5)\n"
        ),
    }
    
    # Initialize SQLDatabase with custom info
    db = SQLDatabase(
        engine=engine,
        include_tables=list(custom_table_info.keys()),
        custom_table_info=custom_table_info,
    )
    
    # Create toolkit and agent
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    agent = create_sql_agent(
        toolkit=toolkit,
        llm=llm,
        agent_type="tool-calling",
        verbose=False,
        handle_parsing_errors=True,
        return_intermediate_steps=False,  # Disable to avoid callback tracing issues
    )
    
    # Execute query through agent (without callbacks to avoid tracing issues)
    try:
        # Invoke agent without inheriting callbacks from the workflow
        response_obj = agent.invoke({"input": query}, config={"callbacks": []})
        response = response_obj.get("output", str(response_obj))
        
        # Set state with response and sources
        state["response"] = response
        state["contexts"] = [f"SQL Agent Query\nResponse: {response}"]
        state["sources"] = ["DuckDB SQL Agent"]
        
    except Exception as e:
        logger.error(f"SQL Agent execution error: {e}")
        state["response"] = f"I encountered an error executing the SQL query: {str(e)}"
        state["contexts"] = [f"Error: {str(e)}"]
        state["sources"] = ["DuckDB SQL Agent (Failed)"]
    
    return state


def general_agent(state):
    """Handle general queries."""
    query = state["query"]
    
    # Get prompt from Langfuse
    general_prompt_object = langfuse.get_prompt("general_agent")
    general_prompt = general_prompt_object.get_langchain_prompt()
    
    
    response = llm.invoke(general_prompt, config={"callbacks": []})
    state["response"] = response.content
    return state

def classifier_agent(state):
    """Classify user query into categories: cars, countries, math, other."""
    
    query = state["query"]
    formatted_query = state["formatted_query"]
    
    classifier_prompt_object = langfuse.get_prompt("classifier_prompt", version=4)
    classifier_prompt = classifier_prompt_object.get_langchain_prompt()

    classifier_prompt = ChatPromptTemplate.from_template(classifier_prompt)

    structured_llm = llm.with_structured_output(ClassifierAgentSchema)
    chain = classifier_prompt | structured_llm
    
    response = chain.invoke({"formatted_query": formatted_query}, config={"callbacks": []})
    
    # Ensure valid category
    if response.category not in ["cars", "countries", "math", "other"]:
        response.category = "other"
    state["category"] = response.category
    state["classifier_reason"] = response.reason
    return state


# def ragas_agent(state):
#     """Perform online evaluation of the RAG query."""
#     logger.info("---PERFORMING RAGAS EVALUATION---")
    
#     query = state["query"]
#     answer = state["answer"]
#     contexts = state["contexts"]

#     # Create a dataset from the online data
#     response_dataset = Dataset.from_dict({
#         "question": [query],
#         "answer": [answer],
#         "contexts": [contexts],
#     })

#     # Evaluate the dataset
#     result = evaluate(
#         response_dataset,
#         metrics=[
#             faithfulness,
#             answer_relevancy,
#         ],
#     )

#     logger.info(f"RAGAS Evaluation Results: {result}")
    
#     # This agent only performs evaluation and does not change the response
#     return state