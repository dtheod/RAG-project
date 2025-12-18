from src.models import llm
from src.logger import logger
from langchain.chains import LLMMathChain
from langchain.agents import Tool
from langchain.prompts import ChatPromptTemplate
from src.retrieval import search_cars_db, search_countries_db
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from datasets import Dataset
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.chains import create_sql_query_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
import duckdb
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent, SQLDatabaseToolkit
from sqlalchemy import create_engine
from langchain_core.tools import tool


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
    from langchain_community.utilities import SQLDatabase
    from langchain_community.agent_toolkits import create_sql_agent, SQLDatabaseToolkit
    from sqlalchemy import create_engine
    
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
    """Handle car-related queries using ReAct agent with SQL and vector search tools."""
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain_core.prompts import PromptTemplate
    
    query = state["query"]
    logger.info(f"Cars agent (ReAct) processing query: {query}")
    
    # Define available tools
    tools = [query_cars_database, search_cars_vector]
    
    # Create ReAct prompt
    react_prompt = PromptTemplate.from_template(
        """You are a car information expert with access to two tools:
        1. query_cars_database: For counting, filtering, statistics (e.g., "How many cars...", "List cars with rating > X")
        2. search_cars_vector: For detailed information about specific cars (e.g., "Tell me about Tesla Model S")
        
        Answer the following question as best you can. You have access to the following tools:

        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Begin!

        Question: {input}
        Thought: {agent_scratchpad}"""
    )
    
    # Create ReAct agent
    agent = create_react_agent(llm, tools, react_prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=5,
    )
    
    # Execute agent without callbacks to avoid tracing issues
    try:
        result = agent_executor.invoke(
            {"input": query},
            config={"callbacks": []}
        )
        
        response = result.get("output", str(result))
        
        # Set state
        state["response"] = response
        state["contexts"] = [f"Tools used: {', '.join([t.name for t in tools])}"]
        state["sources"] = ["ReAct Agent (SQL + Vector Search)"]
        
        logger.info(f"Cars agent completed successfully")
        
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
            llm_response = llm.invoke(formatted_prompt)
            
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


    format_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are an expert at formatting text to be clear and concise. "
            "Fix any grammar or spelling issues. "
            "DO NOT change the meaning of the query leave it as is."
        ),
        HumanMessagePromptTemplate.from_template(
            "Query: {query}"
        ),
    ])

    # Format the prompt with actual values
    formatted_prompt = format_prompt.format_messages(query=query)
    formatted_response = llm.invoke(formatted_prompt)
    logger.info(f"Formatted response: {formatted_response.content}")
    state["response"] = formatted_response.content
    return state


def countries_agent(state):
    """Handle country-related queries."""
    query = state["query"]
    
    # Get relevant context from vector store
    context_docs, context_metadatas = search_countries_db(query)
    context = "\n".join(context_docs) if context_docs else "No specific country information found."
    state["contexts"] = context_docs
    state["sources"] = [meta.get('source', 'Unknown') for meta in context_metadatas]
    
    country_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a country expert specialised in specific country information provided in the context. "
            "If the information is not contained in the context answer 'Not able to answer the question I do not have the information in the context'."
            "CRITICAL: Answer this question using the provided context."
        ),
        HumanMessagePromptTemplate.from_template(
            "Context: {context}\n\nQuestion: {query}\n\nProvide a helpful answer about countries."
        ),
    ])

    # Format the prompt with actual values
    formatted_prompt = country_prompt.format_messages(context=context, query=query)
    response = llm.invoke(formatted_prompt)
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
    
    general_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a helpful assistant "
            "You will respond to the user that the query they asked cannot be answered by the current knowledge base."
            "CRITICAL: You will only respond kindly that they should ask a different question"
        ),
        HumanMessagePromptTemplate.from_template(
            "Question: {query}\n\nProvide a helpful answer about countries."
        ),
    ])

    formatted_prompt = general_prompt.format_messages(query=query)
    response = llm.invoke(formatted_prompt)
    state["response"] = response.content
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