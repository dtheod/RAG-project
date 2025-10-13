from src.models import llm
from src.logger import logger
from langchain.chains import LLMMathChain
from langchain.agents import Tool
from langchain.prompts import ChatPromptTemplate
from src.retrieval import search_cars_db, search_countries_db
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

def cars_agent(state):
    """Handle car-related queries."""
    query = state["query"]
    
    # Get relevant context from vector store
    context_docs = search_cars_db(query)
    context = "\n".join(context_docs) if context_docs else "No specific car information found."
    
    car_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a car expert specialised in specific car information provided in the context. "
            "If the information is not contained in the context answer 'Not able to answer the question I do not have the information in the context'."
            "CRITICAL: Answer this question using the provided context."
        ),
        HumanMessagePromptTemplate.from_template(
            "Context: {context}\n\nQuestion: {query}\n\nProvide a helpful answer about cars."
        ),
    ])

    # Format the prompt with actual values
    formatted_prompt = car_prompt.format_messages(context=context, query=query)
    response = llm.invoke(formatted_prompt)
    state["response"] = response.content
    return state

def format_agent(state):
    """Format the response for better readability."""
    query = state["query"]


    format_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are an expert at formatting text to be clear and concise. "
            "Fix any grammar or spelling issues. "
            "Please format the following response for better readability. "
            "DO NOT change the meaning of the response."
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
    context_docs = search_countries_db(query)
    context = "\n".join(context_docs) if context_docs else "No specific country information found."
    
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
