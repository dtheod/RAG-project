from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os

load_dotenv()

# Simple model instances
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)
