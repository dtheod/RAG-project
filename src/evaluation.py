import ragas
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from ragas.testset import TestsetGenerator
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader

load_dotenv()

generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

# Load the markdown file directly with LangChain
loader = TextLoader("../data/country_data.md")
docs = loader.load()

# Compute and add summary embeddings to documents for Ragas
embeddings = generator_embeddings.embed_documents([doc.page_content for doc in docs])
for i, doc in enumerate(docs):
    doc.metadata['summary_embedding'] = embeddings[i]

# Debug: Check if docs are loaded and have embeddings
print(f"Number of documents loaded: {len(docs)}")
if docs:
    print(f"First document preview: {docs[0].page_content[:200]}...")
    print(f"First document has summary_embedding: {'summary_embedding' in docs[0].metadata}")

generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
dataset = generator.generate_with_langchain_docs(docs, testset_size=10)

print(dataset.to_pandas())






