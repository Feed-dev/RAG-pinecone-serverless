import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone as PineconeClient
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_pinecone import PineconeVectorStore
from langchain_core.tracers.context import tracing_v2_enabled

load_dotenv()

# Pinecone setup
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]
PINECONE_ENVIRONMENT = os.environ["PINECONE_ENVIRONMENT"]

pinecone = PineconeClient(api_key=PINECONE_API_KEY)

# Anthropic setup
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]

# Cohere Embeddings setup
embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")

# Pinecone index setup
vectorstore = Pinecone.from_existing_index(index_name=PINECONE_INDEX_NAME, embedding=embeddings)
retriever = vectorstore.as_retriever()

# RAG prompt
template = """Answer the question based only on the following context:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# RAG
model = ChatAnthropic(
    temperature=0,
    model="claude-3-sonnet-20240229",
    anthropic_api_key=ANTHROPIC_API_KEY,
    max_tokens=1000
)

# Post-processing
def format_docs(docs):
    return "\n\n".join(f"File: {doc.metadata['file']} - Page: {doc.metadata['page']} - Chunk: {doc.metadata['chunk']}\n{doc.page_content}" for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# Run the chain with tracing enabled
def run_chain(query):
    with tracing_v2_enabled(project_name="LangServe Walkthrough"):
        response = chain.invoke(query)
        print("Response:\n", response)

if __name__ == "__main__":
    test_query = "write a report on angel evocation?"
    run_chain(test_query)