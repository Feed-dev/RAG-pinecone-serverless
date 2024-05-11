import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_cohere import CohereEmbeddings
from pinecone import Pinecone as PineconeClient
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda

load_dotenv()

# Pinecone setup
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]
PINECONE_ENVIRONMENT = os.environ["PINECONE_ENVIRONMENT"]

pinecone = PineconeClient(api_key=PINECONE_API_KEY)

# Cohere Embeddings setup
embeddings = CohereEmbeddings(model="multilingual-22-12")

# Pinecone index setup
index = pinecone.Index(PINECONE_INDEX_NAME)


def fetch_documents(question):
    """Fetch documents based on the question."""
    question_vector = embeddings.embed_query(question)
    # Use Pinecone query method to find similar vectors/documents
    response = index.query(vector=question_vector, top_k=5)
    documents = [doc for doc in response['matches']]
    context = " ".join([doc.metadata['text'] for doc in documents])
    return {"context": context, "question": question}


# RAG components setup
template = """Answer the question based only on the following context:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI(temperature=0, model="gpt-4-turbo")
chain = (
        RunnableLambda(fetch_documents)
        | prompt
        | model
        | StrOutputParser()
)

test_input = "What is Ragnarok?"
result = chain.invoke(test_input)
print(result)
