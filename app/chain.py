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
# PINECONE_ENVIRONMENT = os.environ["PINECONE_ENVIRONMENT"]
pinecone = PineconeClient(api_key=PINECONE_API_KEY)
index = pinecone.Index(os.environ["PINECONE_INDEX_NAME"])

# Cohere Embeddings setup
embeddings = CohereEmbeddings(model="multilingual-22-12")


def fetch_documents(question):
    """Retrieve documents based on the question and prepare for the RAG."""
    # Generate the query vector for the question
    question_vector = embeddings.embed_query(question)

    # Query Pinecone to find the most similar documents
    results = index.query(question_vector, top_k=5)

    # Extract the text from the results and compile it into a single context string
    context = " ".join([doc['_source']['text'] for doc in results['matches']])
    return {"context": context, "question": question}


# RAG components setup
template = """Answer the question based only on the following context:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI(temperature=0, model="gpt-4-1106-preview")
chain = (
        RunnableParallel({"context": RunnablePassthrough(), "question": RunnablePassthrough()})
        | RunnableLambda(fetch_documents)
        | prompt
        | model
        | StrOutputParser()
)

# Test the chain with an example question
test_question = "what is Ragnarok?"
result = chain.invoke(test_question)
print(result)
