import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from pinecone import Pinecone as PineconeClient

load_dotenv()

# Keys
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_ENVIRONMENT = os.environ["PINECONE_ENVIRONMENT"]
PINECONE_INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]

pinecone = PineconeClient(api_key=PINECONE_API_KEY,
                          environment=PINECONE_ENVIRONMENT)

embeddings = CohereEmbeddings(model="multilingual-22-12")
vectorstore = Pinecone.from_existing_index(index_name=PINECONE_INDEX_NAME,
                                           embedding=embeddings)
retriever = vectorstore.as_retriever()


def fetch_documents(question):
    """Retrieve documents based on the question and prepare for the RAG."""
    documents = vectorstore.retrieve(question)
    context = " ".join([doc.text for doc in documents])
    return {"context": context, "question": question}


# RAG prompt
template = """Answer the question based only on the following context:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# RAG
model = ChatOpenAI(temperature=0,
                   model="gpt-4-1106-preview")


chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | RunnableLambda(fetch_documents)  # Make sure this function aligns with your data structure
    | prompt
    | model
    | StrOutputParser()
)

print(pinecone.info())  # This should return information about the Pinecone service.
test_query = "what is Ragnarok?"
print(fetch_documents(test_query))  # Check output format and content.
test_response = model.invoke(prompt="Hello world")
print(test_response)
test_output = chain.invoke("what is Ragnarok?")
print(test_output)  # Check if the output makes sense and integrates all components well.
