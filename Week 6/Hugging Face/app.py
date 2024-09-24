from langchain_community.document_loaders import PyMuPDFLoader

def load_pdfs(paths: list) -> list:

    # List of file paths for the PDFs you want to load
    paths = paths

    # Create a list to store loaded documents
    documents = []

    # Loop through each PDF and load it
    for path in paths:
        loader = PyMuPDFLoader(path)
        documents.extend(loader.load())  # Add the documents to the list

    return documents 

#####

from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_docs_recursive(documents: list, chunk_size: int, chunk_overlap: int) -> list:

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    return chunks

#####

from langchain.text_splitter import NLTKTextSplitter
import nltk

nltk.download('punkt')

def chunk_docs_nltk(documents: list, chunk_size: int, chunk_overlap: int) -> list:

    text_splitter = NLTKTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap)

    chunks = text_splitter.split_documents(documents)

    return chunks

#####

from langchain_openai import OpenAIEmbeddings

def create_embeddings_openai(model: str) -> OpenAIEmbeddings:

    # Initialize the OpenAIEmbeddings class
    embeddings_openai = OpenAIEmbeddings(model=model)

    return embeddings_openai

#####

from langchain_huggingface import HuggingFaceEmbeddings

def create_embeddings_opensource(model: str) -> OpenAIEmbeddings:

    # Initialize the OpenAIEmbeddings class
    embeddings_huggingface = HuggingFaceEmbeddings(model_name=model)

    return embeddings_huggingface 

#####

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

def create_vector_store(location: str, collection_name: str, vector_size: int, embeddings, documents: list) -> QdrantVectorStore:

    # Initialize the Qdrant client
    qdrant_client = QdrantClient(
        location=location
        )

    # Create a collection in Qdrant
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size, 
            distance=Distance.COSINE
            )
        )

        # Initialize QdrantVectorStore with the Qdrant client
    qdrant_vector_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=collection_name,
            embedding=embeddings,
        )
    
    qdrant_vector_store.add_documents(documents)
    
    return qdrant_vector_store

#####

def create_retriever_from_qdrant(vector_store: QdrantVectorStore):
  retriever = vector_store.as_retriever()

  return retriever

#####

from langchain.prompts import ChatPromptTemplate

def create_chat_prompt_template() -> ChatPromptTemplate:
    template = """
    Only answer the question using the context below.  If the answer can't be found in the context, respond "I don't know". 

    Question:
    {question}

    Context:
    {context}
    """
    prompt = ChatPromptTemplate.from_template(template)

    return prompt

#####

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from operator import itemgetter

def create_chain_openai(model: str, prompt: ChatPromptTemplate, retriever):
    
    llm = ChatOpenAI(model_name=model, temperature=0)

    # Create the chain with proper itemgetter and context
    chain = (
        {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": prompt | llm, "context": itemgetter("context")}
    )

    return chain

#####

import chainlit as cl

documents = load_pdfs(
    ["https://www.whitehouse.gov/wp-content/uploads/2022/10/Blueprint-for-an-AI-Bill-of-Rights.pdf", 
     "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf"]
     )

chunks = chunk_docs_recursive(documents, 500, 50)
embeddings = create_embeddings_opensource("dstampfli/finetuned-snowflake-arctic-embed-m")
qdrant_vector_store = create_vector_store(":memory:", "Midterm", 768, embeddings, chunks)
retriever = create_retriever_from_qdrant(qdrant_vector_store)
prompt = create_chat_prompt_template()
chain = create_chain_openai("gpt-4o-mini", prompt, retriever)


@cl.on_chat_start
async def main():
    documents = load_pdfs([
        "https://www.whitehouse.gov/wp-content/uploads/2022/10/Blueprint-for-an-AI-Bill-of-Rights.pdf",
        "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf"
    ])

    chunks = chunk_docs_recursive(documents, 500, 50)
    embeddings = create_embeddings_opensource("dstampfli/finetuned-snowflake-arctic-embed-m")
    qdrant_vector_store = create_vector_store(":memory:", "Midterm", 768, embeddings, chunks)
    retriever = create_retriever_from_qdrant(qdrant_vector_store)
    prompt = create_chat_prompt_template()
    chain = create_chain_openai("gpt-4o-mini", prompt, retriever)

    cl.user_session.set("midterm_chain", chain)

@cl.on_message
async def handle_message(message: cl.Message):
    chain = cl.user_session.get("midterm_chain")
    
    # Use invoke() method for synchronous execution
    res = chain.invoke({"question": message.content})  # Invoke the chain synchronously
    
    await cl.Message(content=res["response"]).send()