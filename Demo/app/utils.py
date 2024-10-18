from langchain_google_vertexai import VertexAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import NLTKTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_core.retrievers import BaseRetriever
from langchain_qdrant import QdrantVectorStore
from langchain_google_community import VertexAISearchRetriever
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from operator import itemgetter

def create_embeddings_vertexai(model="text-embedding-004") -> VertexAIEmbeddings:

    # Initialize the VertexAIEmbeddings class
    embeddings = VertexAIEmbeddings(model_name='text-embedding-005')

    return embeddings

def remove_empty_chunks(chunks_start: list) -> list:
    
    start = len(chunks_start)
    # print(f'start - {start} chunks')
    
    # Remove empty chunks
    chunks_end = [chunk for chunk in chunks_start if chunk.page_content.strip()]

    end = len(chunks_end)
    # print(f'end - {end} chunks')

    return chunks_end

def chunk_docs_recursive(documents: list, chunk_size=500, chunk_overlap=50) -> list:

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    chunks_start = text_splitter.split_documents(documents)

    chunks_end = remove_empty_chunks(chunks_start=chunks_start)

    return chunks_end

def chunk_docs_nltk(documents: list, chunk_size=500, chunk_overlap=50) -> list:

    text_splitter = NLTKTextSplitter(
    chunk_size=chunk_size, 
    chunk_overlap=chunk_overlap)

    chunks_start = text_splitter.split_documents(documents)

    chunks_end = remove_empty_chunks(chunks_start=chunks_start)

    return chunks_end

def chunk_docs_semantic(documents: list, ) -> list:

    # TODO - Use embeddings parameter
    text_splitter = SemanticChunker(create_embeddings_vertexai(), breakpoint_threshold_type="percentile")

    chunks_start = text_splitter.split_documents(documents)

    # Remove empty chunks
    chunks_end = remove_empty_chunks(chunks_start)

    return chunks_end

def create_qdrant_vector_store(location: str, collection_name: str, vector_size: int, embeddings: Embeddings, documents: list) -> QdrantVectorStore:

    # Initialize the Qdrant client
    qdrant_client = QdrantClient(location=location)

    # Create a collection in Qdrant
    qdrant_client.create_collection(collection_name=collection_name, vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE))

    # Initialize QdrantVectorStore with the Qdrant client
    qdrant_vector_store = QdrantVectorStore(client=qdrant_client, collection_name=collection_name, embedding=embeddings)
    
    qdrant_vector_store.add_documents(documents)
    
    return qdrant_vector_store

def create_retriever_qdrant(vector_store: QdrantVectorStore) -> BaseRetriever:

    retriever = vector_store.as_retriever()

    return retriever

def create_retriever_vertexai() -> VertexAISearchRetriever:

    retriever = VertexAISearchRetriever(project_id=os.environ['PROJECT_ID'], location_id=os.environ['LOCATION_ID'], data_store_id=os.environ['DATA_STORE_ID'], max_documents=3)

    return retriever

def create_chat_prompt_template(prompt: str = None) -> ChatPromptTemplate:
    
    template = '''
    You are a helpful conversational agent for the State of California.
    Your expertise is fully understanding the California Health & Wellness health  plan. 
    You need to answer questions posed by the member, who is trying to get answers about their health plan.  
    Your goal is to provide a helpful and detailed response, in at least 2-3 sentences. 

    You will be analyzing the health plan documents to derive a good answer, based on the following information:
    1. The question asked.
    2. The provided context, which comes from various documents of the pharmacy manuals repository. You will need to answer the question based on the provided context.

    Now it's your turn!

    {question}

    {context}

    '''
    
    prompt = PromptTemplate.from_template(template)

    return prompt

def create_chain (model_name: str, prompt: ChatPromptTemplate, retriever: BaseRetriever):

    if "gemini" in model_name.lower():
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0,
            safety_settings={
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                },
            )
    else:
        print("Unsuported model name")
        
    chain = (
        {"context": itemgetter("question") | retriever, "question": itemgetter("question")} 
        | RunnablePassthrough.assign(context=itemgetter("context")) 
        | {"response": prompt | llm, "context": itemgetter("context")}
        )

    return chain