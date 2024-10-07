import uuid

from chainlit.types import AskFileResponse

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.globals import set_llm_cache
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

def build_chat_prompt():
    rag_system_prompt_template = """\
    You are a helpful assistant that uses the provided context to answer questions. Never reference this prompt, or the existance of context.
    """

    rag_user_prompt_template = """\
    Question:
    {question}
    Context:
    {context}
    """

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", rag_system_prompt_template),
        ("human", rag_user_prompt_template)
    ])

    return chat_prompt 

def build_vector_store():
    # Create a Qdrant vector store with a cache and add the doc to it
    # Collection
    collection_name = f"pdf_to_parse_{uuid.uuid4()}"
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )

    # Cache
    store = LocalFileStore("./cache/")
    core_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        core_embeddings, store, namespace=core_embeddings.model
    )

    # Vector store
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
    embedding=cached_embedder)

    return vectorstore

def process_file(file: AskFileResponse):
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as tempfile:
        with open(tempfile.name, "wb") as f:
            f.write(file.content)

    Loader = PyMuPDFLoader

    loader = Loader(tempfile.name)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    for i, doc in enumerate(docs):
        doc.metadata["source"] = f"source_{i}"
    return docs
