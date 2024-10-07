### Import Section ###
from dotenv import load_dotenv
from operator import itemgetter

import chainlit as cl

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_openai import ChatOpenAI

from utils import build_chat_prompt, process_file, build_vector_store

### Global Section ###

# Load environment variables
load_dotenv()

# Init global variables
chat_prompt = build_chat_prompt()
chat_model = ChatOpenAI(model="gpt-4o-mini")
set_llm_cache(InMemoryCache())

### On Chat Start (Session Start) Section ###
@cl.on_chat_start
async def on_chat_start():

    # Wait for the user to upload a file
    files = None
    while files == None:
        # Async method: This allows the function to pause execution while waiting for the user to upload a file,
        # without blocking the entire application. It improves responsiveness and scalability.
        files = await cl.AskFileMessage(
            content="Please upload a PDF file to begin!",
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()
    file = files[0]
    
    msg = cl.Message(
    content=f"Processing `{file.name}`...",
    )
    await msg.send()

    # Load the file
    docs = process_file(file)

    # Create the vectorstore and add documents to it
    vectorstore = build_vector_store()
    vectorstore.add_documents(docs)

    # Create the retriever
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3})

    # Create the chain
    retrieval_augmented_qa_chain = (
        {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | chat_prompt | chat_model
    )
    
    # Let the user know that the system is ready
    msg.content = f"Processing `{file.name}` done. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", retrieval_augmented_qa_chain)

### Rename Chains ###
@cl.author_rename
def rename(orig_author: str):
    rename_dict = {
        "Assistant" : "PDF Explorer Bot"
    }
    return rename_dict.get(orig_author, orig_author)

### On Message Section ###
@cl.on_message
async def main(message: cl.Message):
    runnable = cl.user_session.get("chain")

    msg = cl.Message(content="")

    # Async method: Using astream allows for asynchronous streaming of the response,
    # improving responsiveness and user experience by showing partial results as they become available.
    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk.content)

    await msg.send()