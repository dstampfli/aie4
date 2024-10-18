### Import Section ###

import chainlit as cl
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_openai import ChatOpenAI
from operator import itemgetter
import os
import uuid 
from utils import *

### Global Section ###

# Load environment variables from .env file
load_dotenv()

# Google
# ADC  
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'aie4_demo_sa_credentials.json'
# GCP Project
PROJECT_ID = os.environ['PROJECT_ID']
REGION = os.environ['REGION']
# Vertex AI Agent Builder Data Store
LOCATION_ID = os.environ['LOCATION_ID']
DATA_STORE_ID = os.environ['DATA_STORE_ID']

# LangChain 
LANGCHAIN_PROJECT=os.environ['LANGCHAIN_PROJECT'] 
os.environ['LANGCHAIN_PROJECT'] = LANGCHAIN_PROJECT + f" - {uuid.uuid4().hex[0:8]}"
print(LANGCHAIN_PROJECT)

# Init global variables
retreiver = create_retriever_vertexai()
chat_prompt_template = create_chat_prompt_template()

### On Chat Start (Session Start) Section ###
@cl.on_chat_start
async def on_chat_start():

    # Create the chain
    chain = create_chain('gemini-1.5-flash', chat_prompt_template, retreiver)

    cl.user_session.set("chain", chain)

### Rename Chains ###
@cl.author_rename
def rename(orig_author: str):
    rename_dict = {
        "Assistant" : "Health Plan Chatbot"
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