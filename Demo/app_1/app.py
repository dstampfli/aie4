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

def init_env_vars():
    # Load environment variables from .env file
    load_dotenv()

    # Google ADC  
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'credentials.json'
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

init_env_vars()
retreiver = create_retriever_vertexai()
chat_prompt_template = create_chat_prompt_template()

@cl.on_chat_start
async def on_chat_start():
    # Create the chain and save to session memory
    chain = create_chain('gemini-1.5-flash', chat_prompt_template, retreiver)
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    result = chain.invoke({"question": message.content})
    await cl.Message(content=result['response'].content).send()