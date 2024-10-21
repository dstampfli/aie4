# https://github.com/Chainlit/cookbook/blob/main/resume-chat/app.py

from operator import itemgetter

from chainlit.types import ThreadDict
import chainlit as cl

from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableLambda
from langchain.schema.runnable.config import RunnableConfig
from langchain.memory import ConversationBufferMemory

# Additional imports  
import aiofiles
import asyncio
from datetime import datetime
from dotenv import load_dotenv
import os
from pprint import pprint
import uuid

from google.generativeai.types import HarmCategory, HarmBlockThreshold
from langchain_google_community import VertexAISearchRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document

def init_env_vars():
    # Load environment variables from the .env file
    load_dotenv()

    # Set Google ADC to service account credentials
    # https://cloud.google.com/docs/authentication/provide-credentials-adc
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'credentials.json'

def setup_runnable() -> Runnable:
    
    # Get conversation history from session state 
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory
    
    # Create the retriever
    max_documents = 5
    retriever = VertexAISearchRetriever(project_id=os.environ['PROJECT_ID'], 
                                        location_id=os.environ['LOCATION_ID'], 
                                        data_store_id=os.environ['DATA_STORE_ID'], 
                                        max_documents=max_documents)

    # Create the llm
    llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            stream=True,
            temperature=0,
            safety_settings={
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                },
            )
    
    # Create the prompt 
    prompt_template = """
    You are a helpful conversational agent for the State of California.
    Your expertise is fully understanding the California Health & Wellness health plan. 
    You need to answer questions posed by the member, who is trying to get answers about their health plan.  
    Your goal is to provide a helpful and detailed response, in at least 2-3 sentences. 

    You will be analyzing the health plan documents to derive a good answer, based on the following information:
    1. The question asked.
    2. The provided context, which comes from health plan document. You will need to answer the question based on the provided context and the conversation history.

    Now it's your turn!

    {question}

    {context}

    {history}
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["history", "context", "question"]
    )

    # Define the runnable pipeline
    runnable = (
        # Step 1: Extract context and question
        # "context" is created by retrieving relevant documents based on the "question" using the retriever
        # The "question" is simply extracted from the input using itemgetter("question")
        {"context": itemgetter("question") | retriever, "question": itemgetter("question")} 
        
        # Step 2: Transform the context
        # Check if the "context" is a Document object. If it is, extract its "page_content" (the actual text)
        # If "context" is not a Document, pass it through as-is
        # This is handled using a lambda function
        | RunnablePassthrough.assign(context=lambda x: x['context'].page_content if isinstance(x['context'], Document) else x['context'])
        
        # Step 3: Load and assign history
        # Load the conversation history from memory using memory.load_memory_variables via a RunnableLambda.
        # The history is assigned via itemgetter("history"), which retrieves the history from the memory object
        | RunnablePassthrough.assign(history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"))

        # Step 4: Build the prompt
        # Combine the context (retrieved in Step 2), the question, and the conversation history (from Step 3) to form a prompt that is passed to the LLM
        | prompt

        # Step 5: Use the language model (LLM)
        # The LLM processes the prompt to generate a response. This response is then passed to the next step
        | llm 

        # Step 6: Parse the output
        # The StrOutputParser converts the LLM's output into a string format suitable for further processing or displaying to the user.
        | StrOutputParser()
    )

    return runnable

@cl.author_rename
def rename(orig_author: str):
    
    # Rename the chatbot in the UI
    # print(orig_author)
    rename_dict = {
        "Assistant" : "Health Plan Chatbot"
    }
    return rename_dict.get(orig_author, orig_author)

@cl.on_chat_start
async def on_chat_start():

    # Create the memory and save to session state
    cl.user_session.set("memory", ConversationBufferMemory(return_messages=True))
    
    # Create the runnable and save it to session state
    runnable = setup_runnable()
    cl.user_session.set("runnable", runnable)

@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    # Create memory instance 
    memory = ConversationBufferMemory(return_messages=True)
    
    # Load messages 
    root_messages = [m for m in thread["steps"] if m["parentId"] == None]
    for message in root_messages:
        if message["type"] == "user_message":
            memory.chat_memory.add_user_message(message["output"])
        else:
            memory.chat_memory.add_ai_message(message["output"])

    # Save memory to session state 
    cl.user_session.set("memory", memory)

    # Create the runnable and save it to session state
    runnable = setup_runnable()
    cl.user_session.set("runnable", runnable)
    
@cl.on_message
async def on_message(message: cl.Message):

    # Get the memory from session state
    memory = cl.user_session.get("memory")      # type: ConversationBufferMemory
    
    # Get the runnable from session state
    runnable = cl.user_session.get("runnable")  # type: Runnable
    
    # Steam the response
    res = cl.Message(content="")
    async for chunk in runnable.astream({"question": message.content}, config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()])):
        await res.stream_token(chunk)
    await res.send()
    
    # Update the conversation memory with question and the answer
    memory.chat_memory.add_user_message(message.content)
    memory.chat_memory.add_ai_message(res.content)

    print(message.content)
    print(res.content)
    print(len(memory.chat_memory.messages))

    # breakpoint()