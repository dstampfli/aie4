# https://github.com/Chainlit/cookbook/blob/main/resume-chat/app.py

from chainlit.types import ThreadDict
from datetime import datetime
from dotenv import load_dotenv
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from langchain_cohere import CohereRerank
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain_google_community import VertexAISearchRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.schema import Document
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableLambda
from langchain.schema.runnable.config import RunnableConfig
from operator import itemgetter
import chainlit as cl
import os
import vertexai

def init_env_vars():
    print('init_env_vars')

    # Load environment variables from the .env file
    load_dotenv()

    # Set Google ADC to service account credentials
    # https://cloud.google.com/docs/authentication/provide-credentials-adc
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'credentials.json'

    vertexai.init(project=os.environ['PROJECT_ID'], location=os.environ['REGION'])

    print(os.environ['USE_VERTEX_AI'])
    print(os.environ['USE_UNSTRUCTURED'])
    print(os.environ['PROJECT_ID'])
    print(os.environ['REGION'])
    print(os.environ['LOCATION_ID'])
    print(os.environ['DATA_STORE_ID'])

def init_faiss_retriever():
    print('init_faiss_retriever')

    if os.environ['USE_UNSTRUCTURED'] == True:
        vector_store = FAISS.load_local("faiss_index_unstructured", HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2'), allow_dangerous_deserialization=True)
    else:
        vector_store = FAISS.load_local("faiss_index_pymupdfloader", VertexAIEmbeddings(model_name='text-embedding-004'), allow_dangerous_deserialization=True)

    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 4})
    compressor = CohereRerank(model="rerank-english-v3.0")
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)

    return compression_retriever

def setup_runnable() -> Runnable:
    print('setup_runnable')

    # Get conversation history from session state 
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory
    
    # Create the retriever
    if os.environ['USE_VERTEX_AI'] == 'False':
        retriever = init_faiss_retriever()
    else:
        retriever = VertexAISearchRetriever(project_id=os.environ['PROJECT_ID'], 
                                            location_id=os.environ['LOCATION_ID'], 
                                            data_store_id=os.environ['DATA_STORE_ID'], 
                                            max_documents=5)
        
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
    Your expertise is fully understanding the Medi-Cal provider manuals. 
    You need to answer questions posed by a member, who is trying to get answers about services provided by Medi-Cal.  
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
        {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
        | RunnablePassthrough.assign(context=lambda x: x['context'].page_content if isinstance(x['context'], Document) else x['context'])    
        | RunnablePassthrough.assign(history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"))
        | RunnableLambda(inspect)
        | prompt
        | llm 
        | RunnableLambda(inspect)
        | StrOutputParser()
    )

    return runnable

async def inspect(state):
    print(state)
    return state

init_env_vars()

#####

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
    cl.user_session.set('memory', ConversationBufferMemory(return_messages=True))
    
    # Create the runnable and save it to session state
    runnable = setup_runnable()
    cl.user_session.set('runnable', runnable)

@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    # Create memory instance 
    memory = ConversationBufferMemory(return_messages=True)
    
    # Load messages 
    root_messages = [m for m in thread['steps'] if m['parentId'] == None]
    for message in root_messages:
        if message['type'] == 'user_message':
            memory.chat_memory.add_user_message(message['output'])
        else:
            memory.chat_memory.add_ai_message(message['output'])

    # Save memory to session state 
    cl.user_session.set('memory', memory)

    # Create the runnable and save it to session state
    runnable = setup_runnable()
    cl.user_session.set('runnable', runnable)
    
@cl.on_message
async def on_message(message: cl.Message):

    # Get the memory from session state
    memory = cl.user_session.get('memory')      # type: ConversationBufferMemory
    
    # Get the runnable from session state
    runnable = cl.user_session.get('runnable')  # type: Runnable
    
    # Steam the response
    res = cl.Message(content='')
    async for chunk in runnable.astream({'question': message.content}, config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()])):
        await res.stream_token(chunk)
    await res.send()
    # print(message.content)
    # print(res.content)
    
    # Update the conversation memory with question and the answer
    memory.chat_memory.add_user_message(message.content)
    memory.chat_memory.add_ai_message(res.content)
    # print(len(memory.chat_memory.messages))

    # breakpoint()