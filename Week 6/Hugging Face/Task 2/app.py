from utils import *
import chainlit as cl

print("loading pdfs")
documents = load_pdfs([
    "https://www.whitehouse.gov/wp-content/uploads/2022/10/Blueprint-for-an-AI-Bill-of-Rights.pdf",
    "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf"
])

print("chunking pdfs")
chunks = chunk_docs_recursive(documents, 500, 50)
print("generating embeddings")
embeddings = create_embeddings_opensource("sentence-transformers/all-MiniLM-L6-v2")
print("creating vector store")
qdrant_vector_store = create_vector_store(":memory:", "Midterm", 384, embeddings, chunks)
print("creating retriever")
retriever = create_retriever_from_qdrant(qdrant_vector_store)
print("creating prompt")
prompt = create_chat_prompt_template()
print("creating chain")
chain = create_chain_openai("gpt-4o-mini", prompt, retriever)

@cl.on_chat_start
async def main():
    print("on_chat_start")
    cl.user_session.set("midterm_chain", chain)

@cl.on_message
async def handle_message(message: cl.Message):
    print("handle_message")
    chain = cl.user_session.get("midterm_chain")
    res = chain.invoke({"question": message.content})
    await cl.Message(content=res["response"]).send()