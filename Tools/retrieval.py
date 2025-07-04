import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

def load_vectorstore(persist_directory="db"):
    embeddings = OpenAIEmbeddings()
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

def build_lcel_qa_chain(vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template("""
You are an NCERT Study Assistant. Use the following context from the book to answer the user's question clearly and accurately.

Context:
{context}

Question: {question}
""")

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # LCEL-style chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()} 
        | prompt
        | llm
    )

    return chain

def answer_question(query: str):
    vectordb = load_vectorstore()
    chain = build_lcel_qa_chain(vectordb)

    answer = chain.invoke(query)
    print(f"\n📘 Question: {query}")
    print(f"\n🤖 Answer: {answer.content}")

# if __name__ == "__main__":
#     while True:
#         q = input("Ask a question (or 'exit'): ")
#         if q.lower() in ["exit", "quit"]:
#             break
#         answer_question(q)
