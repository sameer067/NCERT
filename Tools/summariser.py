import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


def load_vectorstore(persist_directory="db"):
    embeddings = OpenAIEmbeddings()
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )


def summarize_topic(topic: str):
    """Summarize a topic based on NCERT content."""
    vectordb = load_vectorstore()
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(topic)

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = ChatPromptTemplate.from_template("""
You are a helpful NCERT study guide. Summarize the following topic in clear, student-friendly language with proper explanation, based on the context provided, if the topic is not found, dont provide general information just say "Topic not found in NCERT content and ask for the topic".

Topic: {topic}

Context:
{context}

Summary:
""")

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.4)
    chain = prompt | llm

    result = chain.invoke({"topic": topic, "context": context})
    return result.content