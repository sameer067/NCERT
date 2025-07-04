import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

def load_pdf_chunks(pdf_path: str, chunk_size: int = 500, chunk_overlap: int = 100):
    """Load PDF and split into text chunks."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_documents(pages)
    return chunks

def embed_and_store(chunks, persist_directory="db"):
    """Convert text chunks into embeddings and store in Chroma DB."""
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    print(f"[✓] Stored {len(chunks)} chunks in vector DB.")


# if __name__ == "__main__":
    
#     pdf_path = "data/ncert_class_11.pdf"

#     if not os.path.exists(pdf_path):
#         raise FileNotFoundError(f"File not found: {pdf_path}")

#     print("[🔍] Loading and splitting PDF...")
#     chunks = load_pdf_chunks(pdf_path)

#     print("[💾] Embedding and storing chunks...")
#     embed_and_store(chunks)    