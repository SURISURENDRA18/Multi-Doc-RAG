import os
import sys

# FIX PATH ISSUE
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import DATA_PATH, CHROMA_PATH, EMBEDDING_MODEL


def load_documents():
    docs = []

    if not os.path.exists(DATA_PATH):
        print("❌ data/docs folder not found")
        return docs

    for file in os.listdir(DATA_PATH):
        path = os.path.join(DATA_PATH, file)

        if file.endswith(".pdf"):
            loader = PyPDFLoader(path)
            docs.extend(loader.load())

        elif file.endswith(".txt"):
            loader = TextLoader(path)
            docs.extend(loader.load())

    return docs


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_documents(docs)


def create_db():
    docs = load_documents()

    if not docs:
        print("❌ No documents found")
        return

    chunks = split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    db = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    db.persist()
    print("✅ DB created successfully!")


if __name__ == "__main__":
    create_db()