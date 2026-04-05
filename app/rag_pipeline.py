from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

from app.config import CHROMA_PATH, EMBEDDING_MODEL, OPENAI_API_KEY, LLM_MODEL


# ✅ LOAD EVERYTHING ONCE (GLOBAL)

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

# Vector DB
db = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings
)

# Retriever
retriever = db.as_retriever(search_kwargs={"k": 3})

# LLM
llm = ChatOpenAI(
    model=LLM_MODEL,
    temperature=0,
    api_key=OPENAI_API_KEY
)

# QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)


# ✅ QUERY FUNCTION (FAST)
def query_rag(question: str):
    result = qa_chain.invoke({"query": question})

    answer = result["result"]
    sources = [
        doc.metadata.get("source", "unknown")
        for doc in result["source_documents"]
    ]

    return {
        "answer": answer,
        "sources": list(set(sources))
    }

