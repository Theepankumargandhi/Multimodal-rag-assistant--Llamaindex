import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import DOCS_PATH, VECTOR_DB_PATH, EMBEDDING_MODEL

def ingest_documents():
    docs = []

    # Load documents
    for file in os.listdir(DOCS_PATH):
        file_path = os.path.join(DOCS_PATH, file)

        if file.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
            docs.extend(loader.load())
        elif file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            docs.extend(loader.load())

    if not docs:
        print(" No documents found for ingestion!")
        return

    print(f" Loaded {len(docs)} raw documents")

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)

    # Add metadata
    for d in split_docs:
        d.metadata["source_type"] = "Document"

    print(f" Split into {len(split_docs)} chunks")

    # Store in Chroma
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(split_docs, embedding=embeddings, persist_directory=VECTOR_DB_PATH)
    vectorstore.persist()

    print(f" Ingested {len(split_docs)} document chunks into ChromaDB at {VECTOR_DB_PATH}")

if __name__ == "__main__":
    ingest_documents()
