import os
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from config import AUDIO_PATH, VECTOR_DB_PATH, EMBEDDING_MODEL

client = OpenAI()

def transcribe_audio(audio_file: str) -> str:
    """Transcribe an audio file using OpenAI Whisper API."""
    with open(audio_file, "rb") as f:
        transcript = client.audio.transcriptions.create(model="whisper-1", file=f)
    return transcript.text

def ingest_audio_files():
    print("AUDIO_PATH       =", os.path.abspath(AUDIO_PATH))
    persist_dir = os.path.join(VECTOR_DB_PATH, "audio_db")
    print("Persist dir      =", os.path.abspath(persist_dir))

    all_docs = []

    if not os.path.isdir(AUDIO_PATH):
        print(" AUDIO_PATH does not exist.")
        return

    files = [f for f in os.listdir(AUDIO_PATH) if f.lower().endswith((".mp3", ".wav"))]
    print("Files discovered =", files)

    for file in files:
        file_path = os.path.join(AUDIO_PATH, file)
        print(f"🎤 Processing audio: {file}")

        try:
            transcript_text = transcribe_audio(file_path)
        except Exception as e:
            print(f"⚠️ Transcription failed for {file}: {e}")
            continue

        if transcript_text and transcript_text.strip():
            all_docs.append(Document(
                page_content=transcript_text,
                metadata={"source": file, "source_type": "Audio"}
            ))

    if not all_docs:
        print(" No audio transcripts produced; nothing to ingest.")
        return

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_docs)

    # Store in Chroma under vectorstore/audio_db
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
    vectorstore.add_documents(chunks)
    vectorstore.persist()

    print(f" Ingested {len(chunks)} audio transcript chunks into ChromaDB at {persist_dir}!")

if __name__ == "__main__":
    ingest_audio_files()
