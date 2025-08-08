import os
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from config import VIDEO_PATH, VECTOR_DB_PATH, EMBEDDING_MODEL

client = OpenAI()
MAX_FILE_SIZE_MB = 25  # Whisper API limit

def transcribe_video(video_file):
    size_mb = os.path.getsize(video_file) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        print(f"⚠️ Skipping {video_file} ({size_mb:.2f}MB) — exceeds Whisper limit")
        return None

    with open(video_file, "rb") as f:
        transcript = client.audio.transcriptions.create(model="whisper-1", file=f)
    return transcript.text

def ingest_videos():
    all_docs = []

    for file in os.listdir(VIDEO_PATH):
        if file.endswith(".mp4"):
            video_file = os.path.join(VIDEO_PATH, file)
            print(f"🎥 Processing video: {file}")

            transcript_text = transcribe_video(video_file)
            if transcript_text:
                all_docs.append(Document(
                    page_content=transcript_text,
                    metadata={"source": file, "source_type": "Video"}
                ))

    if not all_docs:
        print(" No video files found!")
        return

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_docs)

    # Store in Chroma
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)
    vectorstore.add_documents(chunks)
    vectorstore.persist()

    print(f" Ingested {len(chunks)} video transcript chunks into ChromaDB!")

if __name__ == "__main__":
    ingest_videos()
