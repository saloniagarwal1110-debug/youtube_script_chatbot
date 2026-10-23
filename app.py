from flask import Flask, render_template, request,session
from langchain_text_splitters import RecursiveCharacterTextSplitter
from  langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from youtube_transcript_api import YouTubeTranscriptApi
import re

app = Flask(__name__)
app.secret_key = "youtube-chatbot-secret-key"

DB_FOLDER = "youtube_db"
COLLECTION_NAME = "youtube_transcript"

## Create embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

## Create Ollama LLM
llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0
)


@app.route("/", methods=["GET", "POST"])
@app.route("/get_transcript", methods=["GET", "POST"])
def index():

    transcript_text = ""
    error = None
    answer = None
    video_id = session.get("video_id", "")
    question = ""

    if request.method == "POST":

        action = request.form.get("action", "load_video")

        if action == "load_video":

            video_id = request.form.get("video_id", "").strip()

            # Check that the user entered a video ID
            if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
                error = "Enter a valid 11-character YouTube video ID."

            else:
                try:
                    ## Fetch YouTube transcript
                    ytt_api = YouTubeTranscriptApi()

                    transcript = ytt_api.fetch(
                        video_id,
                        languages=["hi", "en"])

                    transcript_data = transcript.to_raw_data()

                    transcript_text = "\n".join(
                        item["text"] for item in transcript_data)

                    ## Split transcript
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=300,chunk_overlap=75 )

                    chunks = text_splitter.create_documents( [transcript_text])

                    if not chunks:
                        raise ValueError("No transcript chunks were created.")

                    ## Create a separate Chroma collection per video
                    collection_name = f"youtube_{video_id}"

                    vector_store = Chroma.from_documents(
                        documents=chunks,
                        embedding=embeddings,
                        collection_name=collection_name,
                        persist_directory=DB_FOLDER)

                    ## Save video ID for follow-up questions
                    session["video_id"] = video_id

                except Exception as e:
                    app.logger.exception("Video processing failed")
                    error = f"Video processing error: {e}"

        elif action == "ask":

            video_id = session.get("video_id", "")
            question = request.form.get("question", "").strip()

            if not video_id:
                error = "Please load a video first."

            elif not question:
                error = "Please enter a question."

            else:
                try:
                    ## Open the selected video's vector store
                    collection_name = f"youtube_{video_id}"

                    vector_store = Chroma(
                        collection_name=collection_name,
                        embedding_function=embeddings,
                        persist_directory=DB_FOLDER)

                    ## Retrieve relevant chunks
                    retriever = vector_store.as_retriever(
                        search_type="mmr",
                        search_kwargs={"k": 3})

                    documents = retriever.invoke(question)

                    context = "\n\n".join(
                        doc.page_content for doc in documents)

                    ## Ask Ollama using retrieved context
                    prompt = f"""
You are a YouTube video assistant.
Answer using only the transcript context.
If the answer is not present, say you could not find it
in the transcript.

Transcript context:
{context}

Question:
{question}

Answer:
"""

                    response = llm.invoke(prompt)
                    answer = response.content

                    # Re-fetch transcript for display if needed
                    transcript_text = ""

                except Exception as e:
                    app.logger.exception("Question answering failed")
                    error = f"Could not answer question: {e}"

    return render_template(
        "index.html",
        transcript=transcript_text,
        error=error,
        answer=answer,
        video_id=video_id,
        question=question)


if __name__ == "__main__":
    app.run(debug=True)
