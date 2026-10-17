
import json
import faiss
import numpy as np
import ollama
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI(title="College RAG Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
# Load resources once when the server starts
index = faiss.read_index("vector_store/college.index")

with open("vector_store/chunks.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)

model = SentenceTransformer("all-MiniLM-L6-v2")

DISTANCE_THRESHOLD = 1.0


class Question(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "College RAG API is running"}


@app.post("/ask")
def ask_question(data: Question):
    query = data.question.strip()

    if not query:
        return {"answer": "Please enter a question.", "sources": []}

    query_embedding = model.encode([query])
    query_embedding = np.asarray(query_embedding, dtype="float32")

    distances, indices = index.search(query_embedding, 2)

    context_parts = []
    sources = []

    for distance, idx in zip(distances[0], indices[0]):
        if idx >= 0 and distance < DISTANCE_THRESHOLD:
            context_parts.append(chunks[idx])
            sources.append(chunks[idx].split("\n")[0])

    if not context_parts:
        return {
            "answer": "The available documents do not provide enough relevant information.",
            "sources": []
        }

    context = "\n\n".join(context_parts)

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer using only the provided context. "
                    "Share relevant partial information and explain "
                    "what is missing. If the context cannot answer "
                    "the question, say so. Never invent facts."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}"
            }
        ]
    )

    return {
        "answer": response["message"]["content"],
        "sources": sources
    }
