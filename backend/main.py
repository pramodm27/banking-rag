
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import requests
import uuid
from PyPDF2 import PdfReader

app = FastAPI()

qdrant = QdrantClient(host="qdrant", port=6333)
model = SentenceTransformer("all-MiniLM-L6-v2")

COLLECTION = "banking_docs"

class Query(BaseModel):
    question: str

def init_collection():
    try:
        qdrant.get_collection(COLLECTION)
    except:
        qdrant.recreate_collection(
            collection_name=COLLECTION,
            vectors_config={"size": 384, "distance": "Cosine"}
        )

init_collection()

@app.get("/")
def root():
    return {"message": "Production Banking RAG Running"}

# PDF ingestion
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    reader = PdfReader(file.file)
    texts = []

    for page in reader.pages:
        texts.append(page.extract_text())

    chunks = [t for t in texts if t]

    vectors = model.encode(chunks).tolist()

    points = [
        {"id": str(uuid.uuid4()), "vector": vectors[i], "payload": {"text": chunks[i]}}
        for i in range(len(chunks))
    ]

    qdrant.upsert(collection_name=COLLECTION, points=points)

    return {"status": "PDF ingested", "chunks": len(chunks)}

# query
@app.post("/ask")
def ask(q: Query):
    query_vec = model.encode(q.question).tolist()

    response = qdrant.query_points(
        collection_name=COLLECTION,
        query=query_vec,
        limit=3
    )

    results = getattr(response, "points", response)

    if not results:
        return {"answer": "No relevant documents found."}

    context_chunks = []
    for r in results:
        if hasattr(r, "payload") and r.payload:
            context_chunks.append(r.payload.get("text", ""))

    context = " ".join(context_chunks)

    if not context.strip():
        return {"answer": "Empty context retrieved."}

    try:
        llm_response = requests.post(
            "http://ollama:11434/api/generate",
            json={
                "model": "phi3",
                "prompt": f"{context}\nQuestion: {q.question}",
                "stream": False
            }
        )

        data = llm_response.json()
        answer = data.get("response", "No response from model")

    except Exception as e:
        return {"answer": f"LLM error: {str(e)}"}

    return {"answer": answer}
