# Production Banking RAG

## Run
docker compose up --build

## Pull model
docker exec -it <ollama_container> ollama pull llama3

## Upload PDF
POST /upload_pdf (form-data file)

## Ask
POST /ask
{
  "question": "Explain loan process"
}

## Features
- PDF ingestion
- Vector search
- LLM answers
- Dockerized infra
