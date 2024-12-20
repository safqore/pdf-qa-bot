from fastapi import FastAPI, HTTPException
from qdrant_client import QdrantClient
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Connect to Qdrant
client = QdrantClient(host="qdrant", port=6333)

@app.get("/health")
def health_check():
    return {"status": "running"}

@app.post("/query")
async def query(data: dict):
    user_input = data.get("input")
    if not user_input:
        raise HTTPException(status_code=400, detail="Input is required")
    # Simulate response
    return {"response": f"You said: {user_input}"}

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with ["http://localhost:3000"] for specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)