from fastapi import FastAPI, UploadFile, File
import shutil
import os
from fastapi import FastAPI
from rag.loader import load_document
from rag.chunking import fixed_size_chunking, overlapping_chunking
from rag.embeddings import embed_texts
from rag.vectorstore import create_index, add_embeddings
from rag.retriever import retrieve_relevant_chunks
from rag.generator import generate_answer


app = FastAPI()

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

#file upload endpoint
@app.post("/upload")
def upload_document(file: UploadFile = File(...)):

    file_path = os.path.join(DATA_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 📄 Load text
    text = load_document(file_path)

    # 🔹 Choose chunk strategy (default fixed for now)
    chunks = fixed_size_chunking(text)

    # 🔹 Create embeddings
    embeddings = embed_texts(chunks)

    # 🔹 Create index if first time
    create_index(len(embeddings[0]))

    # 🔹 Store in vector DB
    add_embeddings(embeddings, chunks)

    return {
        "message": "Document uploaded and indexed",
        "chunks_created": len(chunks)
    }


#query endpoint

@app.post("/query")
def query_documents(query: str):

    # 🔹 Retrieve relevant chunks
    relevant_chunks = retrieve_relevant_chunks(query)

    # 🔹 Generate answer
    answer = generate_answer(query, relevant_chunks)

    return {
        "query": query,
        "answer": answer,
        "context": relevant_chunks
    }
