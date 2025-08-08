import os
import asyncio
import hashlib
import logging
import tempfile
import time
from typing import List, Dict, Tuple
import numpy as np
from dataclasses import dataclass
import pickle
import fitz  # PyMuPDF
import httpx
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn
from openai import AsyncOpenAI
import redis
import faiss
from rank_bm25 import BM25Okapi

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API")
EXPECTED_BEARER_TOKEN = os.getenv("BEARER_TOKEN")
REDIS_URL = os.getenv("REDIS_URL")

constitution_chunks = []
constitution_embeddings = None
constitution_chunk_id_to_idx = {}

# Constants
MAX_WORKERS = 2
EMBEDDING_BATCH_SIZE = 100
CHUNK_SIZE = 800
MAX_CHUNKS_PER_PDF = 2000
TOP_K = 20
BM25_PREFILTER_K = 100

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

try:
    redis_client = redis.from_url(REDIS_URL, decode_responses=False)
    redis_client.ping()
    logger.info("Redis cache enabled")
except Exception:
    redis_client = None
    logger.warning("Redis disabled - no caching")

app = FastAPI()

@dataclass
class Chunk:
    text: str
    source: str
    page: int
    chunk_id: str

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

def verify_token(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid auth scheme")
    token = authorization.split(" ")[1]
    if token != EXPECTED_BEARER_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid or expired token")

def sample_pages_aggressively(doc, source: str) -> List[int]:
    total_pages = len(doc)
    sample_size = 400
    if total_pages <= sample_size:
        return list(range(total_pages))
    key_pages = set([0, total_pages//4, total_pages//2, 3*total_pages//4, total_pages-1])
    additional = sample_size - len(key_pages)
    step = max(1, total_pages // additional)
    key_pages.update(range(0, total_pages, step))
    return sorted(list(key_pages))[:sample_size]

def split_text_to_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap_tokens: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += (chunk_size - overlap_tokens)
    return chunks

def extract_text_ultra_fast(pdf_content: bytes, source: str, max_chunks: int = MAX_CHUNKS_PER_PDF) -> List[Chunk]:
    chunks = []
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_content)
        tmp_path = tmp.name
    try:
        with fitz.open(tmp_path) as doc:
            page_indices = sample_pages_aggressively(doc, source)
            for page_num in page_indices:
                if len(chunks) >= max_chunks:
                    break
                page = doc[page_num]
                text = page.get_text()
                if text.strip():
                    splitted_chunks = split_text_to_chunks(text)
                    for chunk_text in splitted_chunks:
                        if len(chunks) >= max_chunks:
                            break
                        chunk_id = hashlib.md5(f"{source}{page_num}{len(chunks)}".encode()).hexdigest()[:12]
                        chunks.append(Chunk(chunk_text.strip(), source, page_num + 1, chunk_id))
    finally:
        os.unlink(tmp_path)
    return chunks

async def download_pdf_ultra_fast(url: str, session: httpx.AsyncClient) -> bytes:
    async with session.stream('GET', url, timeout=5.0) as response:
        response.raise_for_status()
        content = b""
        async for chunk in response.aiter_bytes(8192):
            content += chunk
            if len(content) > 150_000_000:
                raise Exception("PDF too large")
        return content

def get_cache_key_binary(text: str) -> str:
    return f"embed_bin:{hashlib.md5(text.encode()).hexdigest()}"

async def get_embeddings_openai_batch(texts: List[str]) -> List[List[float]]:
    response = await openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [item.embedding for item in response.data]

async def pre_cache_pdf_embeddings(url: str, source: str) -> Tuple[List[Chunk], np.ndarray, Dict[str, int]]:
    async with httpx.AsyncClient() as session:
        pdf_content = await download_pdf_ultra_fast(url, session)
    chunks = extract_text_ultra_fast(pdf_content, url, max_chunks=MAX_CHUNKS_PER_PDF)
    texts = [chunk.text for chunk in chunks]
    embeddings = []
    uncached_texts = []
    uncached_indices = []

    for i, text in enumerate(texts):
        if redis_client:
            cached = redis_client.get(get_cache_key_binary(text))
            if cached:
                embeddings.append(pickle.loads(cached))
                continue
        embeddings.append(None)
        uncached_texts.append(text)
        uncached_indices.append(i)

    if uncached_texts:
        new_embeddings = await get_embeddings_openai_batch(uncached_texts)
        for i, emb in enumerate(new_embeddings):
            original_idx = uncached_indices[i]
            embeddings[original_idx] = np.array(emb, dtype=np.float32)
            if redis_client:
                redis_client.setex(get_cache_key_binary(uncached_texts[i]), 3600*24, pickle.dumps(emb))

    chunk_id_to_idx = {chunk.chunk_id: i for i, chunk in enumerate(chunks)}
    return chunks, np.vstack(embeddings), chunk_id_to_idx

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    embeddings = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype(np.float32))
    return index

def bm25_prefilter(question: str, chunks: List[Chunk], top_k: int = BM25_PREFILTER_K) -> List[int]:
    tokenized_corpus = [chunk.text.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = question.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return top_indices.tolist()

async def search_similar_ultra_fast(question: str, faiss_index, chunks: List[Chunk], chunk_embeddings: np.ndarray, top_k: int = TOP_K) -> List[Chunk]:
    candidate_indices = bm25_prefilter(question, chunks)
    candidate_embeddings = chunk_embeddings[candidate_indices]
    q_embedding = await get_embeddings_openai_batch([question])
    q_embedding = np.array([q_embedding[0]], dtype=np.float32)
    candidate_embeddings = np.array(candidate_embeddings).astype("float32")
    faiss.normalize_L2(q_embedding)

    temp_index = faiss.IndexFlatIP(candidate_embeddings.shape[1])
    faiss.normalize_L2(candidate_embeddings)
    temp_index.add(candidate_embeddings.astype(np.float32))
    scores, indices = temp_index.search(q_embedding.astype(np.float32), top_k)
    final_indices = [candidate_indices[i] for i in indices[0] if i < len(candidate_indices)]
    return [chunks[i] for i in final_indices]

async def generate_answer_fast(question: str, relevant_chunks: List[Chunk]) -> str:
    if not relevant_chunks:
        return "No relevant information found."
    prompt = "You are an expert assistant. Using the context excerpts below, answer the question clearly and briefly. Think step-by-step.\n\n"
    for chunk in relevant_chunks:
        prompt += f"[Source: {chunk.source}, page: {chunk.page}]\n{chunk.text[:350]}...\n\n"
    prompt += f"Question: {question}\nAnswer:"

    response = await openai_client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=350
    )
    return response.choices[0].message.content.strip()

@app.post("/api/v1/hackrx/run")
async def run_ultra_fast(data: QueryRequest, authorization: str = Header(...)):
    verify_token(authorization)
    start_time = time.time()
    url = data.documents
    responses = {}

    unanswered = [q for q in data.questions if q not in responses]

    if unanswered:
        chunks, embeddings, _ = await pre_cache_pdf_embeddings(url, "user_pdf")
        faiss_index = build_faiss_index(embeddings)

        async def answer_one(question):
            relevant_chunks = await search_similar_ultra_fast(question, faiss_index, chunks, embeddings)
            answer = await generate_answer_fast(question, relevant_chunks)
            return question, answer

        results = await asyncio.gather(*[answer_one(q) for q in unanswered])
        responses.update(dict(results))

    total_time = time.time() - start_time
    return {
        "answers": responses,
        "time_taken": total_time
    }

@app.get("/health")
async def health():
    return {"status": "ready"}

if __name__ == "__main__":
    uvicorn.run("rag:app", host="0.0.0.0", port=8000, reload=False, workers=2)
