import os
from dotenv import load_dotenv
from ingredients_db import INGREDIENTS
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

EMBEDDER    = SentenceTransformer("all-MiniLM-L6-v2")
RERANKER    = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
QDRANT      = QdrantClient(":memory:")
COLL        = "etqa_prod"
RETRIEVE_K  = 15
RERANK_TOP  = 4

SYSTEM = """You are a halal food expert. Answer ONLY using the provided context.
Be specific about E-numbers and aliases.
If the context does not contain enough information say: "I need more data on this ingredient."
Never speculate beyond the context."""


# ── STAGE 1: BUILD INDEX ──────────────────────────────────────
def build_index():
    sp = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=60)
    QDRANT.create_collection(COLL, vectors_config=VectorParams(size=384, distance=Distance.COSINE))
    pts, idx = [], 0
    for ing in INGREDIENTS:
        for chunk in sp.split_text(ing["text"]):
            pts.append(PointStruct(
                id=idx,
                vector=EMBEDDER.encode(chunk).tolist()
                payload={"text": chunk, "name": ing["name"], "status": ing["status"]}
            ))
            idx += 1
    QDRANT.upsert(COLL, points=pts)
    print(f"[index] {idx} chunks indexed")


# ── STAGE 2: RETRIEVE ─────────────────────────────────────────
def retrieve(query: str) -> list[dict]:
    qv      = EMBEDDER.encode(query).tolist()
    results = QDRANT.query_points(COLL, query=qv, limit=RETRIEVE_K).points
    return [
        {"text": r.payload["text"], "name": r.payload["name"],
         "status": r.payload["status"], "score": r.score}
        for r in results
    ]


# ── STAGE 3: RE-RANK ──────────────────────────────────────────
def rerank(query: str, candidates: list[dict]) -> list[dict]:
    scores = RERANKER.predict([[query, c["text"]] for c in candidates])
    for c, s in zip(candidates, scores):
        c["rerank_score"] = s
    return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:RERANK_TOP]


# ── STAGE 4: GENERATE ─────────────────────────────────────────
def generate(query: str, top_chunks: list[dict]) -> str:
    ctx    = "\n\n".join([f"-- {c['name']} --\n{c['text']}" for c in top_chunks])
    prompt = f"Context:\n{ctx}\n\nQuestion: {query}"

    # ── OPTION A: Anthropic (uncomment to use) ───────────────
    # import anthropic
    # llm  = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    # resp = llm.messages.create(
    #     model="claude-haiku-4-5-20251001", max_tokens=400,
    #     system=SYSTEM, messages=[{"role": "user", "content": prompt}]
    # )
    # return resp.content[0].text

    # ── OPTION B: Ollama — local, free, no API key ───────────
    # import ollama
    # resp = ollama.chat(
    #     model="llama3.1:8b",
    #     messages=[{"role": "system", "content": SYSTEM},
    #               {"role": "user",   "content": prompt}]
    # )
    # return resp.message.content

    # ── OPTION C: Groq — free tier, very fast ────────────────
    import groq
    llm  = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))
    resp = llm.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        max_tokens=400,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": prompt}
        ]
    )
    return resp.choices[0].message.content


# ── ORCHESTRATOR ──────────────────────────────────────────────
def ask(question: str, verbose: bool = False) -> str:
    candidates = retrieve(question)
    top_chunks = rerank(question, candidates)
    if verbose:
        print(f"  [sources] {[c['name'] for c in top_chunks]}")
    return generate(question, top_chunks)


# ── RUN ───────────────────────────────────────────────────────
if __name__ == "__main__":
    build_index()
    questions = [
        "Can a Muslim consumer eat gummy bears that contain E441?",
        "What E-numbers in snack food flavoring are uncertain for halal status?",
        "Which preservatives in bread are definitely halal?",
        "What should I look for on a candy label to avoid insect-derived ingredients?",
    ]
    for q in questions:
        print(f"\n{'='*60}\nQ: {q}")
        print(f"A: {ask(q, verbose=True)}")
