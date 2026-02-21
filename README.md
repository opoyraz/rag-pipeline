# Production RAG Pipeline

A complete Retrieval-Augmented Generation (RAG) pipeline built with Python. Given a question, the pipeline retrieves the most relevant documents from a vector database, re-ranks them for precision, then generates a grounded answer using an LLM — with zero hallucination beyond the provided context.

## Architecture

```
Question
   │
   ▼
[STAGE 1 — INDEX] (runs once at startup)
   Ingredient texts → chunked → embedded → stored in Qdrant (:memory:)
   │
   ▼
[STAGE 2 — RETRIEVE]
   Embed question → ANN search in Qdrant → top 15 candidate chunks
   │
   ▼
[STAGE 3 — RE-RANK]
   CrossEncoder scores each (question, chunk) pair → keep top 4
   │
   ▼
[STAGE 4 — GENERATE]
   Top 4 chunks + question → LLM → grounded answer
```

## Why RAG?

| Problem | Without RAG | With RAG |
|---|---|---|
| Hallucination | LLM invents facts | Answer grounded in retrieved docs |
| Private data | LLM has no access | Your documents are the source |
| Knowledge cutoff | Stale training data | Always reflects your latest data |

## Tech Stack

| Component | Tool | Purpose |
|---|---|---|
| **Chunking** | `langchain-text-splitters` | Split documents into retrievable pieces |
| **Embedding** | `all-MiniLM-L6-v2` | Encode text into 384-dim vectors |
| **Vector DB** | `Qdrant (:memory:)` | Store and search vectors with ANN |
| **Re-ranker** | `ms-marco-MiniLM-L-6-v2` | Precision scoring with CrossEncoder |
| **LLM** | Groq / Anthropic / Ollama | Generate the final answer |

## Pipeline Stages in Detail

### Stage 1 — Build Index
Runs once at startup. Each ingredient entry is:
1. Split into overlapping chunks (200 chars, 60 char overlap) using `RecursiveCharacterTextSplitter`
2. Encoded into a 384-dimensional vector using `all-MiniLM-L6-v2`
3. Stored in Qdrant in-memory with its text and metadata as payload

### Stage 2 — Retrieve
For each user question:
1. The question is embedded using the same model as indexing (critical — must match)
2. Qdrant performs Approximate Nearest Neighbor (ANN) search using HNSW
3. Returns the top 15 most semantically similar chunks

### Stage 3 — Re-rank
The top 15 candidates are re-scored using a CrossEncoder (`ms-marco-MiniLM-L-6-v2`):
- A CrossEncoder reads the question and chunk **together** — capturing token-level interactions
- More accurate than the bi-encoder used in retrieval, but slower (runs on top-k only)
- Top 4 chunks by CrossEncoder score are passed to the LLM

### Stage 4 — Generate
The top 4 chunks are assembled into a context block and sent to the LLM with a strict system prompt that instructs it to answer only from the provided context.

## Qdrant In-Memory Mode

```python
QDRANT = QdrantClient(":memory:")
```

Qdrant runs entirely inside the Python process — no server, no Docker required. Data lives in RAM and is reset on each run. This is intentional for a self-contained demo.

**To persist data between runs**, switch to disk mode:
```python
QDRANT = QdrantClient(path="./qdrant_db")
```

**For production**, connect to a Qdrant server or Qdrant Cloud:
```python
QDRANT = QdrantClient(host="localhost", port=6333)
# or
QDRANT = QdrantClient(url="https://your-cluster.qdrant.io", api_key="...")
```

## Setup

**1. Clone and install dependencies**
```bash
git clone https://github.com/opoyraz/rag-pipeline.git
cd rag-pipeline
pip install -r requirements.txt
```

**2. Configure API keys**
```bash
cp .env.example .env
# Edit .env and add your keys
```

**3. Run**
```bash
python pipeline.py
```

## LLM Options

The `generate()` function supports three LLM backends. Uncomment the one you want to use:

| Option | Setup | Cost |
|---|---|---|
| **Groq** (default) | `GROQ_API_KEY` in `.env` — free tier at [console.groq.com](https://console.groq.com) | Free |
| **Ollama** | `ollama pull llama3.1:8b` — runs locally | Free |
| **Anthropic** | `ANTHROPIC_API_KEY` in `.env` | Paid |

### Groq model options
```python
"meta-llama/llama-4-scout-17b-16e-instruct"   # default
"llama-3.3-70b-versatile"                      # higher quality
"mixtral-8x7b-32768"                           # 32K context
"gemma2-9b-it"                                 # Google Gemma
```

## Example Output

```
============================================================
Q: Can a Muslim consumer eat gummy bears that contain E441?
  [sources] ['Gelatin', 'Gelatin', 'Carrageenan', 'Xanthan Gum']
A: E441 is gelatin — a mashbooh (doubtful) ingredient. Its halal
   status depends on the source animal and slaughter method used.
   Pork-derived gelatin is haram. Halal-certified beef gelatin is
   permissible. Look for gummies labeled halal-certified or using
   carrageenan (E407) or xanthan gum (E415) as alternatives.
```

## Key Concepts

**Why re-ranking?**
Vector search (bi-encoder) is fast but approximate — it encodes the question and document independently. A CrossEncoder reads them together, catching interactions the bi-encoder misses. The two-stage approach (retrieve top 15, re-rank to top 4) gives both speed and precision.

**Why chunking matters?**
If a document is too long, the embedding averages over too many topics and loses specificity. Splitting into smaller chunks means each vector represents a focused concept, improving retrieval accuracy.

**Why `:memory:` for this demo?**
The ingredient database is small and static. Loading it fresh each run takes under a second. For larger datasets (millions of documents), you would use persistent Qdrant with pre-built indexes.
