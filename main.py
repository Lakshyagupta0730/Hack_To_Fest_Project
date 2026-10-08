import os
import re
import uuid
import math
import logging
from collections import Counter
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from pypdf import PdfReader
from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("paper_explorer")

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Paper Explorer Online Backend", version="2.0")

# Enable CORS for local HTML page or file:// access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()

ai_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY if OPENROUTER_API_KEY else "dummy-key",
    timeout=30.0,
)

# Lightweight pure-Python BM25 vector index (<25MB RAM, instant boot)
class LightweightVectorStore:
    def __init__(self):
        self.papers = {}
        self.doc_freqs = {}

    def tokenize(self, text: str) -> List[str]:
        return [w for w in re.findall(r'\b[a-zA-Z0-9_-]+\b', text.lower()) if len(w) > 1]

    def add(self, paper_id: str, chunks: list):
        if paper_id not in self.papers:
            self.papers[paper_id] = []
            self.doc_freqs[paper_id] = Counter()

        for c in chunks:
            tokens = self.tokenize(c["text"])
            tf = Counter(tokens)
            self.papers[paper_id].append({
                "id": c["id"],
                "text": c["text"],
                "section": c["section"],
                "page": c["page"],
                "tf": tf,
                "len": len(tokens)
            })
            for term in set(tokens):
                self.doc_freqs[paper_id][term] += 1

    def query(self, paper_id: str, query_text: str, top_k: int = 5):
        if paper_id not in self.papers:
            return [], []
        chunks = self.papers[paper_id]
        N = len(chunks)
        if N == 0:
            return [], []

        q_tokens = self.tokenize(query_text)
        if not q_tokens:
            return [c["text"] for c in chunks[:top_k]], chunks[:top_k]

        scores = []
        for c in chunks:
            score = 0.0
            for t in q_tokens:
                if t in c["tf"]:
                    df = self.doc_freqs[paper_id].get(t, 1)
                    idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
                    tf = c["tf"][t]
                    score += idf * (tf * 2.2) / (tf + 1.2 * (0.25 + 0.75 * (c["len"] / 60.0)))
            scores.append((score, c))

        scores.sort(key=lambda x: x[0], reverse=True)
        top = [s[1] for s in scores[:top_k]]
        return [c["text"] for c in top], top

vector_store = LightweightVectorStore()

# Optional dense vector embedding via ChromaDB & SentenceTransformer
USE_CHROMA = False
embedding_model = None
collection = None

try:
    if os.getenv("DISABLE_TORCH", "false").lower() != "true":
        logger.info("Attempting to load SentenceTransformer model...")
        from sentence_transformers import SentenceTransformer
        import chromadb
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        chroma_client = chromadb.Client()
        collection = chroma_client.get_or_create_collection(name="paper_chunks")
        USE_CHROMA = True
        logger.info("SentenceTransformer and ChromaDB loaded successfully.")
except Exception as e:
    logger.warning(f"Dense vector model unavailable ({e}); utilizing high-efficiency vector index.")
    USE_CHROMA = False

# In-memory store for indexed paper metadata
PAPER_STORE: Dict[str, Dict[str, Any]] = {}

SAMPLE_PAPERS: Dict[str, Dict[str, Any]] = {
    "transformer": {
        "id": "transformer",
        "title": "Attention Is All You Need",
        "authors": "Vaswani et al. (Google Brain / Research)",
        "year": "2017",
        "file": str(BASE_DIR / "sample_papers" / "attention_is_all_you_need.pdf"),
        "description": "Introduced the Transformer architecture, replacing recurrent and convolutional networks with Multi-Head Self-Attention for state-of-the-art sequence modeling.",
        "sample_questions": [
            "What is the mathematical formula for Scaled Dot-Product Attention?",
            "How does the Multi-Head Attention mechanism work?",
            "What BLEU score does the Big Transformer model achieve on English-to-German?",
            "Why do the authors use positional encodings and how are they calculated?"
        ]
    },
    "resnet": {
        "id": "resnet",
        "title": "Deep Residual Learning for Image Recognition",
        "authors": "He, Zhang, Ren, Sun (Microsoft Research)",
        "year": "2015",
        "file": str(BASE_DIR / "sample_papers" / "resnet.pdf"),
        "description": "Introduced Deep Residual Learning with skip connections, solving the degradation problem and enabling training of networks over 100 layers deep.",
        "sample_questions": [
            "What is the degradation problem that deep residual learning addresses?",
            "How is the residual block formulated mathematically (F(x) + x)?",
            "What top-5 error rate did ResNet-152 achieve on the ImageNet test set?",
            "Why do identity shortcut connections add no extra parameters?"
        ]
    },
    "ddpm": {
        "id": "ddpm",
        "title": "Denoising Diffusion Probabilistic Models",
        "authors": "Ho, Jain, Abbeel (UC Berkeley)",
        "year": "2020",
        "file": str(BASE_DIR / "sample_papers" / "ddpm.pdf"),
        "description": "Showed that diffusion probabilistic models can synthesize high quality images, establishing deep connections to denoising score matching.",
        "sample_questions": [
            "What is the difference between the forward and reverse diffusion processes?",
            "What simplified loss objective (L_simple) is used for training?",
            "What FID score did the model achieve on unconditional CIFAR-10?",
            "How is the noise variance schedule beta_t scheduled across timesteps?"
        ]
    }
}

class QuestionRequest(BaseModel):
    paper_id: str
    question: str
    api_key: Optional[str] = None

class Citation(BaseModel):
    section: str
    page: Optional[int] = None
    content: str

class AnswerResponse(BaseModel):
    answer: str
    citations: List[Citation]
    model_used: Optional[str] = None


def extract_clean_title(reader: PdfReader, default_title: str) -> str:
    """Attempts to extract a sensible paper title from the first page."""
    try:
        first_page = reader.pages[0].extract_text() or ""
        lines = [l.strip() for l in first_page.split("\n") if l.strip()]
        disclaimers = ("provided proper attribution", "arxiv:", "http", "doi:", "copyright", "submitted to", "under review", "preprint", "proceedings")
        for line in lines[:8]:
            lower = line.lower()
            if len(line) >= 4 and not any(lower.startswith(d) for d in disclaimers):
                if not line.isdigit() and len(line) < 140:
                    return line
    except Exception as e:
        logger.warning(f"Title extraction fallback: {e}")
    return default_title


def parse_pdf_and_chunk(file_path: str, paper_id: str):
    """Parses a PDF into semantic chunks tagged with sections and page numbers."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found at {file_path}")

    reader = PdfReader(file_path)
    total_pages = len(reader.pages)
    chunks = []
    current_section = "Abstract / Introduction"
    detected_sections = []

    section_pattern = re.compile(r'^(?:[0-9]+(?:\.[0-9]+)*\s+[A-Z][A-Za-z0-9\s\-]+|[A-Z\s]{4,35})$')

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        lines = text.split("\n")
        chunk_text = ""

        for line in lines:
            trimmed = line.strip()
            # Check for section header
            if section_pattern.match(trimmed) and 3 < len(trimmed) < 55 and not trimmed.isdigit():
                current_section = trimmed
                if current_section not in detected_sections:
                    detected_sections.append(current_section)
            elif trimmed.isupper() and 3 < len(trimmed) < 40 and not trimmed.isdigit():
                current_section = trimmed
                if current_section not in detected_sections:
                    detected_sections.append(current_section)

            chunk_text += line + " "
            if len(chunk_text) > 480:
                chunks.append({
                    "id": f"{paper_id}_p{page_num}_{uuid.uuid4().hex[:6]}",
                    "text": chunk_text.strip(),
                    "section": current_section,
                    "page": page_num + 1
                })
                chunk_text = chunk_text[-60:]

        if len(chunk_text.strip()) > 30:
            chunks.append({
                "id": f"{paper_id}_p{page_num}_{uuid.uuid4().hex[:6]}",
                "text": chunk_text.strip(),
                "section": current_section,
                "page": page_num + 1
            })

    title = extract_clean_title(reader, Path(file_path).stem.replace("_", " ").title())
    return chunks, title, total_pages, detected_sections


def index_chunks(paper_id: str, title: str, chunks: list, total_pages: int, sections: list, sample_questions: list = None):
    """Indexes chunks into vector storage (both ChromaDB if active and Lightweight index)."""
    if not chunks:
        raise ValueError("No text chunks could be extracted from this document.")

    # Always index into lightweight store
    vector_store.add(paper_id, chunks)

    # Also add to ChromaDB if available
    if USE_CHROMA and collection is not None and embedding_model is not None:
        try:
            documents = [c["text"] for c in chunks]
            ids = [c["id"] for c in chunks]
            metadatas = [{"paper_id": paper_id, "section": c["section"], "page": c["page"]} for c in chunks]
            embeddings = embedding_model.encode(documents, show_progress_bar=False).tolist()
            collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
        except Exception as e:
            logger.warning(f"Dense Chroma indexing error: {e}")

    if not sample_questions:
        sample_questions = [
            f"What is the primary contribution of {title}?",
            "What is the core architecture or methodology proposed?",
            "What are the main experimental results and quantitative metrics?",
            "What datasets and hyper-parameters were used for training?"
        ]

    PAPER_STORE[paper_id] = {
        "paper_id": paper_id,
        "title": title,
        "total_pages": total_pages,
        "total_chunks": len(chunks),
        "sections": sections[:25] if sections else ["Overview", "Methodology", "Results"],
        "sample_questions": sample_questions
    }
    return PAPER_STORE[paper_id]


# --- STATIC FILE ROUTES ---
@app.get("/")
@app.get("/index.html")
async def serve_index():
    index_file = BASE_DIR / "index.html"
    if not index_file.exists():
        index_file = BASE_DIR / "index2.html"
    return FileResponse(index_file)

@app.get("/index2.html")
async def serve_index2():
    return FileResponse(BASE_DIR / "index.html" if (BASE_DIR / "index.html").exists() else BASE_DIR / "index2.html")

@app.get("/style.css")
async def serve_css():
    return FileResponse(BASE_DIR / "style.css")

@app.get("/index.js")
async def serve_js():
    return FileResponse(BASE_DIR / "index.js")


# --- API ROUTES ---
@app.get("/health")
async def health_check():
    total_chunks = collection.count() if USE_CHROMA and collection is not None else sum(len(c) for c in vector_store.papers.values())
    return {
        "status": "healthy",
        "active_papers": len(PAPER_STORE),
        "total_chunks_indexed": total_chunks,
        "loaded_paper_ids": list(PAPER_STORE.keys()),
        "dense_embeddings_active": USE_CHROMA
    }

@app.get("/sample-papers")
async def get_sample_papers():
    return [
        {
            "id": v["id"],
            "title": v["title"],
            "authors": v["authors"],
            "year": v["year"],
            "description": v["description"],
            "sample_questions": v["sample_questions"]
        }
        for v in SAMPLE_PAPERS.values()
    ]

@app.post("/load-sample/{sample_id}")
async def load_sample_paper(sample_id: str):
    if sample_id not in SAMPLE_PAPERS:
        raise HTTPException(status_code=404, detail=f"Sample paper '{sample_id}' not found.")

    if sample_id in PAPER_STORE:
        return {
            "message": f"Sample paper '{SAMPLE_PAPERS[sample_id]['title']}' loaded from cache.",
            "paper_id": sample_id,
            **PAPER_STORE[sample_id]
        }

    sample_meta = SAMPLE_PAPERS[sample_id]
    file_path = sample_meta["file"]

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Sample paper PDF missing on disk: {file_path}")

    try:
        chunks, extracted_title, total_pages, sections = parse_pdf_and_chunk(file_path, sample_id)
        info = index_chunks(
            paper_id=sample_id,
            title=sample_meta["title"],
            chunks=chunks,
            total_pages=total_pages,
            sections=sections,
            sample_questions=sample_meta["sample_questions"]
        )
        return {
            "message": f"Sample paper '{info['title']}' successfully processed and indexed.",
            **info
        }
    except Exception as e:
        logger.error(f"Error indexing sample paper: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to index sample paper: {str(e)}")


@app.post("/upload-paper")
async def upload_paper(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    paper_id = f"paper_{uuid.uuid4().hex[:8]}"
    temp_filename = BASE_DIR / f"temp_{paper_id}.pdf"

    try:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        with open(temp_filename, "wb") as f:
            f.write(content)

        chunks, detected_title, total_pages, sections = parse_pdf_and_chunk(str(temp_filename), paper_id)
        
        display_title = detected_title if detected_title != "Research Paper" else Path(file.filename).stem
        info = index_chunks(
            paper_id=paper_id,
            title=display_title,
            chunks=chunks,
            total_pages=total_pages,
            sections=sections
        )

        return {
            "message": "Paper successfully processed and indexed.",
            **info
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing uploaded PDF: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")
    finally:
        if os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except Exception:
                pass


@app.get("/paper-info/{paper_id}")
async def get_paper_info(paper_id: str):
    if paper_id in PAPER_STORE:
        return PAPER_STORE[paper_id]
    raise HTTPException(status_code=404, detail="Paper not found.")


@app.post("/ask-question", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    retrieved_docs = []
    retrieved_meta = []

    # 1. Attempt ChromaDB dense search if available
    if USE_CHROMA and collection is not None and embedding_model is not None:
        try:
            query_embedding = embedding_model.encode([request.question.strip()]).tolist()
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=5,
                where={"paper_id": request.paper_id}
            )
            if results.get("documents") and len(results["documents"]) > 0:
                retrieved_docs = results["documents"][0]
                retrieved_meta = results["metadatas"][0]
        except Exception as e:
            logger.warning(f"Dense vector query failed: {e}")

    # 2. Fallback to lightweight vector store
    if not retrieved_docs:
        docs, metas = vector_store.query(request.paper_id, request.question.strip(), top_k=5)
        retrieved_docs = docs
        retrieved_meta = metas

    if not retrieved_docs:
        raise HTTPException(status_code=404, detail="No relevant context found for this paper. Please check the paper ID or ask a related question.")

    context_blocks = []
    citations = []

    for doc, meta in zip(retrieved_docs, retrieved_meta):
        page_num = meta.get("page", 1) if isinstance(meta, dict) else getattr(meta, "page", 1)
        sec_name = meta.get("section", "General") if isinstance(meta, dict) else getattr(meta, "section", "General")
        section_label = f"Page {page_num} ({sec_name})"
        context_blocks.append(f"[{section_label}]:\n{doc}")
        citations.append(Citation(
            section=sec_name,
            page=page_num,
            content=doc.strip()
        ))

    context_str = "\n\n".join(context_blocks)
    paper_title = PAPER_STORE.get(request.paper_id, {}).get("title", "the research paper")

    prompt = f"""You are an expert AI research scientist assisting a researcher. 
Answer the question thoroughly, clearly, and accurately based ONLY on the provided excerpts from '{paper_title}'.

Guidelines:
1. Ground your answer strictly in the facts, methodology, equations, and experimental results provided in the context below.
2. Directly reference the source sections and pages (e.g. "According to Page 3, Section 3.2...").
3. Format your response cleanly using Markdown: use bolding for key terms, bullet points for multi-part explanations, and clear mathematical notation where appropriate.
4. If the provided context does not have sufficient information to answer a specific detail, state what is known from the context and clearly acknowledge the limitation.

--- PAPER CONTEXT ---
{context_str}

--- RESEARCH QUESTION ---
{request.question}

--- GROUNDED ANSWER ---"""

    # Determine API key to use (passed from frontend or environment)
    active_key = (request.api_key or "").strip() or os.getenv("OPENROUTER_API_KEY", "").strip()

    candidate_models = [
        "openrouter/free",
        "qwen/qwen3.8-27b:free",
        "liquid/lfm-2.5-2.6b:free",
        "google/gemini-2.0-flash-exp:free"
    ]

    answer_text = None
    model_used = None

    if active_key:
        try:
            custom_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=active_key,
                timeout=30.0,
            )
            for model_name in candidate_models:
                try:
                    logger.info(f"Calling OpenRouter with model: {model_name}")
                    response = custom_client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    if response.choices and len(response.choices) > 0:
                        answer_text = response.choices[0].message.content
                        model_used = model_name
                        break
                except Exception as e:
                    logger.warning(f"Model {model_name} failed: {e}")
                    continue
        except Exception as e:
            logger.warning(f"OpenRouter client initialization error: {e}")

    # Resilient grounded fallback: construct rich, grounded response from retrieved chunks
    if not answer_text:
        logger.info("Constructing high-accuracy grounded response from retrieved chunks.")
        model_used = "Local Grounded Engine (ChromaDB / BM25)"
        
        findings = []
        for i, c in enumerate(citations):
            clean_excerpt = " ".join(c.content.split())
            if len(clean_excerpt) > 400:
                clean_excerpt = clean_excerpt[:400] + "..."
            findings.append(f"**[{i+1}] {c.section} (Page {c.page})**\n> \"{clean_excerpt}\"")

        answer_text = (
            f"### Grounded Findings from '{paper_title}'\n\n"
            f"Based on the indexed document sections and semantic retrieval for: **\"{request.question}\"**:\n\n"
            + "\n\n".join(findings) +
            "\n\n---\n*💡 **Note:** Grounded directly from verified document chunks. You can also configure an OpenRouter API key in the header to enable synthesized LLM reasoning.*"
        )

    return AnswerResponse(
        answer=answer_text,
        citations=citations,
        model_used=model_used
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)