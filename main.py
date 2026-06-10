from pathlib import Path

from dotenv import load_dotenv
from typing import Annotated
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pypdf import PdfReader as PdfPageCounter

from agno.agent import Agent
from agno.knowledge import Knowledge
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.knowledge.reader.pdf_reader import PDFReader
from agno.models.groq import Groq
from agno.vectordb.lancedb.lance_db import LanceDb
from crud import (
    create_document,
    delete_document,
    get_document,
    init_db,
    read_documents,
    update_document_name,
)

load_dotenv()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "tmp" / "lancedb"
DATA_DIR.mkdir(exist_ok=True)

vector_db = LanceDb(
    table_name="my_pdf_docs",
    uri=str(DB_DIR),
    embedder=OpenAIEmbedder(id="text-embedding-3-small"),
)

knowledge_base = Knowledge(
    vector_db=vector_db,
    readers={".pdf": PDFReader()},
)

agent = Agent(
    model=Groq(id="llama-3.1-8b-instant"),
    markdown=False,
    instructions="Answer ONLY using the context provided. If the answer is not in the context, say 'I don't know'.",
)


# ── Pydantic models (shapes of request bodies) ────────────────────────────────

class RenameBody(BaseModel):
    filename: str


class ChatMessage(BaseModel):
    message: str


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(title="RAG Chatbot API")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
def startup():
    init_db()
    _bootstrap_existing_docs()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_page_count(filepath: Path) -> int:
    try:
        return len(PdfPageCounter(filepath).pages)
    except Exception:
        return 0


def _doc_stem(filename: str) -> str:
    """The name LanceDB stores for a PDF — stem with spaces replaced by underscores."""
    return Path(filename).stem.replace(" ", "_")


def _bootstrap_existing_docs():
    """Backfill SQLite from /data/ on first run. Idempotent — skips already-recorded files."""
    for f in DATA_DIR.glob("*.pdf"):
        try:
            create_document(f.name, f.stat().st_size, _get_page_count(f))
        except Exception:
            pass  # UNIQUE constraint fires if already recorded — that's fine


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return FileResponse("static/index.html")


# CREATE — upload one or more PDFs
@app.post("/documents", status_code=201)
async def upload_documents(files: Annotated[list[UploadFile], File()]):
    created = []
    for upload in files:
        dest = DATA_DIR / upload.filename
        dest.write_bytes(await upload.read())
        knowledge_base.insert(path=str(dest), skip_if_exists=True)
        doc_id = create_document(upload.filename, dest.stat().st_size, _get_page_count(dest))
        created.append({"id": doc_id, "filename": upload.filename})
    return {"uploaded": created}


# READ — list all documents
@app.get("/documents")
def list_documents():
    return read_documents()


# UPDATE — rename a document
@app.put("/documents/{doc_id}")
def rename_document(doc_id: int, body: RenameBody):
    doc = get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    new_name = body.filename if body.filename.endswith(".pdf") else body.filename + ".pdf"
    old_path = DATA_DIR / doc["filename"]
    new_path = DATA_DIR / new_name

    if old_path.exists():
        old_path.rename(new_path)                               # 1. Disk

    vector_db.delete_by_name(_doc_stem(doc["filename"]))       # 2a. Remove old vectors
    knowledge_base.insert(path=str(new_path), skip_if_exists=False)  # 2b. Re-index
    update_document_name(doc_id, new_name)                     # 3. SQLite

    return {"id": doc_id, "filename": new_name}


# DELETE — remove a document
@app.delete("/documents/{doc_id}")
def remove_document(doc_id: int):
    deleted = delete_document(doc_id)                          # 1. SQLite
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")

    filepath = DATA_DIR / deleted["filename"]
    if filepath.exists():
        filepath.unlink()                                      # 2. Disk

    vector_db.delete_by_name(_doc_stem(deleted["filename"]))  # 3. LanceDB
    return {"deleted": deleted["filename"]}


MIN_PAGE_COUNT = 3

# CHAT — ask a question about the indexed documents
@app.post("/chat")
def chat(body: ChatMessage):
    if not read_documents():
        return {"reply": "No documents uploaded yet. Please upload a PDF first."}

    results = knowledge_base.search(query=body.message, max_results=5)
    if not results:
        return {"reply": "I don't know."}

    context = "\n\n".join(r.content for r in results)

    prompt = f"""Using ONLY the context below, answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {body.message}
Answer:"""

    response = agent.run(prompt)
    return {"reply": response.content}
