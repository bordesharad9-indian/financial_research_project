from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pipeline.pipeline import RAGPipelineOrchestrator

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"

app = FastAPI(title="Financial Research Assistant API", version="1.0")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# Uses the existing "Data" folder and its vector_store (same as your repo)
rag = RAGPipelineOrchestrator(
    data_directory=str(DATA_DIR),
    persist_directory=str(DATA_DIR / "vector_store"),
)


class IngestRequest(BaseModel):
  pdf_path: str
  company: str
  year: int


class ResearchRequest(BaseModel):
  query: str
  companies: list[str] = []


@app.get("/")
def read_index():
  return FileResponse(BASE_DIR / "static" / "index.html")


@app.post("/ingest")
def ingest_report(req: IngestRequest):
  pdf = Path(req.pdf_path)
  if not pdf.is_absolute():
    pdf = BASE_DIR / pdf
  if not pdf.exists():
    raise HTTPException(status_code=404, detail=f"PDF file not found at {req.pdf_path}")
  try:
    pages = PyPDFLoader(str(pdf)).load()
    for p in pages:
      p.metadata.update(
          {"source_file": pdf.name, "file_type": "pdf",
           "company": req.company, "year": req.year}
      )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(pages)
    embeddings = rag.embedding_manager.generate_embeddings(
        [c.page_content for c in chunks]
    )
    rag.vector_store.add_documents(chunks, embeddings)
    return {
        "status": "success",
        "indexed_chunks": len(chunks),
        "company": req.company,
        "year": req.year,
    }
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))


@app.post("/research")
def execute_research(req: ResearchRequest):
  try:
    docs = rag.retriever.retrieve(req.query, top_k=15)
    wanted = {c.lower() for c in req.companies}
    if wanted:
      # keep chunks tagged with a selected company, or untagged ones
      # (PDFs indexed in bulk have no company metadata)
      docs = [
          d for d in docs
          if not d["metadata"].get("company")
          or d["metadata"]["company"].lower() in wanted
      ]
    docs = docs[:5]
    if not docs:
      return {"query": req.query, "report": "No relevant documents found.", "sources": []}
    if not rag.llm:
      raise HTTPException(status_code=503, detail="Groq LLM not initialized. Set GROQ_API_KEY in .env")
    context = "\n\n---\n\n".join(d["content"] for d in docs)
    report = rag.llm.generate_response(query=req.query, context=context)
    sources = [
        {
            "source_file": d["metadata"].get("source_file", "unknown"),
            "page": d["metadata"].get("page"),
            "company": d["metadata"].get("company"),
            "year": d["metadata"].get("year"),
            "score": round(d["similarity_score"], 3),
        }
        for d in docs
    ]
    return {"query": req.query, "report": report, "sources": sources}
  except HTTPException:
    raise
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
  import uvicorn

  uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)