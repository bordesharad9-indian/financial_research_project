import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pipeline.agent_workflow import FinancialResearchGraph
from pipeline.pipeline import FinancialDocumentPipeline
from pipeline.vector_store import FinancialVectorStoreManager

app = FastAPI(title="Financial Research Assistant API", version="1.0")

app.mount("/static", StaticFiles(directory="static"), name="static")

db_manager = FinancialVectorStoreManager()
research_graph = FinancialResearchGraph(db_manager)
pipeline = FinancialDocumentPipeline()


class IngestRequest(BaseModel):
  pdf_path: str
  company: str
  year: int


class ResearchRequest(BaseModel):
  query: str
  companies: list[str]


@app.get("/")
def read_index():
  return FileResponse("static/index.html")


@app.post("/ingest")
def ingest_report(req: IngestRequest):
  if not os.path.exists(req.pdf_path):
    raise HTTPException(
        status_code=404, detail=f"PDF file not found at {req.pdf_path}"
    )
  try:
    markdown = pipeline.parse_pdf(req.pdf_path)
    chunks = pipeline.chunk_document(markdown, req.company, req.year)
    db_manager.add_chunks(chunks)
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
    initial_state = {
        "user_query": req.query,
        "target_companies": req.companies,
        "retrieved_context": [],
        "draft_report": "",
        "messages": [],
    }
    result = research_graph.app.invoke(initial_state)
    return {
        "query": req.query,
        "report": result["draft_report"],
        "sources": [c["metadata"] for c in result["retrieved_context"]],
    }
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
  import uvicorn

  uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)