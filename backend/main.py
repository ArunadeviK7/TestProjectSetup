
from fastapi import UploadFile, File, Form, FastAPI, Request
import os
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from design_agent import design_workflow
from execution_agent import execution_workflow
from insights_agent import insights_workflow
from regulations_injestion_agent import regulations_ingestion_workflow


print('[Backend] Starting FastAPI app...')
app = FastAPI()
print('[Backend] Endpoints will be registered.')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/design")
async def design_stage(data: dict):
    print("[Backend] /design endpoint called.")
    print("[Backend] Received data at /design:", data)
    result = design_workflow(data)
    print("[Backend] Returning from /design:", result)
    return result


@app.post("/execution")
async def execution_stage(data: dict):
    print("[Backend] /execution endpoint called.")
    print("[Backend] Received data at /execution:", data)
    result = execution_workflow(data)
    print("[Backend] Returning from /execution:", result)
    return result


@app.post("/insights")
async def insights_stage(data: dict):
    print("[Backend] /insights endpoint called.")
    print("[Backend] Received data at /insights:", data)
    result = insights_workflow(data)
    print("[Backend] Returning from /insights:", result)
    return result


@app.post("/upload_regulation")
async def upload_regulation(
    name: str = Form(...),
    region: str = Form(...),
    effectiveFrom: str = Form(...),
    document: UploadFile = File(...)
):
    # Read uploaded document content
    file_content = (await document.read()).decode("utf-8")
    file_name = document.filename
    # Trigger agentic ingestion workflow with file content
    try:
        agent = regulations_ingestion_workflow(file_content, file_name)
        graph = agent.graph
        article_count = graph.evaluate("MATCH (a:Article) RETURN count(a)")
        facet_count = graph.evaluate("MATCH (f:Facet) RETURN count(f)")
        action_count = graph.evaluate(
            "MATCH (c:ComplianceAction) RETURN count(c)")
        return {
            "status": "success",
            "message": "Regulation uploaded and agentic workflow triggered.",
            "article_count": article_count,
            "facet_count": facet_count,
            "action_count": action_count
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


class HistoryRequest(BaseModel):
    history: List[str]


@app.get("/")
def read_root():
    return {"message": "Backend is running."}


@app.post("/next")
async def next_step(req: HistoryRequest):
    history = req.history
    questions = [
        "Can you describe the main goal of your AI project?",
        "What data will your AI use? Are there any privacy or regulatory concerns?",
        "What are the key risks or ethical considerations for your project?",
        "What kind of test data or augmentations do you expect to need?",
        "What metrics will you use to evaluate your AI system?"
    ]
    if len(history) < len(questions):
        return {"question": questions[len(history)]}
    else:
        summary = "Summary:\n" + \
            '\n'.join([f"Q{i+1}: {ans}" for i, ans in enumerate(history)])
        summary += "\n\nTest Strategy: Based on your answers, the system recommends aligning with GDPR and Responsible AI facets such as fairness and transparency. Suggested augmentations: synthetic data for edge cases. Metrics: accuracy, fairness, robustness."
        return {"summary": summary}
