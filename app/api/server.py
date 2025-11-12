from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any
import uuid
import traceback

# Import the pipeline runner
from app.test import run_data_fetching

app = FastAPI(title="Market Analysis Pipeline API")

# Allow CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RunRequest(BaseModel):
    user_query: str = ""


# Job storage
JOBS: Dict[str, Dict[str, Any]] = {}
executor = ThreadPoolExecutor(max_workers=2)


def _workspace_root() -> Path:
    # app/api/server.py -> parents[2] == repository root
    return Path(__file__).resolve().parents[2]


def _charts_dir() -> Path:
    return _workspace_root() / "charts"


def _output_dir() -> Path:
    return _workspace_root() / "output"


def _run_job(job_id: str, user_query: str):
    try:
        JOBS[job_id]["status"] = "running"
        result = run_data_fetching(user_query=user_query)
        # store result and paths
        JOBS[job_id]["status"] = "finished"
        JOBS[job_id]["result"] = result
        # try to attach paths to saved files
        out_dir = _output_dir()
        JOBS[job_id]["json_path"] = str(out_dir / "final_analysis_output.json")
        JOBS[job_id]["report_path"] = str(out_dir / "final_analysis_report.txt")
    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(e)
        JOBS[job_id]["traceback"] = traceback.format_exc()


@app.post("/api/run")
def start_pipeline(req: RunRequest):
    """Start the pipeline in background. Returns a job_id to poll status/result."""
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "queued", "result": None}
    executor.submit(_run_job, job_id, req.user_query)
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/status/{job_id}")
def job_status(job_id: str):
    info = JOBS.get(job_id)
    if not info:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, "status": info.get("status"), "error": info.get("error")}


@app.get("/api/output/{job_id}")
def get_output_json(job_id: str):
    info = JOBS.get(job_id)
    if not info:
        raise HTTPException(status_code=404, detail="Job not found")
    if info.get("status") != "finished":
        return JSONResponse(status_code=202, content={"status": info.get("status"), "detail": "Not ready"})
    json_path = info.get("json_path")
    if not json_path or not Path(json_path).exists():
        raise HTTPException(status_code=404, detail="Output JSON not found")
    return FileResponse(json_path, media_type="application/json", filename=Path(json_path).name)


@app.get("/api/report/{job_id}")
def get_report_txt(job_id: str):
    info = JOBS.get(job_id)
    if not info:
        raise HTTPException(status_code=404, detail="Job not found")
    if info.get("status") != "finished":
        return JSONResponse(status_code=202, content={"status": info.get("status"), "detail": "Not ready"})
    report_path = info.get("report_path")
    if not report_path or not Path(report_path).exists():
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(report_path, media_type="text/plain", filename=Path(report_path).name)


@app.get("/api/charts/{filename}")
def serve_chart(filename: str):
    charts_dir = _charts_dir()
    file_path = charts_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Chart not found")
    return FileResponse(str(file_path), media_type="image/png", filename=file_path.name)


@app.get("/api/jobs")
def list_jobs():
    return {k: {"status": v.get("status")} for k, v in JOBS.items()}
