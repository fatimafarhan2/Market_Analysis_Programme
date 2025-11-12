Market Analysis Pipeline API

How to run

1. Install dependencies (if you don't already have FastAPI/uvicorn):

   python -m pip install fastapi uvicorn

2. Run the server (from repository root):

   python -m uvicorn app.api.server:app --host 0.0.0.0 --port 8000

Endpoints

- POST /api/run
  - Body: { "user_query": "..." }
  - Returns: { "job_id": "...", "status": "queued" }

- GET /api/status/{job_id}
  - Returns job status (queued/running/finished/error)

- GET /api/output/{job_id}
  - Returns the generated JSON output file when ready

- GET /api/report/{job_id}
  - Returns the beautified report (.txt) when ready

- GET /api/charts/{filename}
  - Returns chart PNG files located in the repository's `charts/` folder

Notes

- The server runs the pipeline in a background thread and writes outputs to `output/final_analysis_output.json` and `output/final_analysis_report.txt`.
- Multiple runs will overwrite the same files under `output/`. If you need separate outputs per job, modify the pipeline writer to include job-specific filenames.
