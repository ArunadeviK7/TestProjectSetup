# Backend API

## Quick Start

1. Activate the Conda environment:
   ```
   conda activate ./backend/.conda
   ```
2. Install dependencies (if not already):
   ```
   pip install -r requirements.txt
   ```
3. Run the FastAPI server:
   ```
   uvicorn main:app --reload
   ```

## Endpoints
- `GET /` — Health check (returns backend running message)

Extend `main.py` to add LangGraph agentic AI endpoints as needed.
