# AI Resume Screening & Job Matching
Stack: FastAPI + SQLite + scikit-learn (TF-IDF) backend, React frontend (served by FastAPI).

## Run
    cd backend
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=...   # optional, enables Claude summary
    uvicorn main:app --reload
Open http://localhost:8000

## How scoring works
score = 60% skill overlap (resume vs job skills) + 40% TF-IDF cosine similarity of resume vs job description.
Upload PDF, DOCX or TXT. Add/delete jobs in the UI; skills are auto-extracted from descriptions.
