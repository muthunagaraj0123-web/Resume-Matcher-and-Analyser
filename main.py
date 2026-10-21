import io, os, re, json, sqlite3
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DB = Path(__file__).parent / "app.db"
FRONT = Path(__file__).parent.parent / "frontend" / "index.html"

SKILLS = """python java javascript typescript c++ c# sql html css react angular vue node.js express fastapi flask django
spring boot machine learning deep learning nlp computer vision pandas numpy scikit-learn tensorflow pytorch keras
power bi tableau excel data analysis data visualization statistics aws azure gcp docker kubernetes git github ci/cd
linux rest api graphql mongodb postgresql mysql redis java spring agile scrum communication teamwork problem solving
generative ai llm prompt engineering opencv sap salesforce cybersecurity testing selenium""".replace("\n", " ")
SKILL_LIST = sorted({s.strip() for s in re.split(r"\s{1}(?=\S)", SKILLS)}, key=len)
# multi-word skills must be listed explicitly (the split above is only a fallback)
MULTI = ["machine learning", "deep learning", "computer vision", "power bi", "data analysis", "data visualization",
         "spring boot", "problem solving", "prompt engineering", "generative ai", "rest api", "scikit-learn"]
SKILL_LIST = sorted(set(SKILL_LIST) | set(MULTI))

app = FastAPI(title="AI Resume Screening & Job Matching")


def db():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c


def init():
    with db() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS jobs(id INTEGER PRIMARY KEY AUTOINCREMENT,
                     title TEXT, company TEXT, description TEXT, skills TEXT)""")
        if c.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0:
            for t, co, d in [
                ("Technology Analyst", "Deloitte", "Analyze business requirements, build dashboards using SQL, Power BI and Excel. Python scripting, data analysis, communication and teamwork."),
                ("Junior Software Developer", "Acme Tech", "Build REST API services with Python, FastAPI and PostgreSQL. React front end, git, docker, agile practices."),
                ("Machine Learning Intern", "DataWorks", "Train models with Python, pandas, numpy, scikit-learn and TensorFlow. NLP and computer vision projects, statistics."),
            ]:
                c.execute("INSERT INTO jobs(title,company,description,skills) VALUES(?,?,?,?)",
                          (t, co, d, json.dumps(extract_skills(d))))


def extract_skills(text: str):
    t = text.lower()
    return sorted(s for s in SKILL_LIST if re.search(r"(?<![\w+#])" + re.escape(s) + r"(?![\w+#])", t))


def read_resume(f: UploadFile, data: bytes) -> str:
    name = (f.filename or "").lower()
    if name.endswith(".pdf"):
        return "\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(data)).pages)
    if name.endswith(".docx"):
        from docx import Document
        return "\n".join(p.text for p in Document(io.BytesIO(data)).paragraphs)
    return data.decode("utf-8", errors="ignore")


def analyze(text: str):
    sections = {k: bool(re.search(k, text, re.I)) for k in ["education", "experience|internship", "projects", "skills", "certification"]}
    email = re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text)
    tips = []
    if not sections["projects"]: tips.append("Add a Projects section with measurable results.")
    if not sections["certification"]: tips.append("List relevant certifications.")
    if len(text.split()) < 150: tips.append("Resume looks short; add detail on impact.")
    if not email: tips.append("No email address detected.")
    return {"email": email.group(0) if email else None, "words": len(text.split()),
            "sections": sections, "tips": tips}


class JobIn(BaseModel):
    title: str
    company: str = ""
    description: str


@app.get("/api/jobs")
def list_jobs():
    with db() as c:
        return [dict(r, skills=json.loads(r["skills"])) for r in c.execute("SELECT * FROM jobs ORDER BY id DESC")]


@app.post("/api/jobs")
def add_job(j: JobIn):
    with db() as c:
        c.execute("INSERT INTO jobs(title,company,description,skills) VALUES(?,?,?,?)",
                  (j.title, j.company, j.description, json.dumps(extract_skills(j.description))))
    return {"ok": True}


@app.delete("/api/jobs/{jid}")
def del_job(jid: int):
    with db() as c:
        c.execute("DELETE FROM jobs WHERE id=?", (jid,))
    return {"ok": True}


@app.post("/api/screen")
async def screen(resume: UploadFile = File(...), explain: bool = Form(False)):
    text = read_resume(resume, await resume.read()).strip()
    if len(text) < 30:
        raise HTTPException(400, "Could not read text from the resume.")
    jobs = list_jobs()
    if not jobs:
        raise HTTPException(400, "Add at least one job first.")
    rskills = set(extract_skills(text))
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    m = vec.fit_transform([text] + [j["description"] for j in jobs])
    sims = cosine_similarity(m[0], m[1:])[0]
    results = []
    for j, sim in zip(jobs, sims):
        js = set(j["skills"])
        matched, missing = sorted(js & rskills), sorted(js - rskills)
        skill_score = len(matched) / len(js) if js else 0
        score = round(100 * (0.6 * skill_score + 0.4 * min(1, float(sim) * 2.5)))
        results.append({**j, "score": score, "matched": matched, "missing": missing})
    results.sort(key=lambda r: r["score"], reverse=True)
    out = {"skills": sorted(rskills), "analysis": analyze(text), "matches": results, "summary": None}
    if explain and os.getenv("ANTHROPIC_API_KEY") and results:
        try:
            import anthropic
            top = results[0]
            msg = anthropic.Anthropic().messages.create(
                model="claude-sonnet-4-6", max_tokens=400,
                messages=[{"role": "user", "content":
                           f"Resume:\n{text[:4000]}\n\nJob: {top['title']}\n{top['description']}\n\n"
                           f"Missing skills: {top['missing']}. In 4 short sentences, say why this is a fit and what to improve."}])
            out["summary"] = msg.content[0].text
        except Exception as e:
            out["summary"] = f"(AI summary unavailable: {e})"
    return out


@app.get("/")
def index():
    return FileResponse(FRONT)


init()
