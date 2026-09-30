# 🧠 Machine Learning Based Resume Screening Platform

A web platform that parses PDF resumes, classifies them into job roles with a trained ML model, scores resume quality, and ranks candidates against job openings. It has two portals: **candidates** analyze their own resume, and **HR** screens many resumes at once.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![Flask](https://img.shields.io/badge/Flask-3.x-black) ![scikit-learn](https://img.shields.io/badge/scikit--learn-TF--IDF%20%2B%20LogReg-orange) ![License](https://img.shields.io/badge/License-MIT-green)

## ✨ Features

**Candidate portal**
- 📄 Extracts **14+ parameters**: name, email, phone, LinkedIn, GitHub, skills (147 skills in 12 groups), years of experience, education level, education lines, certifications, detected sections, word count, role probabilities and quality score
- 🤖 **ML role prediction:** top 3 roles from 10 categories with probabilities
- ✅ **Resume quality score (0-100)** with breakdown and improvement tips
- 💼 **Job recommendations** with matched skills and skill gaps

**HR portal**
- 🏢 Post jobs with required skills and minimum experience
- 📂 **Bulk screening:** upload up to 20 resumes, get a ranked shortlist with verdicts (Strong / Potential / Low)
- ⬇ Export results to CSV

**Security:** hashed passwords, role-based access, upload size cap, resumes processed in memory only.

## 🧠 How it works

```
PDF ─► text (pdfplumber) ─► regex + alias dictionary ─► structured fields
                          ├─► TF-IDF (1-2 grams) + Logistic Regression ─► role probabilities
                          └─► TF-IDF cosine similarity ─► job match
Match score = 55% required-skill coverage + 25% text similarity + 20% experience fit
```

## 📚 Datasets

Place CSV files in the `data/` folder. They are detected automatically by column names (see `data/README.md`):

| Purpose | Columns | Example |
|---|---|---|
| Resume categorization | `Category`, `Resume` | `UpdatedResumeDataSet.csv` |
| Job recommendations | job title + description (optional company, location, skills) | `jobs.csv` |

- On startup the app trains on the resume dataset if present. It removes duplicate resumes (to avoid train/test leakage), compares **Logistic Regression, Linear SVM and Naive Bayes** with 5-fold cross-validation, and keeps the best one.
- Results appear on the **About** page. Without a dataset the app falls back to generated bootstrap data, whose near-perfect scores **overstate real-world accuracy**.
- Retrain any time: `python train_model.py` (or delete `model.joblib`). To re-import jobs, delete `screening.db`.

## 🤖 Local AI assistant (Ollama)

Optional. The platform can call a local LLM through [Ollama](https://ollama.com) for candidate feedback (summary, strengths, fixes, interview questions) and recruiter notes on why a candidate fits. Nothing leaves your machine.

```bash
ollama pull qwen2.5-coder:7b
ollama serve                      # skip if Ollama is already running
python App.py
```

Use the **Generate feedback** button on the results page or **AI notes** on the HR screening table. Configure with environment variables `OLLAMA_MODEL` (default `qwen2.5-coder:7b`) and `OLLAMA_HOST` (default `http://localhost:11434`).

The AI is advisory only: scores always come from the ML model and rules. Resume text is treated as untrusted input. A 7B model on CPU can take a minute per request.

## 🚀 Run locally

```bash
git clone https://github.com/snehal447/Resume-Parser-.git
cd Resume-Parser-
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python App.py
```

Open **http://127.0.0.1:5000**. The model trains automatically on first run and is cached as `model.joblib`. Sign up as a *Candidate* to analyze a resume, or as *HR* to screen candidates. Set `SECRET_KEY` in your environment to keep sessions across restarts.

## 📁 Structure

```
App.py               Flask routes, auth, role protection
parser_utils.py      Extraction, ML prediction, quality score, matching
train_model.py       Train, compare and save the classifier
datasets.py          Auto-detect and load CSVs from data/
ai_helper.py         Optional Ollama (Qwen) integration
data/                Put your datasets here
jobs.py              SQLite storage and seed jobs
*.html, style.css    Interface (flat layout, base.html + macros.html shared)
*.ipynb, *.png       Original research notebooks and screenshots
```

## ⚠️ Limitations

- Text-based PDFs only (no OCR for scanned resumes)
- Name and experience extraction are heuristic and can miss unusual layouts
- Skill list lives in `parser_utils.py`; extend `GROUPS` for new domains

## 🔮 Roadmap

OCR and DOCX support · transformer-based NER · explainable predictions · deployment

## 👥 Authors

- **Snehal**: [@snehal447](https://github.com/snehal447)
- **Your friend's name**: [@their-username](https://github.com/their-username)

## 📄 License

MIT. See [LICENSE](LICENSE).
