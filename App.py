"""Machine Learning Based Resume Screening Platform - Flask app."""
import csv
import io
import os
import secrets
from collections import OrderedDict
from functools import wraps

from flask import (Flask, Response, flash, jsonify, redirect, render_template, request, send_from_directory,
                   session, url_for)
from werkzeug.security import check_password_hash, generate_password_hash

import ai_helper
import jobs as jobstore
import parser_utils as pu

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=BASE, static_folder=None)  # flat layout: HTML sits next to App.py
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY") or secrets.token_hex(16)
app.config["MAX_CONTENT_LENGTH"] = 40 * 1024 * 1024
DB = os.path.join(BASE, "screening.db")
jobstore.init_db(DB)
METRICS = pu.get_model()["metrics"]  # trains on first run if model.joblib is missing
RESULTS = OrderedDict()  # in-memory store for the latest analyses (local use)


def stash(obj):
    rid = secrets.token_hex(8)
    RESULTS[rid] = obj
    while len(RESULTS) > 200:
        RESULTS.popitem(last=False)
    return rid


def login_required(role=None):
    def deco(fn):
        @wraps(fn)
        def wrapper(*a, **kw):
            if "user_id" not in session:
                return redirect(url_for("login"))
            if role and session.get("role") != role:
                flash(f"That page is for {role.upper() if role == 'hr' else role} accounts.", "error")
                return redirect(url_for("index"))
            return fn(*a, **kw)
        return wrapper
    return deco


@app.route("/")
def index():
    return render_template("index.html", m=METRICS, n_skills=len(pu.ALL_SKILLS))


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name, email = request.form["name"].strip(), request.form["email"].strip().lower()
        password = request.form["password"]
        role = "hr" if request.form.get("role") == "hr" else "candidate"
        if not name or not email or len(password) < 6:
            flash("Fill all fields; password needs at least 6 characters.", "error")
        else:
            try:
                with jobstore.get_db(DB) as db:
                    db.execute("INSERT INTO users (name,email,password,role) VALUES (?,?,?,?)",
                               (name, email, generate_password_hash(password), role))
                flash("Account created. Please log in.", "ok")
                return redirect(url_for("login"))
            except Exception:
                flash("That email is already registered.", "error")
    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        with jobstore.get_db(DB) as db:
            user = db.execute("SELECT * FROM users WHERE email=?",
                              (request.form["email"].strip().lower(),)).fetchone()
        if user and check_password_hash(user["password"], request.form["password"]):
            session.clear()
            session.update(user_id=user["id"], name=user["name"], role=user["role"])
            return redirect(url_for("upload" if user["role"] == "candidate" else "screen"))
        flash("Invalid email or password.", "error")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


def read_pdf(f):
    if not f or not f.filename.lower().endswith(".pdf"):
        return None, "Please upload a PDF file."
    try:
        text = pu.extract_text(f)
    except Exception:
        return None, "Could not read that PDF."
    if not text.strip():
        return None, "No text found. Scanned PDFs are not supported yet."
    return text, None


@app.route("/upload", methods=["GET", "POST"])
@login_required("candidate")
def upload():
    if request.method == "POST":
        text, err = read_pdf(request.files.get("resume"))
        if err:
            flash(err, "error")
            return render_template("candidate.html")
        data = pu.parse_resume(text)
        data["_text"] = text[:6000]  # kept in memory only, for the optional AI feedback
        data["recs"] = pu.match_jobs(text, data, jobstore.all_jobs(DB))
        session["rid"] = stash(data)
        return redirect(url_for("result"))
    return render_template("candidate.html")


def current_result():
    return RESULTS.get(session.get("rid"))


@app.route("/result")
@login_required("candidate")
def result():
    data = current_result()
    return render_template("result.html", d=data) if data else redirect(url_for("upload"))


@app.route("/recommendations")
@login_required("candidate")
def recommendations():
    data = current_result()
    return render_template("recommendations.html", d=data) if data else redirect(url_for("upload"))


@app.route("/hr/post-job", methods=["GET", "POST"])
@login_required("hr")
def post_job():
    if request.method == "POST":
        f = request.form
        title, company = f.get("title", "").strip(), f.get("company", "").strip()
        if title and company:
            skills = ", ".join(pu.normalize_skill(s) for s in f.get("skills", "").split(",") if s.strip())
            try:
                min_exp = max(0, min(int(f.get("min_exp") or 0), 40))
            except ValueError:
                min_exp = 0
            with jobstore.get_db(DB) as db:
                db.execute("INSERT INTO jobs (title,company,location,skills,min_exp,description) VALUES (?,?,?,?,?,?)",
                           (title, company, f.get("location", "").strip(), skills, min_exp,
                            f.get("description", "").strip()))
            flash("Job posted.", "ok")
            return redirect(url_for("post_job"))
        flash("Title and company are required.", "error")
    return render_template("hr_post_job.html", jobs=jobstore.all_jobs(DB))


@app.route("/hr/screen", methods=["GET", "POST"])
@login_required("hr")
def screen():
    jobs = jobstore.all_jobs(DB)
    if request.method == "POST":
        job = next((j for j in jobs if str(j["id"]) == request.form.get("job_id")), None)
        files = [f for f in request.files.getlist("resumes") if f.filename][:20]
        if not job or not files:
            flash("Choose a job and upload at least one PDF resume.", "error")
            return render_template("hr_screening.html", jobs=jobs)
        cands, failed = [], []
        for f in files:
            text, err = read_pdf(f)
            if err:
                failed.append(f.filename)
            else:
                cands.append({"file": f.filename, "text": text, "parsed": pu.parse_resume(text)})
        if failed:
            flash("Skipped (unreadable or not PDF): " + ", ".join(failed), "error")
        rows = pu.rank_candidates(job, cands) if cands else []
        rid = stash({"job": job, "rows": rows})
        return render_template("hr_screening.html", jobs=jobs, job=job, rows=rows, rid=rid)
    return render_template("hr_screening.html", jobs=jobs)


@app.route("/hr/screen/<rid>.csv")
@login_required("hr")
def export(rid):
    data = RESULTS.get(rid)
    if not data:
        return redirect(url_for("screen"))
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["Rank", "Name", "Email", "Phone", "Experience (yrs)", "Education", "Predicted role",
                "Match %", "Verdict", "Matched skills", "Missing skills", "Resume quality"])
    for r in data["rows"]:
        w.writerow([r["rank"], r["name"], r["email"], r["phone"], r["experience"], r["education_level"],
                    r["category"], r["match"], r["verdict"], "; ".join(r["matched"]), "; ".join(r["missing"]),
                    r["ats"]])
    return Response(out.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=screening_{data['job']['title']}.csv"})


@app.route("/api/ai/status")
def ai_status():
    return jsonify(ai_helper.status())


@app.route("/api/ai/feedback", methods=["POST"])
@login_required("candidate")
def ai_feedback():
    d = current_result()
    if not d:
        return jsonify(error="Analyze a resume first."), 400
    try:
        return jsonify(ai_helper.resume_feedback(d, d["_text"]))
    except ai_helper.AIError as e:
        return jsonify(error=str(e)), 503


@app.route("/api/ai/explain/<rid>/<int:idx>", methods=["POST"])
@login_required("hr")
def ai_explain(rid, idx):
    data = RESULTS.get(rid)
    if not data or not 0 <= idx < len(data.get("rows", [])):
        return jsonify(error="Result expired. Run the screening again."), 400
    row = data["rows"][idx]
    try:
        return jsonify(ai_helper.explain_match(data["job"], row, row["_text"]))
    except ai_helper.AIError as e:
        return jsonify(error=str(e)), 503


@app.route("/about")
def about():
    return render_template("about.html", m=METRICS, n_skills=len(pu.ALL_SKILLS), groups=pu.GROUPS)


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/style.css")
def style():
    return send_from_directory(BASE, "style.css")


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
