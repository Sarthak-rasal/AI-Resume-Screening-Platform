"""Auto-detect and load datasets placed in the data/ folder.

Resume dataset : CSV with 'Category' and 'Resume' columns (e.g. UpdatedResumeDataSet.csv)
Jobs dataset   : CSV with a job title column and a description column
                 (title / job title / position ...  and  description / job description ...)
"""
import glob
import os

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _norm(c):
    return str(c).strip().lower().replace("_", " ")


def _read(path, **kw):
    try:
        return pd.read_csv(path, **kw)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", **kw)


def _pick(cols, options):
    for o in options:
        for c in cols:
            if _norm(c) == o:
                return c
    return None


def csv_files():
    return sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))


def find_resume_csv():
    for p in csv_files():
        cols = list(_read(p, nrows=1).columns)
        if _pick(cols, ["category"]) and _pick(cols, ["resume", "resume str", "resume text"]):
            return p
    return None


def find_jobs_csv():
    for p in csv_files():
        cols = list(_read(p, nrows=1).columns)
        if _pick(cols, ["category"]) and _pick(cols, ["resume", "resume str", "resume text"]):
            continue
        if _pick(cols, ["title", "job title", "position", "job position", "role"]) and \
                _pick(cols, ["description", "job description", "details", "responsibilities"]):
            return p
    return None


def load_resumes(path, min_per_class=10):
    df = _read(path)
    cat = _pick(df.columns, ["category"])
    res = _pick(df.columns, ["resume", "resume str", "resume text"])
    df = df[[cat, res]].dropna().rename(columns={cat: "Category", res: "Resume"})
    n_raw = len(df)
    df = df.drop_duplicates(subset="Resume")  # duplicates would leak between train and test
    counts = df["Category"].value_counts()
    df = df[df["Category"].isin(counts[counts >= min_per_class].index)]
    src = (f"{os.path.basename(path)} ({len(df)} unique resumes from {n_raw} rows, "
           f"{df['Category'].nunique()} categories)")
    return df["Resume"].tolist(), df["Category"].tolist(), src


def load_jobs(path, limit=300):
    """Return job tuples (title, company, location, skills_text, min_exp, description)."""
    from parser_utils import extract_experience, extract_skills
    df = _read(path)
    cols = df.columns
    t = _pick(cols, ["title", "job title", "position", "job position", "role"])
    d = _pick(cols, ["description", "job description", "details", "responsibilities"])
    c = _pick(cols, ["company", "company name", "employer"])
    l = _pick(cols, ["location", "city", "job location"])
    s = _pick(cols, ["skills", "skill", "key skills", "required skills", "skills required"])
    df = df.dropna(subset=[t, d])
    df = df.sample(n=min(limit, len(df)), random_state=42) if len(df) > limit else df
    out = []
    for _, r in df.iterrows():
        desc = str(r[d])[:1500]
        skills = ", ".join(x.strip() for x in str(r[s]).replace("|", ",").replace(";", ",").split(",")
                           if x.strip()) if s and pd.notna(r[s]) else ", ".join(extract_skills(desc))
        out.append((str(r[t])[:120], str(r[c])[:80] if c and pd.notna(r[c]) else "", str(r[l])[:80]
                    if l and pd.notna(r[l]) else "", skills, int(min(extract_experience(desc), 15)), desc))
    return out
