"""Resume parsing, feature extraction, ML classification, ATS scoring and matching."""
import datetime
import os
import re

import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "model.joblib")

# group -> {canonical skill: [aliases]}
GROUPS = {
    "Programming": {"python": [], "java": [], "c++": [], "c#": ["c sharp"], "javascript": ["js", "ecmascript"],
                    "typescript": [], "kotlin": [], "swift": [], "php": [], "ruby": [], "golang": ["go lang"],
                    "scala": [], "matlab": [], "bash": ["shell scripting"], "dart": []},
    "Web & Backend": {"html": ["html5"], "css": ["css3"], "react": ["reactjs", "react.js"], "angular": ["angularjs"],
                      "vue.js": ["vue", "vuejs"], "next.js": ["nextjs"], "node.js": ["nodejs", "node js"],
                      "express.js": ["express", "expressjs"], "flask": [], "django": [], "fastapi": [],
                      "spring": [], "spring boot": ["springboot"], "hibernate": [], "microservices": [],
                      "maven": [], "rest api": ["restful", "rest apis", "restful api"], "bootstrap": [],
                      "tailwind css": ["tailwind"]},
    "Data & ML": {"machine learning": ["ml"], "deep learning": [], "nlp": ["natural language processing"],
                  "computer vision": ["opencv"], "pandas": [], "numpy": [], "scikit-learn": ["sklearn", "scikit learn"],
                  "tensorflow": [], "keras": [], "pytorch": [], "xgboost": [], "statistics": ["statistical analysis"],
                  "data analysis": ["data analytics"], "data visualization": ["data visualisation"],
                  "feature engineering": [], "model deployment": ["mlops"], "generative ai": ["llm", "llms"],
                  "spark": ["pyspark", "apache spark"], "hadoop": [], "matplotlib": [], "seaborn": []},
    "BI & Databases": {"power bi": ["powerbi"], "tableau": [], "excel": ["ms excel", "microsoft excel"], "looker": [],
                       "etl": [], "sql": [], "mysql": [], "postgresql": ["postgres"], "mongodb": ["mongo"],
                       "oracle": [], "sqlite": [], "redis": [], "nosql": [], "firebase": []},
    "Cloud & DevOps": {"aws": ["amazon web services"], "azure": [], "gcp": ["google cloud"], "docker": [],
                       "kubernetes": ["k8s"], "terraform": [], "ansible": [], "jenkins": [],
                       "ci/cd": ["cicd", "ci cd"], "git": [], "github": [], "linux": [], "nginx": [],
                       "prometheus": [], "grafana": []},
    "Testing": {"selenium": [], "junit": [], "testng": [], "pytest": [], "postman": [], "manual testing": [],
                "test automation": ["automation testing"], "jira": [], "cypress": [], "api testing": []},
    "Mobile": {"android": [], "ios": [], "flutter": [], "react native": [], "xcode": [], "android studio": []},
    "Security": {"network security": [], "penetration testing": ["pentesting", "pen testing"], "ethical hacking": [],
                 "cryptography": [], "siem": [], "firewall": [], "vulnerability assessment": [], "owasp": [],
                 "wireshark": [], "nmap": [], "kali linux": [], "incident response": []},
    "HR": {"recruitment": ["talent acquisition", "hiring"], "payroll": [], "onboarding": [],
           "employee engagement": [], "performance management": [], "hris": ["hr information system"],
           "talent management": [], "labor law": ["labour law"]},
    "Marketing": {"seo": [], "sem": [], "google analytics": [], "content marketing": [],
                  "social media marketing": ["social media"], "email marketing": [], "google ads": ["adwords"],
                  "copywriting": [], "content writing": [], "market research": [], "brand management": [],
                  "facebook ads": []},
    "Design": {"figma": [], "adobe xd": [], "photoshop": [], "illustrator": [], "wireframing": [],
               "prototyping": [], "user research": [], "ux design": ["ui/ux"], "ui design": []},
    "Professional": {"communication": [], "leadership": [], "teamwork": ["team work"], "problem solving": [],
                     "project management": [], "agile": ["scrum"]},
}

ALIAS_TO_CANON, SKILL_GROUP, _PATTERNS = {}, {}, {}
for _g, _skills in GROUPS.items():
    for _canon, _aliases in _skills.items():
        SKILL_GROUP[_canon] = _g
        _names = [_canon] + _aliases
        for _n in _names:
            ALIAS_TO_CANON[_n] = _canon
        _PATTERNS[_canon] = re.compile(
            r"(?<![\w+#.])(?:%s)(?![\w+#]|\.\w)" % "|".join(re.escape(n) for n in sorted(_names, key=len, reverse=True)),
            re.I)
ALL_SKILLS = sorted(SKILL_GROUP)

LEVELS = [
    ("PhD", re.compile(r"ph\.?\s?d|doctorate", re.I)),
    ("Master's", re.compile(r"m\.?\s?tech|m\.?\s?sc|\bmca\b|\bmba\b|(?-i:M\.E)|master", re.I)),
    ("Bachelor's", re.compile(r"b\.?\s?tech|b\.?\s?sc|\bbca\b|\bbba\b|b\.?\s?com|(?-i:B\.E)|bachelor", re.I)),
    ("Diploma", re.compile(r"diploma", re.I)),
    ("Higher Secondary", re.compile(r"\bhsc\b|\bssc\b|12th|10th", re.I)),
]
EDU_LINE = re.compile("|".join(p.pattern for _, p in LEVELS), re.I)
SECTIONS = {
    "Summary": r"summary|objective|profile|about me",
    "Skills": r"(?:technical |key |core )?skills|technologies",
    "Experience": r"(?:work |professional )?experience|employment|internships?",
    "Education": r"education|academic",
    "Projects": r"projects?",
    "Certifications": r"certifications?|courses|licen[cs]es",
}
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE = re.compile(r"(?:\+?\d{1,3}[\s.-]?)?\(?\d{3,5}\)?[\s.-]?\d{3,5}(?:[\s.-]?\d{3,5})?")
NOT_NAME = ("resume", "curriculum", "vitae", "profile", "objective", "summary", "email", "phone", "contact",
            "address", "skills", "education", "experience", "engineer", "developer", "analyst", "manager")


def extract_text(file_obj) -> str:
    with pdfplumber.open(file_obj) as pdf:
        return "\n".join((p.extract_text() or "") for p in pdf.pages)


def clean_text(t: str) -> str:
    t = re.sub(r"https?://\S+|www\.\S+|\S+@\S+", " ", t.lower())
    t = re.sub(r"[^a-z0-9+#./\s-]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def normalize_skill(s: str) -> str:
    s = s.strip().lower()
    return ALIAS_TO_CANON.get(s, s)


# ------------------------------------------------------------------ extraction
def extract_name(text, email):
    for line in text.splitlines()[:10]:
        line = line.strip()
        words = line.split()
        if 2 <= len(words) <= 4 and re.fullmatch(r"[A-Za-z][A-Za-z.'-]*", "".join(words[:1])) \
                and all(re.fullmatch(r"[A-Za-z.'-]+", w) for w in words) \
                and not any(k in line.lower() for k in NOT_NAME):
            return line.title()
    if email:
        local = re.sub(r"\d+", " ", email.split("@")[0])
        parts = [p for p in re.split(r"[._-]+|\s+", local) if len(p) > 1]
        if parts:
            return " ".join(parts[:3]).title()
    return "Not found"


def extract_skills(text):
    return [s for s in ALL_SKILLS if _PATTERNS[s].search(text)]


def group_skills(skills):
    out = {}
    for s in skills:
        out.setdefault(SKILL_GROUP[s], []).append(s)
    return out


def extract_experience(text):
    stated = [float(x) for x in re.findall(r"(\d{1,2}(?:\.\d)?)\s*\+?\s*(?:years?|yrs?)\b", text, re.I)]
    year_now, total = datetime.datetime.now().year, 0
    for line in text.splitlines():
        if EDU_LINE.search(line):
            continue
        for a, b in re.findall(r"((?:19|20)\d{2})\s*(?:-|–|—|to)\s*((?:19|20)\d{2}|present|current|now)", line, re.I):
            end = year_now if b.lower() in ("present", "current", "now") else int(b)
            if end >= int(a):
                total += end - int(a)
    return round(min(max(stated) if stated else total, 40), 1)


def extract_education(text):
    level = next((n for n, p in LEVELS if p.search(text)), "Not found")
    lines = []
    for ln in text.splitlines():
        ln = ln.strip()
        if EDU_LINE.search(ln) and 4 < len(ln) < 120 and ln not in lines:
            lines.append(ln)
    return level, lines[:4]


def extract_links(text):
    gh = re.search(r"(?:https?://)?(?:www\.)?github\.com/[\w.-]+", text, re.I)
    li = re.search(r"(?:https?://)?(?:[a-z]{2,3}\.)?linkedin\.com/in/[\w%-]+", text, re.I)
    return {"github": gh.group() if gh else None, "linkedin": li.group() if li else None}


def extract_certifications(text):
    out = []
    for ln in text.splitlines():
        if re.search(r"certif|certified", ln, re.I) and 8 < len(ln.strip()) < 110 and ln.strip() not in out:
            out.append(ln.strip())
    return out[:5]


def detect_sections(text):
    return {n: bool(re.search(r"^\s*(?:%s)\b" % p, text, re.I | re.M)) for n, p in SECTIONS.items()}


def ats_score(p, words):
    """Resume-quality score out of 100 with breakdown and tips."""
    b, tips = [], []
    contact = (10 if p["email"] != "Not found" else 0) + (10 if p["phone"] != "Not found" else 0)
    b.append(("Contact details", contact, 20))
    if contact < 20:
        tips.append("Add a clear email address and phone number at the top.")
    online = 10 if (p["links"]["github"] or p["links"]["linkedin"]) else 0
    b.append(("Online presence", online, 10))
    if not online:
        tips.append("Link your LinkedIn or GitHub profile.")
    weights = {"Skills": 12, "Education": 12, "Experience": 12, "Projects": 9}
    sec = 0
    for name, w in weights.items():
        if p["sections"].get(name):
            sec += w
        else:
            tips.append(f"Add a dedicated '{name}' section with a clear heading.")
    b.append(("Key sections", sec, 45))
    depth = round(min(len(p["skills"]) / 10, 1) * 15)
    b.append(("Skills depth", depth, 15))
    if len(p["skills"]) < 8:
        tips.append("List more relevant technical skills (aim for 8 or more).")
    length = 10 if 250 <= words <= 1100 else 5 if 120 <= words < 250 or 1100 < words <= 1600 else 0
    b.append(("Length & readability", length, 10))
    if length < 10:
        tips.append("Keep the resume between roughly 250 and 1,100 words.")
    return sum(x[1] for x in b), b, tips


# ------------------------------------------------------------------ ML
_model = None


def get_model():
    global _model
    if _model is None:
        import joblib
        import train_model
        try:
            _model = joblib.load(MODEL_PATH)
        except Exception:
            _model = train_model.train_and_save()
    return _model


def predict_category(text):
    m = get_model()
    pipe = m["pipeline"]
    probs = pipe.predict_proba([clean_text(text)])[0]
    ranked = sorted(zip(pipe.classes_, probs), key=lambda x: x[1], reverse=True)[:3]
    return [{"name": str(c), "prob": round(float(p) * 100, 1)} for c, p in ranked]


def parse_resume(text):
    email_m = EMAIL.search(text)
    email = email_m.group() if email_m else "Not found"
    phone = next((m.group().strip() for m in PHONE.finditer(text)
                  if 10 <= len(re.sub(r"\D", "", m.group())) <= 13), "Not found")
    skills = extract_skills(text)
    level, edu_lines = extract_education(text)
    words = len(text.split())
    p = {
        "name": extract_name(text, email_m.group() if email_m else None),
        "email": email, "phone": phone,
        "links": extract_links(text),
        "skills": skills, "skill_groups": group_skills(skills),
        "experience": extract_experience(text),
        "education_level": level, "education": edu_lines,
        "certifications": extract_certifications(text),
        "sections": detect_sections(text),
        "word_count": words,
        "categories": predict_category(text),
    }
    p["category"] = p["categories"][0]["name"]
    p["confidence"] = p["categories"][0]["prob"]
    p["ats"], p["ats_breakdown"], p["tips"] = ats_score(p, words)
    return p


# ------------------------------------------------------------------ matching
def similarities(target, docs):
    if not docs:
        return []
    m = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True).fit_transform(
        [clean_text(d) for d in docs] + [clean_text(target)])
    return cosine_similarity(m[-1], m[:-1]).flatten()


def verdict(score):
    return "Strong match" if score >= 70 else "Potential match" if score >= 45 else "Low match"


def score_match(parsed, job, sim):
    """0.55 skill coverage + 0.25 text similarity + 0.20 experience fit."""
    req = [normalize_skill(s) for s in job["skills"].split(",") if s.strip()]
    have = set(parsed["skills"])
    matched = [s for s in req if s in have]
    missing = [s for s in req if s not in have]
    skill = len(matched) / len(req) if req else min(sim / 0.35, 1)
    need = job.get("min_exp") or 0
    exp_fit = min(parsed["experience"] / need, 1) if need else 1
    score = round(100 * (0.55 * skill + 0.25 * min(sim / 0.35, 1) + 0.20 * exp_fit))
    return {"match": score, "matched": matched, "missing": missing, "verdict": verdict(score)}


def job_doc(j):
    return f"{j['title']} {j['skills']} {j['description']}"


def match_jobs(text, parsed, jobs, top_n=6):
    sims = similarities(text, [job_doc(j) for j in jobs])
    out = [{**j, **score_match(parsed, j, float(s))} for j, s in zip(jobs, sims)]
    return sorted(out, key=lambda r: r["match"], reverse=True)[:top_n]


def rank_candidates(job, cands):
    sims = similarities(job_doc(job), [c["text"] for c in cands])
    rows = []
    for c, s in zip(cands, sims):
        p = c["parsed"]
        rows.append({"file": c["file"], "name": p["name"], "email": p["email"], "phone": p["phone"],
                     "experience": p["experience"], "education_level": p["education_level"],
                     "category": p["category"], "ats": p["ats"], "_text": c["text"][:6000], **score_match(p, job, float(s))})
    rows.sort(key=lambda r: r["match"], reverse=True)
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows
