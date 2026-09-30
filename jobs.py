"""Job storage (SQLite) and seed data."""
import sqlite3

SEED_JOBS = [
    ("Data Scientist", "Acme Analytics", "Pune", "python, machine learning, pandas, scikit-learn, statistics", 2,
     "Build predictive models, run experiments and communicate insights to product teams."),
    ("Machine Learning Engineer", "NeuralWorks", "Bengaluru", "python, deep learning, pytorch, model deployment, docker", 3,
     "Train and deploy deep learning models into production services."),
    ("Full Stack Developer", "WebCraft", "Remote", "javascript, react, node.js, sql, rest api", 2,
     "Develop responsive web apps with React front ends and Node.js back ends."),
    ("Data Analyst", "InsightWorks", "Mumbai", "sql, excel, power bi, tableau, data analysis", 1,
     "Turn business data into dashboards and reports for stakeholders."),
    ("DevOps Engineer", "CloudNine", "Bengaluru", "docker, kubernetes, aws, linux, ci/cd, terraform", 3,
     "Automate deployments and manage cloud infrastructure at scale."),
    ("Java Backend Developer", "FinServe", "Hyderabad", "java, spring boot, mysql, rest api, microservices", 2,
     "Design and maintain Java microservices for payment platforms."),
    ("QA Automation Engineer", "TestRight", "Pune", "selenium, java, test automation, api testing, jira", 2,
     "Build automated regression suites and own release quality."),
    ("Android Developer", "AppNest", "Remote", "android, kotlin, rest api, firebase, git", 1,
     "Build and ship consumer Android apps."),
    ("Security Analyst", "SafeNet", "Gurugram", "network security, siem, incident response, vulnerability assessment, linux", 2,
     "Monitor threats, investigate incidents and harden systems."),
    ("HR Executive", "PeopleFirst", "Pune", "recruitment, onboarding, payroll, communication, employee engagement", 1,
     "Run end to end recruitment, onboarding and payroll coordination."),
    ("Digital Marketing Specialist", "GrowthLab", "Remote", "seo, google ads, google analytics, content marketing, email marketing", 2,
     "Plan campaigns, improve search ranking and report on conversions."),
]


def get_db(path):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(path):
    with get_db(path) as db:
        db.execute("""CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY, name TEXT, email TEXT UNIQUE,
            password TEXT, role TEXT DEFAULT 'candidate')""")
        db.execute("""CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY, title TEXT, company TEXT, location TEXT,
            skills TEXT, min_exp INTEGER DEFAULT 0, description TEXT)""")
        if db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0:
            rows = SEED_JOBS
            try:  # prefer a jobs dataset from data/ when one is present
                from datasets import find_jobs_csv, load_jobs
                p = find_jobs_csv()
                rows = load_jobs(p) if p else SEED_JOBS
            except Exception as e:
                print("Jobs dataset not loaded, using sample jobs:", e)
            db.executemany("INSERT INTO jobs (title,company,location,skills,min_exp,description) VALUES (?,?,?,?,?,?)",
                           rows)


def all_jobs(path):
    with get_db(path) as db:
        return [dict(r) for r in db.execute("SELECT * FROM jobs ORDER BY id DESC")]
