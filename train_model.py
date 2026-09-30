"""Train the resume category classifier.

Usage:
    python train_model.py                      # uses a resume CSV from data/ if present
    python train_model.py --csv resumes.csv    # explicit file: columns 'Category' and 'Resume'
"""
import argparse
import random

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from datasets import find_resume_csv, load_resumes
from parser_utils import MODEL_PATH, clean_text

SKILLS = {
    "Data Science": "python,machine learning,deep learning,nlp,pandas,numpy,scikit-learn,tensorflow,keras,pytorch,statistics,feature engineering,xgboost,computer vision,matplotlib,seaborn,spark,generative ai,model deployment,data visualization",
    "Web Development": "html,css,javascript,typescript,react,angular,vue.js,node.js,express.js,flask,django,fastapi,rest api,bootstrap,tailwind css,next.js,mongodb,git",
    "Data Analytics": "sql,excel,power bi,tableau,data analysis,data visualization,statistics,python,etl,looker,mysql,google analytics,pandas",
    "DevOps Engineer": "docker,kubernetes,aws,azure,gcp,terraform,ansible,jenkins,ci/cd,linux,git,nginx,prometheus,grafana,bash",
    "Java Developer": "java,spring,spring boot,mysql,postgresql,rest api,junit,hibernate,microservices,maven,git",
    "Software Testing": "selenium,junit,testng,pytest,postman,manual testing,test automation,jira,cypress,api testing,java,python,sql",
    "Mobile App Development": "android,ios,flutter,react native,kotlin,swift,dart,firebase,xcode,android studio,rest api,java",
    "Cybersecurity": "network security,penetration testing,ethical hacking,cryptography,siem,firewall,vulnerability assessment,owasp,wireshark,nmap,kali linux,incident response,linux,python",
    "Human Resources": "recruitment,payroll,onboarding,employee engagement,performance management,hris,talent management,labor law,communication,leadership,excel",
    "Digital Marketing": "seo,sem,google analytics,content marketing,social media marketing,email marketing,google ads,copywriting,content writing,market research,brand management,facebook ads",
}
PHRASES = {
    "Data Science": ["built and deployed predictive models", "performed feature engineering and hyperparameter tuning", "trained classification and regression models", "analyzed large datasets to extract insights"],
    "Web Development": ["developed responsive web applications", "built REST APIs and connected the front end to back end", "optimized page load performance", "implemented authentication and user dashboards"],
    "Data Analytics": ["created interactive dashboards and reports", "wrote complex queries to analyze business data", "presented insights to stakeholders", "automated recurring reporting"],
    "DevOps Engineer": ["automated deployment pipelines", "managed containerized infrastructure in the cloud", "monitored system uptime and reliability", "wrote infrastructure as code"],
    "Java Developer": ["developed enterprise microservices", "built backend services with dependency injection", "wrote unit tests and handled database persistence", "tuned JVM performance"],
    "Software Testing": ["designed and executed test cases", "reported and tracked defects", "automated regression test suites", "performed functional and API testing"],
    "Mobile App Development": ["published apps on app stores", "built cross platform mobile applications", "integrated push notifications and APIs", "optimized app performance on devices"],
    "Cybersecurity": ["conducted vulnerability assessments", "monitored security events and responded to incidents", "hardened systems and networks", "performed penetration tests and wrote reports"],
    "Human Resources": ["managed end to end recruitment", "coordinated employee onboarding and engagement", "handled payroll and compliance", "ran performance review cycles"],
    "Digital Marketing": ["planned and ran paid ad campaigns", "improved organic search ranking", "grew social media audience", "tracked campaign analytics and conversions"],
}
POOLS = {c: s.split(",") for c, s in SKILLS.items()}


def make_resume(cat, rng):
    pool = POOLS[cat]
    skills = rng.sample(pool, rng.randint(3, min(8, len(pool))))
    if rng.random() < 0.7:  # noise: skills from other domains
        others = [s for c, p in POOLS.items() if c != cat for s in p]
        skills += rng.sample(others, rng.randint(1, 5))
    rng.shuffle(skills)
    phrases = rng.sample(PHRASES[cat], rng.randint(1, 3))
    if rng.random() < 0.45:
        phrases.append(rng.choice(PHRASES[rng.choice([c for c in PHRASES if c != cat])]))
    return (f"Summary\nProfessional with {rng.randint(0, 12)} years of experience.\nSkills\n{', '.join(skills)}\n"
            f"Experience\n{'. '.join(phrases)} using {', '.join(rng.sample(skills, min(3, len(skills))))}.")


def synthetic(n_per_class=250, seed=42):
    rng = random.Random(seed)
    X, y = [], []
    for cat in POOLS:
        for _ in range(n_per_class):
            X.append(make_resume(cat, rng))
            y.append(cat)
    return X, y


CANDIDATES = {
    "Logistic Regression": lambda: LogisticRegression(C=10, max_iter=2000),
    "Linear SVM (calibrated)": lambda: CalibratedClassifierCV(LinearSVC(C=1), cv=3),
    "Multinomial Naive Bayes": lambda: MultinomialNB(alpha=0.1),
}


def build_pipeline(name="Logistic Regression"):
    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True, min_df=2)),
        ("clf", CANDIDATES[name]()),
    ])


def train_and_save(csv_path=None):
    path = csv_path or find_resume_csv()
    if path:
        X, y, source = load_resumes(path)
        real = True
    else:
        X, y = synthetic()
        source, real = f"generated bootstrap data ({len(X)} resumes, {len(set(y))} roles)", False
    X = [clean_text(t) for t in X]
    comparison = {}
    for name in CANDIDATES:  # model selection by 5-fold cross-validation
        cv = cross_val_score(build_pipeline(name), X, y, cv=5)
        comparison[name] = [round(float(np.mean(cv)) * 100, 1), round(float(np.std(cv)) * 100, 1)]
    best = max(comparison, key=lambda n: comparison[n][0])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pred = build_pipeline(best).fit(Xtr, ytr).predict(Xte)
    metrics = {"accuracy": round(accuracy_score(yte, pred) * 100, 1),
               "macro_f1": round(f1_score(yte, pred, average="macro") * 100, 1),
               "cv_mean": comparison[best][0], "cv_std": comparison[best][1],
               "classes": len(set(y)), "samples": len(X), "source": source, "real_data": real,
               "algorithm": f"TF-IDF (1-2 grams) + {best}", "comparison": comparison, "best": best}
    model = {"pipeline": build_pipeline(best).fit(X, y), "metrics": metrics}
    joblib.dump(model, MODEL_PATH)
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="CSV with 'Category' and 'Resume' columns")
    m = train_and_save(ap.parse_args().csv)["metrics"]
    print("Saved model.joblib")
    for k, v in m.items():
        print(f"  {k}: {v}")
