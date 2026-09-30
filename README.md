# 🧠 Machine Learning Based Resume Screening Platform

A web-based **Machine Learning Resume Screening Platform** developed using Python and Flask to automate resume analysis, candidate classification, resume quality assessment, and job matching.

The platform provides two portals:

* **Candidate Portal** – Allows candidates to upload and analyze their resumes, identify suitable job roles, evaluate resume quality, and receive job recommendations.
* **HR Portal** – Allows recruiters to create job openings, upload multiple resumes, automatically screen candidates, rank applicants, and export screening results.

---

## ✨ Features

### 👤 Candidate Portal

* Upload and analyze PDF resumes
* Extract important resume information including:

  * Name
  * Email
  * Phone number
  * LinkedIn
  * GitHub
  * Skills
  * Years of experience
  * Education
  * Certifications
  * Resume sections
  * Word count
* Detect **147+ skills across 12 skill groups**
* Predict suitable job roles using a trained Machine Learning model
* Display **Top 3 predicted roles with probabilities**
* Generate a **Resume Quality Score (0–100)**
* Provide a score breakdown and improvement suggestions
* Recommend suitable job openings
* Show matched skills and missing skill gaps
* Generate optional AI-powered resume feedback

### 🏢 HR Portal

* HR registration and authentication
* Create and manage job openings
* Specify required skills and minimum experience
* Upload and screen multiple resumes
* Bulk screening of up to **20 resumes**
* Automatically rank candidates based on match score
* Categorize candidates as:

  * Strong
  * Potential
  * Low
* Display matched skills and skill gaps
* Generate optional AI-based recruiter notes
* Export screening results to CSV

### 🔐 Security

* Passwords stored using hashing
* Role-based access control
* File upload size restrictions
* Resume files processed in memory
* Resume content treated as untrusted input
* Environment variables used for sensitive configuration

---

## 🧠 Machine Learning Pipeline

The platform combines **Natural Language Processing, Machine Learning, and rule-based information extraction** to analyze resumes.

```text
Resume PDF
    │
    ▼
PDF Text Extraction
(pdfplumber)
    │
    ▼
Text Preprocessing
    │
    ├───────────────┐
    ▼               ▼
Information       TF-IDF
Extraction        Features
    │               │
    │               ▼
    │        Logistic Regression
    │               │
    │               ▼
    │        Job Role Prediction
    │
    ▼
Resume Quality Analysis
    │
    ▼
Job Matching
    │
    ├── Required Skill Coverage
    ├── Text Similarity
    └── Experience Fit
```

### Job Matching Score

The candidate-job matching score is calculated using:

```text
Match Score =
55% Required Skill Coverage
+ 25% Text Similarity
+ 20% Experience Fit
```

For text similarity, the system uses **TF-IDF vectorization and cosine similarity**.

---

## 🤖 Resume Classification Model

The resume classification module uses **TF-IDF (Term Frequency–Inverse Document Frequency)** to convert resume text into numerical features.

The system evaluates multiple classification algorithms:

* Logistic Regression
* Linear Support Vector Machine
* Naive Bayes

The models are compared using **5-fold cross-validation**, and the best-performing model is selected for the final classifier.

The classifier predicts the most relevant job categories and provides probability scores where supported by the selected model.

Duplicate resumes are removed before training to reduce the possibility of train/test data leakage.

---

## 📄 Resume Information Extraction

The platform extracts structured information from unstructured resume text using a combination of:

* Regular expressions
* Keyword matching
* Skill dictionaries
* Alias mapping
* Rule-based extraction

The extracted information includes contact details, skills, education, certifications, experience, resume sections, and other resume characteristics.

The current implementation supports **147+ skills across 12 categories**, which can be extended by modifying the skill groups in `parser_utils.py`.

---

## 📊 Resume Quality Score

Each uploaded resume receives a quality score between **0 and 100**.

The score considers multiple characteristics of the resume, such as:

* Resume sections
* Skills
* Education
* Experience
* Certifications
* Resume length
* Contact information
* Overall completeness

The platform also provides improvement suggestions to help candidates identify missing or weak areas in their resumes.

---

## 💼 Job Recommendations

The candidate portal compares the resume with available job openings.

The matching system considers:

1. Required skill coverage
2. Resume-job text similarity
3. Candidate experience
4. Skill gaps

The result provides candidates with recommended job roles along with the skills they already possess and the skills they may need to improve.

---

## 🤖 Optional Local AI Assistant

The platform optionally integrates with **Ollama** to provide AI-generated feedback using a locally running Large Language Model.

The AI assistant can generate:

* Resume summaries
* Candidate strengths
* Resume improvement suggestions
* Interview questions
* Recruiter notes
* Candidate-job fit explanations

The AI component is optional and runs locally, so resume content does not need to be sent to an external AI API.

### Ollama Setup

Install Ollama and pull the required model:

```bash
ollama pull qwen2.5-coder:7b
```

Start Ollama if it is not already running:

```bash
ollama serve
```

Then start the Flask application:

```bash
python App.py
```

The AI configuration can be changed using environment variables:

```text
OLLAMA_MODEL=qwen2.5-coder:7b
OLLAMA_HOST=http://localhost:11434
```

The AI assistant is used only for advisory feedback. The candidate scores and screening results are generated by the application's Machine Learning models and rule-based logic.

---

## 📚 Dataset

The application supports CSV-based datasets placed inside the `data/` directory.

### Resume Classification Dataset

Required columns:

```text
Category
Resume
```

Example:

```text
Category,Resume
Data Science,"Python SQL Machine Learning..."
Web Development,"HTML CSS JavaScript..."
```

### Job Dataset

Job datasets can contain:

* Job title
* Job description
* Company
* Location
* Required skills

The application automatically detects supported datasets based on their column names.

Refer to `data/README.md` for additional dataset information.

---

## 🚀 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Resume-Screening-Platform.git
cd Resume-Screening-Platform
```

### 2. Create a virtual environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add datasets

Place the required CSV files inside:

```text
data/
```

### 5. Run the application

```bash
python App.py
```

The application will be available at:

```text
http://127.0.0.1:5000
```

Register as either a **Candidate** or **HR** user to access the corresponding portal.

---

## 🔄 Training the Model

The model is trained automatically when the application is started for the first time if a suitable dataset is available.

The trained model is cached as:

```text
model.joblib
```

To manually retrain the model:

```bash
python train_model.py
```

Alternatively, delete the existing `model.joblib` file and restart the application.

---

## 📁 Project Structure

```text
Resume-Screening-Platform/
│
├── App.py                  # Flask application and routes
├── parser_utils.py         # Resume parsing, ML prediction and matching
├── train_model.py          # Model training and evaluation
├── datasets.py             # Dataset loading and detection
├── ai_helper.py            # Optional Ollama integration
├── jobs.py                 # Job management and SQLite storage
│
├── data/
│   └── README.md           # Dataset instructions
│
├── templates/              # HTML templates
├── static/                 # CSS and static assets
│
├── notebooks/              # Research and experimentation notebooks
├── screenshots/            # Application screenshots
│
├── requirements.txt        # Python dependencies
├── LICENSE                 # MIT License
└── README.md
```

---

## ⚠️ Limitations

* Currently supports text-based PDF resumes.
* Scanned/image-based resumes require OCR and are not currently supported.
* Resume name and experience extraction is heuristic and may not work perfectly with unusual resume layouts.
* The skill dictionary is manually maintained and can be extended as required.
* Machine Learning performance depends heavily on the quality and diversity of the training dataset.
* The optional local AI assistant may take longer to generate responses when running a 7B model on CPU.

---

## 🔮 Future Enhancements

Planned improvements include:

* OCR support for scanned resumes
* DOCX resume support
* Transformer-based Named Entity Recognition
* Explainable Machine Learning predictions
* Improved skill and experience extraction
* Advanced candidate ranking
* Resume comparison and analytics dashboard
* Cloud deployment
* More comprehensive job recommendation models

---

## 🛠️ Technologies Used

| Technology          | Purpose                     |
| ------------------- | --------------------------- |
| Python              | Core programming language   |
| Flask               | Web application framework   |
| Scikit-learn        | Machine Learning            |
| TF-IDF              | Text feature extraction     |
| Logistic Regression | Resume classification       |
| Linear SVM          | Model comparison            |
| Naive Bayes         | Model comparison            |
| Pandas              | Data processing             |
| pdfplumber          | PDF text extraction         |
| SQLite              | Database                    |
| HTML/CSS            | Frontend                    |
| Ollama              | Optional local AI assistant |
| Git/GitHub          | Version control             |

---

## 🎯 Project Objective

The primary objective of this project is to develop an automated resume screening system that can reduce the manual effort involved in evaluating resumes and matching candidates with suitable job opportunities.

The platform combines **NLP, Machine Learning, rule-based information extraction, and similarity-based matching** to provide both candidates and recruiters with a unified resume analysis and screening system.

---

## 👨‍💻 Author

**Sarthak Rasal**

Computer Engineering Undergraduate

LinkedIn: [linkedin.com/in/sarthak-rasal-cs](https://www.linkedin.com/in/sarthak-rasal-cs)

---
