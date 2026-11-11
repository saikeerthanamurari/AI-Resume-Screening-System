# AI-Based Resume Screening & Candidate Analysis System

## 1. Project Overview

The AI-Based Resume Screening & Candidate Analysis System is a Python-based web application developed using Flask, Machine Learning, and Natural Language Processing (NLP).

The system analyzes resumes against a given job description, extracts relevant skills, predicts resume categories, calculates skill-matching scores, and compares candidates based on textual similarity.

It supports both single-resume analysis and multiple-resume analysis.

## 2. Project Objectives

- Automate preliminary resume screening.
- Extract relevant information and skills from resumes.
- Compare candidate skills with job requirements.
- Predict resume categories using Machine Learning.
- Calculate skill-matching percentages.
- Measure textual similarity using TF-IDF and cosine similarity.
- Generate candidate analysis reports in CSV format.

## 3. Features

- Single-resume upload and analysis.
- Multiple-resume upload and comparison.
- Support for PDF, DOCX, and TXT files.
- Resume category prediction using a trained ML classifier.
- Automatic skill extraction.
- Matched, missing, and extra skill identification.
- Skill-matching score calculation.
- TF-IDF similarity calculation.
- Candidate ranking based on similarity.
- Downloadable CSV reports.
- Machine Learning model comparison.
- Hyperparameter tuning and evaluation.

## 4. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Flask | Web application framework |
| HTML | Web page structure |
| CSS | Web page styling |
| JavaScript | Frontend interactions |
| Scikit-learn | Machine Learning |
| TF-IDF | Text feature extraction |
| Cosine Similarity | Resume and job-description similarity |
| Pandas | Data processing |
| NumPy | Numerical operations |
| PyMuPDF | PDF text extraction |
| python-docx | DOCX text extraction |
| Joblib | Saving and loading trained models |

## 5. Project Structure

```text
Resume_Screening_Python_Submission/
│
├── app.py
├── resume_parser.py
├── preprocessing.py
├── candidate_scoring.py
├── train_model.py
├── train_classifier.py
├── compare_models.py
├── tune_model.py
├── requirements.txt
├── README.md
├── Resume_Screening_Report.ipynb
│
├── models/
│   ├── resume_classifier.joblib
│   └── resume_classifier_tuned.joblib
│
├── templates/
│   ├── index.html
│   └── results.html
│
├── static/
│   └── css/
│       └── style.css
│
├── sample_data/
│   ├── Candidate_A.txt
│   └── Candidate_B.txt
│
└── outputs/
    ├── classification_report.txt
    ├── confusion_matrix.csv
    ├── hyperparameter_tuning_results.csv
    ├── ml_metrics.json
    ├── model_comparison.csv
    ├── test_predictions.csv
    ├── tuned_classification_report.txt
    └── tuned_ml_metrics.json
```

**Note:** Uploaded resumes, generated candidate reports, and private screenshots are excluded from the public GitHub repository.

## 6. Installation and Setup

### Step 1: Create a Virtual Environment

```bash
python3 -m venv .venv
```

### Step 2: Activate the Virtual Environment

```bash
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Application

The application can be started on port 5001 using:

```bash
python -c "import app; app.app.run(host='127.0.0.1', port=5001, debug=False)"
```

### Step 5: Open the Application

Open your browser and visit:

http://127.0.0.1:5001

## 7. Supported Resume Formats

The application supports:

- PDF
- DOCX
- TXT

Maximum upload size: 10 MB per request.

## 8. Working Methodology

### Step 1: Resume Upload

The user can upload either one resume or multiple resumes through the application.

### Step 2: Job Description Input

The user provides the job description containing the required skills and qualifications.

### Step 3: Resume Text Extraction

The system extracts text from uploaded PDF, DOCX, or TXT files.

### Step 4: Text Preprocessing

The extracted text is processed to identify relevant information and skills.

### Step 5: Machine Learning Classification

The trained Machine Learning classifier predicts the resume category.

### Step 6: Candidate Scoring

The system compares extracted candidate skills with the skills required in the job description.

### Step 7: Similarity Analysis

TF-IDF and cosine similarity are used to calculate textual similarity between the resume and job description.

### Step 8: Result Generation

The application displays candidate analysis results and provides downloadable CSV reports.

## 9. Candidate Evaluation Metrics

The application uses three separate evaluation measures.

### A. Predicted Resume Category

The trained Machine Learning classifier predicts the category of the uploaded resume.

### B. Skill Matching Score

The skill-matching score represents the percentage of required job-description skills identified in the resume.

**Formula:**

Skill Matching Score = (Total Matched Skills / Total Required Skills) × 100

### C. TF-IDF Similarity Score

TF-IDF converts text into numerical features. Cosine similarity measures the similarity between the resume and job description.

These three measures serve different purposes and should not be treated as interchangeable.

## 10. Machine Learning Implementation

The project uses Scikit-learn to train and evaluate resume classification models.

The implementation includes:

- Resume text preprocessing.
- TF-IDF feature extraction.
- Machine Learning classification.
- Model comparison.
- Cross-validation.
- Hyperparameter tuning.
- Classification reports.
- Confusion matrix analysis.
- Test-set prediction and evaluation.

The tuned classifier is saved in:

`models/resume_classifier_tuned.joblib`

The Flask application loads this tuned model for resume category prediction.

## 11. Output Files

The `outputs/` directory contains the machine learning evaluation results.

### Machine Learning Outputs

- `classification_report.txt`
- `confusion_matrix.csv`
- `ml_metrics.json`
- `model_comparison.csv`
- `hyperparameter_tuning_results.csv`
- `test_predictions.csv`
- `tuned_classification_report.txt`
- `tuned_ml_metrics.json`

### Application Outputs

The application generates single-candidate and multiple-candidate analysis reports in CSV format. These reports are stored locally and excluded from the public GitHub repository to protect candidate information.

## 12. Limitations

- Skill extraction depends on predefined skill keywords.
- TF-IDF similarity measures textual similarity and does not directly measure actual candidate ability.
- Predicted resume categories do not represent final candidate suitability.
- Machine Learning predictions may contain classification errors.
- Human review is recommended before making recruitment decisions.

## 13. Conclusion

The AI-Based Resume Screening & Candidate Analysis System demonstrates the use of Machine Learning, NLP, and web technologies for preliminary resume analysis.

It combines resume parsing, skill extraction, candidate scoring, classification, and similarity analysis in a single web application.

The project provides an organized approach to comparing resumes with job descriptions and generating candidate analysis reports.