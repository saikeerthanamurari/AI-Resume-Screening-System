
import os
import uuid
import csv
import joblib

from flask import (
    Flask,
    render_template,
    request,
    send_from_directory
)

from werkzeug.utils import secure_filename

from resume_parser import extract_resume_text
from preprocessing import extract_skills
from candidate_scoring import analyze_candidate
from train_model import calculate_text_similarity


app = Flask(__name__)

# Folder configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
REPORT_FOLDER = os.path.join(BASE_DIR, "reports")
MODEL_PATH = os.path.join(BASE_DIR, "models", "resume_classifier_tuned.joblib")
resume_classifier = joblib.load(MODEL_PATH)

ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)


# Check supported resume formats
def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# Homepage
@app.route("/")
def home():
    return render_template("index.html")


# Resume analysis
@app.route("/analyze", methods=["POST"])
def analyze():

    job_description = request.form.get(
        "job_description", ""
    ).strip()

    upload_mode = request.form.get("upload_mode", "single")

    resume_files = request.files.getlist("resumes")

    if not job_description:
        return "Please enter a job description.", 400

    if upload_mode not in {"single", "multiple"}:
        return "Invalid upload mode.", 400

    valid_files = [
        file for file in resume_files
        if file and file.filename
    ]

    if not valid_files:
        return "Please upload at least one resume.", 400

    # Enforce single upload on the server
    if upload_mode == "single" and len(valid_files) != 1:
        return "Please upload exactly one resume in single mode.", 400

    required_skills = extract_skills(job_description)

    results = []

    # Process uploaded resumes
    for file in valid_files:

        if not allowed_file(file.filename):
            continue

        original_name = secure_filename(file.filename)

        if not original_name:
            continue

        unique_name = f"{uuid.uuid4().hex}_{original_name}"

        file_path = os.path.join(
            UPLOAD_FOLDER,
            unique_name
        )

        file.save(file_path)

        try:
            # Extract resume text
            resume_text = extract_resume_text(file_path)

            if not resume_text.strip():
                continue

            # Predict resume occupational category using trained ML model
            predicted_category = resume_classifier.predict([resume_text])[0]

            # Extract candidate skills
            candidate_skills = extract_skills(resume_text)

            # Calculate skill matching
            skill_analysis = analyze_candidate(
                candidate_skills,
                required_skills
            )

            # Calculate TF-IDF similarity
            similarity_score = calculate_text_similarity(
                resume_text,
                job_description
            )

            results.append({
    "filename": original_name,
    "predicted_category": predicted_category,
    "skills": candidate_skills,

    "matched_skills": skill_analysis["matched_skills"],
    "missing_skills": skill_analysis["missing_skills"],
    "extra_skills": skill_analysis["extra_skills"],

    "skill_score": skill_analysis["match_score"],

    "total_required_skills": skill_analysis["total_required_skills"],
    "total_matched_skills": skill_analysis["total_matched_skills"],
    "total_missing_skills": skill_analysis["total_missing_skills"],

    "summary": skill_analysis["summary"],

    "similarity_score": similarity_score
})

        except Exception as error:
            print(f"Could not process {original_name}: {error}")

    if not results:
        return (
            "No readable resumes were processed. "
            "Check that your files are valid PDF, DOCX or TXT files.",
            400
        )

    # Rank candidates only in multiple mode
    if upload_mode == "multiple":
        results.sort(
            key=lambda item: item["similarity_score"],
            reverse=True
        )

    # Generate a unique CSV report

    # Generate a unique CSV report
    report_filename = f"candidate_report_{uuid.uuid4().hex}.csv"

    report_path = os.path.join(
        REPORT_FOLDER,
        report_filename
    )

    with open(
        report_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as csvfile:

        fieldnames = [
            "Candidate",
            "Predicted Resume Category",
            "Identified Skills",
            "Matched Skills",
            "Missing Skills",
            "Extra Skills",
            "Total Required Skills",
            "Total Matched Skills",
            "Total Missing Skills",
            "Skill Matching Score",
            "TF-IDF Similarity Score",
            "Candidate Summary"
        ]

        writer = csv.DictWriter(
            csvfile,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for candidate in results:

            writer.writerow({
                "Candidate": candidate["filename"],
                "Predicted Resume Category": candidate["predicted_category"],
                "Identified Skills": ", ".join(candidate["skills"]),
                "Matched Skills": ", ".join(candidate["matched_skills"]),
                "Missing Skills": ", ".join(candidate["missing_skills"]),
                "Extra Skills": ", ".join(candidate["extra_skills"]),
                "Total Required Skills": candidate["total_required_skills"],
                "Total Matched Skills": candidate["total_matched_skills"],
                "Total Missing Skills": candidate["total_missing_skills"],
                "Skill Matching Score": candidate["skill_score"],
                "TF-IDF Similarity Score": candidate["similarity_score"],
                "Candidate Summary": candidate["summary"]
            })


    # Display results
    return render_template(
        "results.html",
        results=results,
        required_skills=required_skills,
        report_filename=report_filename,
        upload_mode=upload_mode
    )


# Download CSV report
@app.route("/download/<filename>")
def download_report(filename):

    if (
        not filename.startswith("candidate_report_")
        or not filename.endswith(".csv")
        or "/" in filename
        or "\\" in filename
    ):
        return "Invalid report filename.", 400

    return send_from_directory(
        REPORT_FOLDER,
        filename,
        as_attachment=True
    )


# Handle oversized uploads
@app.errorhandler(413)
def file_too_large(error):
    return (
        "The uploaded files are too large. Maximum total size is 10 MB.",
        413
    )


# Run Flask application
if __name__ == "__main__":
    app.run(debug=True, port=5000)
