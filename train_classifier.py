import csv
import json
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

DATASET = Path.home() / "Downloads/archive/Resume/Resume.csv"

with open(DATASET, "r", encoding="utf-8-sig") as file:
    rows = list(csv.DictReader(file))

texts = []
labels = []

for row in rows:
    text = (row.get("Resume_str") or "").strip()
    category = (row.get("Category") or "").strip()

    if text and category:
        texts.append(text)
        labels.append(category)

print("Total usable resumes:", len(texts))
print("Total categories:", len(set(labels)))

X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.20,
    random_state=42,
    stratify=labels
)

print("Training resumes:", len(X_train))
print("Testing resumes:", len(X_test))

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=20000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )),
    ("classifier", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ))
])

print("\nTraining Logistic Regression model...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

metrics = {
    "dataset": "Resume Dataset",
    "total_resumes": len(texts),
    "training_resumes": len(X_train),
    "testing_resumes": len(X_test),
    "categories": len(set(labels)),
    "algorithm": "Logistic Regression",
    "feature_extraction": "TF-IDF",
    "accuracy": accuracy_score(y_test, predictions),
    "precision_macro": precision_score(
        y_test, predictions, average="macro", zero_division=0
    ),
    "recall_macro": recall_score(
        y_test, predictions, average="macro", zero_division=0
    ),
    "f1_macro": f1_score(
        y_test, predictions, average="macro", zero_division=0
    ),
    "precision_weighted": precision_score(
        y_test, predictions, average="weighted", zero_division=0
    ),
    "recall_weighted": recall_score(
        y_test, predictions, average="weighted", zero_division=0
    ),
    "f1_weighted": f1_score(
        y_test, predictions, average="weighted", zero_division=0
    )
}

Path("models").mkdir(exist_ok=True)
Path("outputs").mkdir(exist_ok=True)

joblib.dump(model, "models/resume_classifier.joblib")

with open("outputs/ml_metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

report = classification_report(
    y_test, predictions, zero_division=0
)

with open("outputs/classification_report.txt", "w") as file:
    file.write(report)

categories = sorted(set(labels))
matrix = confusion_matrix(y_test, predictions, labels=categories)

with open("outputs/confusion_matrix.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Actual / Predicted"] + categories)

    for category, values in zip(categories, matrix):
        writer.writerow([category] + values.tolist())

with open("outputs/test_predictions.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Actual Category", "Predicted Category"])
    writer.writerows(zip(y_test, predictions))

print("\n========== MODEL EVALUATION ==========")
print("Accuracy:", round(metrics["accuracy"] * 100, 2), "%")
print("Macro Precision:", round(metrics["precision_macro"] * 100, 2), "%")
print("Macro Recall:", round(metrics["recall_macro"] * 100, 2), "%")
print("Macro F1-score:", round(metrics["f1_macro"] * 100, 2), "%")
print("Weighted F1-score:", round(metrics["f1_weighted"] * 100, 2), "%")

print("\nClassification Report:\n")
print(report)

print("Confusion matrix saved.")
print("Trained model saved.")
print("Training completed successfully.")
