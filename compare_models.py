import json
import joblib
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

data_path = Path.home() / "Downloads/archive/Resume/Resume.csv"
df = pd.read_csv(data_path)

df = df.dropna(subset=["Resume_str", "Category"])
df["Resume_str"] = df["Resume_str"].astype(str).str.strip()
df = df[df["Resume_str"] != ""]
df = df.drop_duplicates(subset=["Resume_str"])

X = df["Resume_str"]
y = df["Category"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

models = {
    "Logistic Regression - Word TF-IDF": Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=30000,
            min_df=2,
            sublinear_tf=True
        )),
        ("classifier", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42
        ))
    ]),

    "Linear SVM - Word TF-IDF": Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=30000,
            min_df=2,
            sublinear_tf=True
        )),
        ("classifier", LinearSVC(
            class_weight="balanced",
            random_state=42
        ))
    ]),

    "Logistic Regression - Character TF-IDF": Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=30000,
            min_df=2,
            sublinear_tf=True
        )),
        ("classifier", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42
        ))
    ]),

    "Linear SVM - Character TF-IDF": Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=30000,
            min_df=2,
            sublinear_tf=True
        )),
        ("classifier", LinearSVC(
            class_weight="balanced",
            random_state=42
        ))
    ])
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []
best_name = None
best_score = -1
best_model = None

print("Training resumes:", len(X_train))
print("Testing resumes:", len(X_test))
print("\nComparing four models using 5-fold cross-validation...\n")

for name, model in models.items():
    scores = cross_val_score(
        model, X_train, y_train,
        cv=cv,
        scoring="f1_macro",
        n_jobs=1
    )

    mean_score = scores.mean()
    results.append({
        "model": name,
        "mean_macro_f1_cv": mean_score,
        "std_macro_f1_cv": scores.std()
    })

    print(name)
    print("Mean CV Macro F1:", round(mean_score * 100, 2), "%")
    print("Standard deviation:", round(scores.std() * 100, 2), "%\n")

    if mean_score > best_score:
        best_score = mean_score
        best_name = name
        best_model = model

print("Best model:", best_name)
print("\nTraining the selected model on all training data...")
best_model.fit(X_train, y_train)

predictions = best_model.predict(X_test)

metrics = {
    "selected_model": best_name,
    "cross_validation_macro_f1": float(best_score),
    "accuracy": float(accuracy_score(y_test, predictions)),
    "precision_macro": float(precision_score(
        y_test, predictions, average="macro", zero_division=0
    )),
    "recall_macro": float(recall_score(
        y_test, predictions, average="macro", zero_division=0
    )),
    "f1_macro": float(f1_score(
        y_test, predictions, average="macro", zero_division=0
    )),
    "f1_weighted": float(f1_score(
        y_test, predictions, average="weighted", zero_division=0
    )),
    "training_samples": len(X_train),
    "testing_samples": len(X_test),
    "categories": int(y.nunique())
}

Path("models").mkdir(exist_ok=True)
Path("outputs").mkdir(exist_ok=True)

joblib.dump(best_model, "models/resume_classifier.joblib")

with open("outputs/ml_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

report = classification_report(
    y_test, predictions, zero_division=0
)

with open("outputs/classification_report.txt", "w") as f:
    f.write(report)

categories = sorted(y.unique())
matrix = confusion_matrix(y_test, predictions, labels=categories)

with open("outputs/confusion_matrix.csv", "w", newline="") as f:
    import csv
    writer = csv.writer(f)
    writer.writerow(["Actual / Predicted"] + categories)
    for category, values in zip(categories, matrix):
        writer.writerow([category] + values.tolist())

with open("outputs/test_predictions.csv", "w", newline="") as f:
    import csv
    writer = csv.writer(f)
    writer.writerow(["Actual Category", "Predicted Category"])
    writer.writerows(zip(y_test, predictions))

pd.DataFrame(results).to_csv(
    "outputs/model_comparison.csv", index=False
)

print("\n========== FINAL TEST RESULTS ==========")
print("Selected model:", best_name)
print("Accuracy:", round(metrics["accuracy"] * 100, 2), "%")
print("Macro Precision:", round(metrics["precision_macro"] * 100, 2), "%")
print("Macro Recall:", round(metrics["recall_macro"] * 100, 2), "%")
print("Macro F1:", round(metrics["f1_macro"] * 100, 2), "%")
print("Weighted F1:", round(metrics["f1_weighted"] * 100, 2), "%")
print("\n", report)
print("Saved selected model and evaluation outputs.")
