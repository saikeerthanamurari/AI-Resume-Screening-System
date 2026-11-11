import json
import joblib
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

print("Loading resume dataset...")

data_path = Path.home() / "Downloads/archive/Resume/Resume.csv"
df = pd.read_csv(data_path)

df = df.dropna(subset=["Resume_str", "Category"])
df["Resume_str"] = df["Resume_str"].astype(str).str.strip()
df = df[df["Resume_str"] != ""]
df = df.drop_duplicates(subset=["Resume_str"])

X = df["Resume_str"]
y = df["Category"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

word_features = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_features=30000,
    min_df=2,
    sublinear_tf=True
)

char_features = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    max_features=30000,
    min_df=2,
    sublinear_tf=True
)

combined_features = FeatureUnion([
    ("word", TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=30000,
        min_df=2,
        sublinear_tf=True
    )),
    ("char", TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=30000,
        min_df=2,
        sublinear_tf=True
    ))
])

pipeline = Pipeline([
    ("features", word_features),
    ("classifier", LinearSVC(random_state=42))
])

parameters = [
    {
        "features": [word_features],
        "classifier__C": [0.1, 0.3, 1, 3, 10],
        "classifier__class_weight": [None, "balanced"]
    },
    {
        "features": [combined_features],
        "classifier__C": [0.1, 0.3, 1, 3, 10],
        "classifier__class_weight": [None, "balanced"]
    }
]

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

search = GridSearchCV(
    pipeline,
    parameters,
    scoring="accuracy",
    cv=cv,
    n_jobs=1,
    verbose=2,
    refit=True
)

print("\nStarting hyperparameter tuning...")
print("This may take some time on your Mac.")

search.fit(X_train, y_train)

print("\nBest configuration:")
print(search.best_params_)

print(
    "Best cross-validation accuracy:",
    round(search.best_score_ * 100, 2), "%"
)

best_model = search.best_estimator_

predictions = best_model.predict(X_test)

metrics = {
    "selected_model": "Tuned Linear SVM",
    "best_parameters": {
        key: str(value)
        for key, value in search.best_params_.items()
    },
    "cross_validation_accuracy": float(search.best_score_),
    "test_accuracy": float(accuracy_score(y_test, predictions)),
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
    ))
}

Path("models").mkdir(exist_ok=True)
Path("outputs").mkdir(exist_ok=True)

joblib.dump(
    best_model,
    "models/resume_classifier_tuned.joblib"
)

with open("outputs/tuned_ml_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

with open("outputs/tuned_classification_report.txt", "w") as f:
    f.write(classification_report(
        y_test, predictions, zero_division=0
    ))

pd.DataFrame(search.cv_results_).to_csv(
    "outputs/hyperparameter_tuning_results.csv",
    index=False
)

print("\n========== TUNED MODEL RESULTS ==========")
print("CV Accuracy:", round(search.best_score_ * 100, 2), "%")
print("Test Accuracy:", round(metrics["test_accuracy"] * 100, 2), "%")
print("Macro Precision:", round(metrics["precision_macro"] * 100, 2), "%")
print("Macro Recall:", round(metrics["recall_macro"] * 100, 2), "%")
print("Macro F1:", round(metrics["f1_macro"] * 100, 2), "%")
print("Weighted F1:", round(metrics["f1_weighted"] * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test, predictions, zero_division=0
))

print("\nTuned model saved separately.")
print("Your existing model has NOT been changed.")
