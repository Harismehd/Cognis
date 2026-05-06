import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    precision_recall_fscore_support,
)

CRITICAL_PAIRS = [
    ("missing_colon", "indentation_error"),
    ("spelling_error", "undefined_variable"),
    ("comparison_vs_assignment", "wrong_operator"),
    ("type_confusion", "wrong_formula"),
    ("index_error", "off_by_one_error"),
    ("missing_parenthesis", "missing_bracket"),
    ("infinite_loop_risk", "wrong_loop_condition"),
    ("concept_gap", "wrong_formula"),
]

def evaluate() -> None:
    print("Loading precision data...")
    examples = []
    with open(os.path.join("data", "precision_diagnostics.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))

    df = pd.DataFrame(examples)
    y_error_all = df["TARGET"].apply(lambda t: t["diagnosis"][0]["error_type"]).astype(str)
    
    if len(df) > 30000:
        print("Sampling 30,000 records from the full generated dataset for evaluation...")
        df, _ = train_test_split(
            df,
            train_size=30000,
            stratify=y_error_all,
            random_state=42,
            shuffle=True,
        )
        y_error_all = df["TARGET"].apply(lambda t: t["diagnosis"][0]["error_type"]).astype(str)

    X_raw = df["INPUT"]
    y_error = df["TARGET"].apply(lambda t: t["diagnosis"][0]["error_type"]).astype(str)

    from expert_transformers import PrecisionInputExtractor, JsonNumericalExtractor, JsonCategoricalExtractor
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer

    input_expander = PrecisionInputExtractor()
    X_expanded = input_expander.transform(X_raw)
    X_expanded["code"] = X_expanded["code"].fillna("")

    preprocessor = ColumnTransformer(
        transformers=[
            ("code_vec", CountVectorizer(analyzer="char", ngram_range=(1, 2), max_features=1200), "code"),
            ("key_oh", OneHotEncoder(handle_unknown="ignore"), ["key"]),
            ("cursor_ext", JsonNumericalExtractor("cursor", ["lineNumber", "column"]), ["cursor"]),
            ("hist_num", JsonNumericalExtractor("history", ["past_errors", "struggle_concepts"]), ["history"]),
            (
                "hist_cat",
                Pipeline([
                    ("ext", JsonCategoricalExtractor("history", "hint_style")),
                    ("oh", OneHotEncoder(handle_unknown="ignore")),
                ]),
                ["history"],
            ),
            ("typing_state", OneHotEncoder(handle_unknown="ignore"), ["typing_state"]),
            ("compile_type", OneHotEncoder(handle_unknown="ignore"), ["compile_error_type"]),
            ("compile_numeric", "passthrough", ["compile_error_flag", "compile_error_line", "compile_error_severity"]),
        ],
        sparse_threshold=0,
    )

    print("Fitting preprocessor...")
    X_processed = preprocessor.fit_transform(X_expanded)

    X_train, X_test, y_error_train, y_error_test = train_test_split(
        X_processed,
        y_error,
        test_size=0.15,
        stratify=y_error,
        random_state=42,
    )

    print("Loading trained model...")
    bundle = joblib.load(os.path.join("models", "precision_cognis_bundle.joblib"))
    error_model = bundle["models"]["error_type"]

    print("Evaluating on holdout test set...")
    y_error_pred = error_model.predict(X_test)
    y_error_proba = error_model.predict_proba(X_test)

    classes = list(error_model.classes_)
    precision, recall, f1, support = precision_recall_fscore_support(y_error_test, y_error_pred, labels=classes, zero_division=0)
    report = classification_report(y_error_test, y_error_pred, labels=classes, zero_division=0, output_dict=True)
    conf_mat = confusion_matrix(y_error_test, y_error_pred, labels=classes)

    no_error_mask = np.array(y_error_test) == "no_error"
    if np.sum(no_error_mask) > 0:
        false_positive = np.sum((np.array(y_error_pred)[no_error_mask] != "no_error"))
        false_positive_rate = float(false_positive) / float(np.sum(no_error_mask))
    else:
        false_positive_rate = float("nan")

    max_probs = np.max(y_error_proba, axis=1)
    correct_mask = np.array(y_error_pred) == np.array(y_error_test)
    mean_conf_correct = float(np.mean(max_probs[correct_mask])) if np.any(correct_mask) else float("nan")
    mean_conf_incorrect = float(np.mean(max_probs[~correct_mask])) if np.any(~correct_mask) else float("nan")

    misclassification_rates = {}
    for a, b in CRITICAL_PAIRS:
        mask_a = np.array(y_error_test) == a
        mask_b = np.array(y_error_test) == b
        rate_ab = float(np.sum(np.array(y_error_pred)[mask_a] == b)) / max(1, int(np.sum(mask_a)))
        rate_ba = float(np.sum(np.array(y_error_pred)[mask_b] == a)) / max(1, int(np.sum(mask_b)))
        misclassification_rates[f"{a}_as_{b}"] = rate_ab
        misclassification_rates[f"{b}_as_{a}"] = rate_ba

    metrics = {
        "total_examples": int(X_processed.shape[0]),
        "holdout_size": int(X_test.shape[0]),
        "error_type_accuracy": float(accuracy_score(y_error_test, y_error_pred)),
        "false_positive_rate": false_positive_rate,
        "mean_confidence_correct": mean_conf_correct,
        "mean_confidence_incorrect": mean_conf_incorrect,
        "per_type": {
            classes[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(len(classes))
        },
        "confusion_matrix": {
            "labels": classes,
            "matrix": conf_mat.tolist(),
        },
        "critical_pair_misclassification": misclassification_rates,
    }

    report_path = os.path.join("models", "precision_diagnostics_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved evaluation report to {report_path}")

    bundle_size = os.path.getsize(os.path.join("models", "precision_cognis_bundle.joblib")) / 1024.0
    print(f"Model bundle size: {bundle_size:.2f} KB")
    print("=== Error type holdout report ===")
    print(f"Total accuracy: {metrics['error_type_accuracy']:.4f}")
    print(f"False positive rate (no_error as positive): {false_positive_rate:.4f}")
    print(f"Mean confidence correct: {mean_conf_correct:.4f}")
    print(f"Mean confidence incorrect: {mean_conf_incorrect:.4f}")
    print("\n=== Per-type precision, recall, f1 ===")
    for et in classes:
        v = metrics['per_type'].get(et, {})
        print(f"{et:30s} | Precision: {v.get('precision', 0):.3f} | Recall: {v.get('recall', 0):.3f} | F1: {v.get('f1', 0):.3f} | Support: {v.get('support', 0)}")
    
    print("\n=== Critical pair misclassification rates ===")
    for pair_name, rate in misclassification_rates.items():
        print(f"  {pair_name}: {rate:.4f}")


if __name__ == "__main__":
    evaluate()
