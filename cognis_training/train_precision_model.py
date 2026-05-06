import json
import os
import joblib
import numpy as np
import pandas as pd
from typing import Any, Dict, List
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    precision_recall_fscore_support,
)

from expert_transformers import JsonNumericalExtractor, JsonCategoricalExtractor, PrecisionInputExtractor

MODEL_OUTPUT_PATH = os.path.join("models", "precision_cognis_bundle.joblib")
REPORT_OUTPUT_PATH = os.path.join("models", "precision_diagnostics_report.json")

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


def train() -> None:
    print("Loading precision data...")
    examples: List[Dict[str, Any]] = []
    with open(os.path.join("data", "precision_diagnostics.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))

    df = pd.DataFrame(examples)
    y_error_all = df["TARGET"].apply(lambda t: t["diagnosis"][0]["error_type"]).astype(str)
    if len(df) > 50000:
        print("Sampling 50,000 records from the full generated dataset for training efficiency...")
        df, _ = train_test_split(
            df,
            train_size=50000,
            stratify=y_error_all,
            random_state=42,
            shuffle=True,
        )

    X_raw = df["INPUT"]
    y_risk = df["TARGET"].apply(lambda t: t["error_risk"]).astype(float)
    y_speak = df["TARGET"].apply(lambda t: 1 if t["should_speak"] else 0).astype(int)
    y_intervention = df["TARGET"].apply(lambda t: t["intervention_type"]).astype(str)
    y_error = df["TARGET"].apply(lambda t: t["diagnosis"][0]["error_type"]).astype(str)
    y_msg = df["TARGET"].apply(lambda t: t["message"]).astype(str)
    y_tone = df["TARGET"].apply(lambda t: t["mentor_tone"]).astype(str)
    y_action = df["TARGET"].apply(lambda t: t["next_best_action"]).astype(str)

    input_expander = PrecisionInputExtractor()
    X_expanded = input_expander.transform(X_raw)
    X_expanded["code"] = X_expanded["code"].fillna("")

    preprocessor = ColumnTransformer(
        transformers=[
            ("code_vec", CountVectorizer(analyzer="char", ngram_range=(1, 3), max_features=2500), "code"),
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

    X_train, X_test, y_risk_train, y_risk_test, y_speak_train, y_speak_test, y_intervention_train, y_intervention_test, y_error_train, y_error_test, y_tone_train, y_tone_test, y_action_train, y_action_test = train_test_split(
        X_processed,
        y_risk,
        y_speak,
        y_intervention,
        y_error,
        y_tone,
        y_action,
        test_size=0.15,
        stratify=y_error,
        random_state=42,
    )

    bundle: Dict[str, Any] = {
        "preprocessor": preprocessor,
        "input_expander": input_expander,
        "models": {},
    }

    targets = {
        "risk": (
            y_risk_train,
            HistGradientBoostingRegressor(
                random_state=42,
                max_depth=6,
                learning_rate=0.07,
                max_iter=140,
                max_bins=128,
                early_stopping=True,
                n_iter_no_change=20,
                validation_fraction=0.1,
            ),
        ),
        "speak": (
            y_speak_train,
            HistGradientBoostingClassifier(
                random_state=42,
                max_depth=4,
                learning_rate=0.08,
                max_iter=120,
                max_bins=128,
                early_stopping=True,
                n_iter_no_change=20,
                validation_fraction=0.1,
            ),
        ),
        "intervention": (
            y_intervention_train,
            HistGradientBoostingClassifier(
                random_state=42,
                max_depth=4,
                learning_rate=0.08,
                max_iter=120,
                max_bins=128,
                early_stopping=True,
                n_iter_no_change=20,
                validation_fraction=0.1,
            ),
        ),
        "error_type": (
            y_error_train,
            HistGradientBoostingClassifier(
                random_state=42,
                max_depth=8,
                learning_rate=0.08,
                max_iter=250,
                max_bins=128,
                early_stopping=True,
                n_iter_no_change=25,
                validation_fraction=0.1,
            ),
        ),
        "tone": (
            y_tone_train,
            HistGradientBoostingClassifier(
                random_state=42,
                max_depth=4,
                learning_rate=0.08,
                max_iter=120,
                max_bins=128,
                early_stopping=True,
                n_iter_no_change=20,
                validation_fraction=0.1,
            ),
        ),
        "action": (
            y_action_train,
            HistGradientBoostingClassifier(
                random_state=42,
                max_depth=4,
                learning_rate=0.08,
                max_iter=120,
                max_bins=128,
                early_stopping=True,
                n_iter_no_change=20,
                validation_fraction=0.1,
            ),
        ),
    }

    msg_map: Dict[str, str] = {}
    for et, msg, is_rejection in zip(y_error, y_msg, df["TARGET"].apply(lambda t: "rejected_type" in t)):
        if et not in msg_map and not is_rejection:
            msg_map[et] = msg
    
    # Fallback for types that might only have rejection messages (unlikely but safe)
    for et, msg in zip(y_error, y_msg):
        if et not in msg_map:
            msg_map[et] = msg
            
    bundle["msg_map"] = msg_map

    for name, (y_train, model) in targets.items():
        print(f"Training {name} model...")
        model.fit(X_train, y_train)
        bundle["models"][name] = model

    print("Evaluating holdout performance...")
    error_model = bundle["models"]["error_type"]
    y_error_pred = error_model.predict(X_test)
    y_error_proba = error_model.predict_proba(X_test)

    classes = list(error_model.classes_)
    precision, recall, f1, support = precision_recall_fscore_support(y_error_test, y_error_pred, labels=classes, zero_division=0)
    report = classification_report(y_error_test, y_error_pred, labels=classes, zero_division=0)
    conf_mat = confusion_matrix(y_error_test, y_error_pred, labels=classes)

    no_error_mask = y_error_test == "no_error"
    if np.sum(no_error_mask) > 0:
        false_positive = np.sum((y_error_pred[no_error_mask] != "no_error"))
        false_positive_rate = float(false_positive) / float(np.sum(no_error_mask))
    else:
        false_positive_rate = float("nan")

    max_probs = np.max(y_error_proba, axis=1)
    correct_mask = y_error_pred == np.array(y_error_test)
    mean_conf_correct = float(np.mean(max_probs[correct_mask])) if np.any(correct_mask) else float("nan")
    mean_conf_incorrect = float(np.mean(max_probs[~correct_mask])) if np.any(~correct_mask) else float("nan")

    misclassification_rates: Dict[str, float] = {}
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

    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    joblib.dump(bundle, MODEL_OUTPUT_PATH)
    print(f"Saved precision bundle to {MODEL_OUTPUT_PATH}")

    bundle_size = os.path.getsize(MODEL_OUTPUT_PATH) / 1024.0
    print(f"Model bundle size: {bundle_size:.2f} KB")
    print("=== Error type holdout report ===")
    print(report)
    print(f"False positive rate (no_error as positive): {false_positive_rate:.4f}")
    print(f"Mean confidence correct: {mean_conf_correct:.4f}")
    print(f"Mean confidence incorrect: {mean_conf_incorrect:.4f}")
    print("=== Critical pair misclassification rates ===")
    for pair_name, rate in misclassification_rates.items():
        print(f"  {pair_name}: {rate:.4f}")

    with open(REPORT_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved evaluation report to {REPORT_OUTPUT_PATH}")


if __name__ == "__main__":
    train()
