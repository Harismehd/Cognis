from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from cognis_features import extract_features


SchemaStudent = Dict[str, Any]


SCAFFOLDING_CLASSES: Tuple[str, ...] = ("high_support", "moderate_hints", "blank_canvas")


def load_jsonl(path: str) -> List[SchemaStudent]:
    items: List[SchemaStudent] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _to_row(numeric: Dict[str, float], categorical: Dict[str, str], n_keys: List[str], c_keys: List[str]) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    for k in n_keys:
        row[k] = float(numeric.get(k, 0.0))
    for k in c_keys:
        row[k] = str(categorical.get(k, ""))
    return row


def build_feature_table(students: List[SchemaStudent]) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    numeric_keys: List[str] = []
    categorical_keys: List[str] = []

    # Determine stable feature name ordering based on first record.
    n0, c0 = extract_features(students[0])
    numeric_keys = sorted(n0.keys())
    categorical_keys = sorted(c0.keys())

    rows: List[Dict[str, Any]] = []
    for s in students:
        n, c = extract_features(s)
        rows.append(_to_row(n, c, numeric_keys, categorical_keys))
    return rows, numeric_keys, categorical_keys


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def derive_labels(students: List[SchemaStudent], seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create training labels from the telemetry itself (bootstrapping).
    This keeps the model runnable before real production labels exist.
    """
    rng = random.Random(seed)
    y_struggle: List[float] = []
    y_scaffold: List[str] = []
    y_intervene: List[int] = []

    for s in students:
        perf = s["performance"]
        ea = s["error_analysis"]
        av = s["anti_vibe_coding"]
        mastery = s["concept_mastery"]
        ls = s["learning_style"]
        cl = s["cognitive_load"]

        total_problems = float(perf["total_problems"])
        total_errors = float(sum(ea["error_types"].values()))
        error_rate = (total_errors / total_problems) if total_problems else 0.0
        repeated = float(ea["repeated_errors"])
        paste = float(av["paste_count"])
        tabs = float(av["tab_switches"])
        loops_m = float(mastery["loops"])
        confidence = float(ls["confidence"])
        intermediate_ratio = (
            float(cl["difficulty_distribution"]["intermediate"])
            / max(1.0, float(cl["difficulty_distribution"]["beginner"]) + float(cl["difficulty_distribution"]["intermediate"]))
        )

        # Anti-vibe / intervention: deterministic-ish, with slight noise.
        intervene_score = 0.9 * paste + 0.05 * tabs + 0.35 * repeated + 0.4 * error_rate * 10.0
        intervene = 1 if intervene_score >= 7.5 else 0
        if rng.random() < 0.03:
            intervene = 1 - intervene
        y_intervene.append(int(intervene))

        # Struggle risk proxy (next intermediate concept): logistic of low loops mastery + high error/repeats + low confidence
        z = (
            1.4 * (1.0 - loops_m / 100.0)
            + 1.2 * error_rate
            + 0.06 * repeated
            + 0.04 * tabs
            + 0.22 * paste
            + 0.6 * (1.0 - confidence / 100.0)
            + 0.4 * intermediate_ratio
        )
        z += rng.uniform(-0.25, 0.25)
        struggle_prob = float(_sigmoid(z))
        y_struggle.append(struggle_prob)

        # Scaffolding recommendation: map to 3-class using thresholds & signals.
        # - high_support: high struggle or very low confidence/mastery
        # - moderate_hints: mid struggle / wants hints
        # - blank_canvas: low struggle / high confidence & mastery
        prefers_hints = bool(ls["prefers_hints"])

        if struggle_prob >= 0.72 or loops_m < 35 or confidence < 25:
            cls = "high_support"
        elif struggle_prob <= 0.28 and loops_m >= 70 and confidence >= 60 and not prefers_hints:
            cls = "blank_canvas"
        else:
            cls = "moderate_hints"

        # If student prefers hints and is not low-risk, bias to moderate/high.
        if prefers_hints and cls == "blank_canvas":
            cls = "moderate_hints"
        if intervene and cls == "blank_canvas":
            cls = "moderate_hints"

        y_scaffold.append(cls)

    return np.array(y_struggle, dtype=float), np.array(y_scaffold, dtype=object), np.array(y_intervene, dtype=int)


def make_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )


def train_models(rows: List[Dict[str, Any]], numeric_cols: List[str], categorical_cols: List[str], y_struggle, y_scaffold, y_intervene, seed: int):
    import pandas as pd

    X = pd.DataFrame(rows)

    X_train, X_test, ys_train, ys_test = train_test_split(
        X,
        np.c_[y_struggle, y_scaffold, y_intervene],
        test_size=0.2,
        random_state=seed,
        shuffle=True,
    )
    yS_train = ys_train[:, 0].astype(float)
    yC_train = ys_train[:, 1].astype(str)
    yI_train = ys_train[:, 2].astype(int)

    yS_test = ys_test[:, 0].astype(float)
    yC_test = ys_test[:, 1].astype(str)
    yI_test = ys_test[:, 2].astype(int)

    pre = make_preprocessor(numeric_cols, categorical_cols)

    struggle_model = Pipeline(
        steps=[
            ("pre", pre),
            (
                "model",
                HistGradientBoostingRegressor(
                    random_state=seed,
                    max_depth=6,
                    learning_rate=0.08,
                    max_iter=250,
                ),
            ),
        ]
    )

    scaffolding_model = Pipeline(
        steps=[
            ("pre", pre),
            (
                "model",
                HistGradientBoostingClassifier(
                    random_state=seed,
                    max_depth=6,
                    learning_rate=0.08,
                    max_iter=250,
                ),
            ),
        ]
    )

    intervention_model = Pipeline(
        steps=[
            ("pre", pre),
            (
                "model",
                HistGradientBoostingClassifier(
                    random_state=seed,
                    max_depth=4,
                    learning_rate=0.08,
                    max_iter=200,
                ),
            ),
        ]
    )

    struggle_model.fit(X_train, yS_train)
    scaffolding_model.fit(X_train, yC_train)
    intervention_model.fit(X_train, yI_train)

    # Lightweight sanity metrics (synthetic labels -> expected to be fairly learnable)
    predS = np.clip(struggle_model.predict(X_test), 0.0, 1.0)
    # Struggle is trained as a regressor over [0,1] probabilities, so use regression metrics.
    maeS = mean_absolute_error(yS_test, predS)
    r2S = r2_score(yS_test, predS)

    predC = scaffolding_model.predict(X_test)
    accC = accuracy_score(yC_test, predC)

    probI = intervention_model.predict_proba(X_test)[:, 1]
    predI = (probI >= 0.5).astype(int)
    try:
        aucI = roc_auc_score(yI_test, probI)
    except Exception:
        aucI = float("nan")
    accI = accuracy_score(yI_test, predI)

    metrics = {
        "struggle_mae": float(maeS),
        "struggle_r2": float(r2S),
        "scaffolding_accuracy": float(accC),
        "intervention_auc": float(aucI) if np.isfinite(aucI) else None,
        "intervention_accuracy": float(accI),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    return struggle_model, scaffolding_model, intervention_model, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Cognis lightweight models from schema-fixed telemetry JSONL.")
    parser.add_argument("--data", type=str, default=os.path.join("data", "students.jsonl"), help="Path to students.jsonl")
    parser.add_argument("--seed", type=int, default=42, help="Seed (default: 42)")
    parser.add_argument("--out", type=str, default=os.path.join("models", "cognis_bundle.joblib"), help="Output model bundle")
    args = parser.parse_args()

    data_path = os.path.normpath(args.data)
    students = load_jsonl(data_path)
    if len(students) < 1000:
        raise SystemExit(f"Expected >= 1000 students, got {len(students)} from {data_path}")

    rows, numeric_cols, categorical_cols = build_feature_table(students)
    y_struggle, y_scaffold, y_intervene = derive_labels(students, seed=int(args.seed))

    struggle_model, scaffolding_model, intervention_model, metrics = train_models(
        rows, numeric_cols, categorical_cols, y_struggle, y_scaffold, y_intervene, seed=int(args.seed)
    )

    # IMPORTANT: persist as a plain dict (not a dataclass) so joblib can load
    # reliably regardless of how training was invoked (__main__ vs module import).
    bundle: Dict[str, Any] = {
        "struggle_model": struggle_model,
        "scaffolding_model": scaffolding_model,
        "intervention_model": intervention_model,
        "numeric_feature_names": numeric_cols,
        "categorical_feature_names": categorical_cols,
    }

    out_path = os.path.normpath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump({"bundle": bundle, "metrics": metrics}, out_path)
    print(f"Wrote model bundle to {out_path}")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

