import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.base import BaseEstimator, TransformerMixin

from expert_transformers import JsonNumericalExtractor, JsonCategoricalExtractor

def train():
    df = pd.read_csv(os.path.join("data", "expert_keystrokes.csv"))
    df['code'] = df['code'].fillna("")
    
    # Define features
    # We'll use 'code' (text), 'key_pressed' (cat), 'cursor' (json), 'student_history' (json)
    
    X = df[['code', 'key_pressed', 'cursor', 'student_history']]
    
    # Targets
    y_risk = df['risk_score']
    y_intervention = df['intervention_type']
    y_category = df['error_category']
    y_hint = df['hint_id']

    # Preprocessing
    # Code: CountVectorizer (n-grams of chars)
    # Key: OneHot
    # Cursor/History: JsonFeatureExtractor
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('code_vec', CountVectorizer(analyzer='char', ngram_range=(1, 3)), 'code'),
            ('key_oh', OneHotEncoder(handle_unknown='ignore'), ['key_pressed']),
            ('cursor_ext', JsonNumericalExtractor('cursor', ['lineNumber', 'column']), ['cursor']),
            ('hist_num', JsonNumericalExtractor('student_history', ['past_errors', 'struggle_concepts']), ['student_history']),
            ('hist_cat', Pipeline([
                ('ext', JsonCategoricalExtractor('student_history', 'hint_style')),
                ('oh', OneHotEncoder(handle_unknown='ignore'))
            ]), ['student_history'])
        ],
        sparse_threshold=0
    )

    # We need 4 models
    models = {
        "risk": HistGradientBoostingRegressor(max_iter=100),
        "intervention": HistGradientBoostingClassifier(max_iter=100),
        "category": HistGradientBoostingClassifier(max_iter=100),
        "hint": HistGradientBoostingClassifier(max_iter=100)
    }

    # Since HistGradientBoosting doesn't support sparse natively in some versions or needs special handling,
    # and ColumnTransformer might output sparse, we'll use a dense pipeline if data size allows.
    # Given 60k rows, dense might be large. Let's see. 
    # Actually, HistGradientBoosting handles dense numpy arrays.
    
    bundle = {}
    
    # Fit preprocessor once
    print("Fitting preprocessor...")
    X_processed = preprocessor.fit_transform(X)
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()

    for name, model in models.items():
        print(f"Training {name} model...")
        target = locals()[f"y_{name}"]
        model.fit(X_processed, target)
        bundle[name] = model

    bundle["preprocessor"] = preprocessor
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(bundle, os.path.join("models", "expert_cognis_bundle.joblib"))
    print("Model bundle saved to models/expert_cognis_bundle.joblib")

if __name__ == "__main__":
    train()
