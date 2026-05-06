import json
import re
import numpy as np
from typing import Any, Dict, List, Union
from sklearn.base import BaseEstimator, TransformerMixin

class JsonNumericalExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, column, fields):
        self.column = column
        self.fields = fields
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        rows = []
        for val in X[self.column]:
            data = json.loads(val)
            row = []
            for f in self.fields:
                val_f = data.get(f)
                if isinstance(val_f, dict):
                    row.append(float(sum(val_f.values())))
                elif isinstance(val_f, list):
                    row.append(float(len(val_f)))
                elif val_f is None:
                    row.append(0.0)
                else:
                    try:
                        row.append(float(val_f))
                    except (ValueError, TypeError):
                        row.append(0.0)
            rows.append(row)
        return np.array(rows)

class JsonCategoricalExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, column, field):
        self.column = column
        self.field = field
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.array([str(json.loads(val).get(self.field, "")) for val in X[self.column]]).reshape(-1, 1)

def _infer_typing_state(code: str) -> str:
    text = str(code or "").rstrip()
    if not text or text.endswith((".", ":", "(", "[", "{", "+", "-", "*", "/", "=")):
        return "partial"
    if "\n" in text and text.splitlines()[-1].strip() == "":
        return "partial"
    return "complete"


def _compile_error_features(code: str) -> Dict[str, float] | Dict[str, str]:
    raw_code = str(code or "")
    try:
        compile(raw_code, "<string>", "exec")
        return {
            "compile_error_type": "no_error",
            "compile_error_flag": 0.0,
            "compile_error_line": 0.0,
            "compile_error_severity": 0.0,
        }
    except IndentationError as exc:
        lines = raw_code.splitlines()
        lineno = exc.lineno or 0
        error_type = "indentation_error"
        evidence = "Python expected consistent indentation. Check: is the line above missing a colon? Are you mixing tabs and spaces?"
        
        # Check line before the error line
        if lineno > 1 and lineno - 2 < len(lines):
            prev_line = lines[lineno - 2].strip()
            if prev_line.startswith(("if", "for", "while", "def", "elif", "else")) and not prev_line.endswith(":"):
                error_type = "missing_colon"
        
        # Check for mixed tabs and spaces
        if lineno > 0 and lineno <= len(lines):
            current_line = lines[lineno - 1]
            if " " in current_line and "\t" in current_line:
                error_type = "indentation_error"

        return {
            "compile_error_type": error_type,
            "compile_error_flag": 1.0,
            "compile_error_line": float(lineno),
            "compile_error_severity": 0.95,
            "evidence": evidence,
        }
    except SyntaxError as exc:
        raw_message = str(exc).strip()
        line = float(exc.lineno or 0)
        text = (exc.text or "").strip()
        lower = raw_message.lower()
        if "expected ':'" in lower or (text and text.startswith(("if ", "for ", "while ", "def ", "elif ")) and not text.endswith(":")):
            error_type = "missing_colon"
        elif "unterminated f-string" in lower or "unterminated string" in lower or "eol while scanning string literal" in lower:
            error_type = "missing_quote"
        elif "unexpected eof while parsing" in lower or ("invalid syntax" in lower and text.count("(") > text.count(")")):
            error_type = "missing_parenthesis"
        elif "invalid syntax" in lower and text.count("[") > text.count("]"):
            error_type = "missing_bracket"
        elif "invalid syntax" in lower and re.search(r"\d\s+\d|\)\s*\(", text):
            error_type = "wrong_formula"
        elif "invalid syntax" in lower:
            error_type = "missing_colon" if text and text.startswith(("if ", "for ", "while ", "def ", "elif ")) and not text.endswith(":") else "multiple_errors"
        else:
            error_type = "multiple_errors"
        return {
            "compile_error_type": error_type,
            "compile_error_flag": 1.0,
            "compile_error_line": line,
            "compile_error_severity": 0.95,
        }


class PrecisionInputExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        import pandas as pd
        rows = []
        for val in X:
            data = json.loads(val) if isinstance(val, str) else val
            code = str(data.get("code", "") or "")
            compile_features = data.get("compile_features") or _compile_error_features(code)
            rows.append([
                code,
                data.get("key_pressed", ""),
                json.dumps(data.get("cursor", {})),
                json.dumps(data.get("student_history", {})),
                data.get("typing_state") or _infer_typing_state(code),
                compile_features.get("compile_error_type", "no_error"),
                float(compile_features.get("compile_error_flag", 0.0)),
                float(compile_features.get("compile_error_line", 0.0)),
                float(compile_features.get("compile_error_severity", 0.0)),
            ])
        return pd.DataFrame(
            rows,
            columns=[
                "code",
                "key",
                "cursor",
                "history",
                "typing_state",
                "compile_error_type",
                "compile_error_flag",
                "compile_error_line",
                "compile_error_severity",
            ],
        )
