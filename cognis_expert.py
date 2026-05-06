import pandas as pd
import numpy as np
import joblib
import json
import os
from expert_transformers import JsonNumericalExtractor, JsonCategoricalExtractor, _compile_error_features
from generate_precision_data import ERROR_MESSAGES

# Load model bundle
BUNDLE_PATH = os.path.join(os.path.dirname(__file__), "models", "precision_cognis_bundle.joblib")
_BUNDLE = None


def _get_bundle():
    global _BUNDLE
    if _BUNDLE is None:
        _BUNDLE = joblib.load(BUNDLE_PATH)
    return _BUNDLE


def _get_line_content(code: str, cursor: dict) -> str:
    """Extract the actual line the student is on, capped at 60 chars for readability."""
    if not code or not cursor:
        return ""
    lines = code.splitlines()
    line_num = cursor.get("lineNumber", 1) - 1
    if 0 <= line_num < len(lines):
        return lines[line_num].strip()[:60]
    return ""


def _build_context_message(error_type: str, line_content: str, style: str, total_errors: int) -> str:
    """
    Build a code-aware, Socratic message referencing the student's actual line.
    Every one of the 20 error types gets a specific, non-generic message.
    The hint_style and total_errors determine framing — not the base content.
    """
    line_ref = f'`{line_content}`' if line_content else "this line"

    # Code-aware base messages for all 20 error types
    bases = {
        "missing_colon": (
            f"{line_ref} starts a block — what symbol must end a block header in Python?"
        ),
        "spelling_error": (
            f"{line_ref} — does Python recognize that keyword or name? "
            "Check each character carefully."
        ),
        "missing_parenthesis": (
            f"Count every `(` and `)` on {line_ref}. Do they pair up?"
        ),
        "missing_bracket": (
            f"Count every `[` and `]` on {line_ref}. Does every opener have a closer?"
        ),
        "missing_quote": (
            f"Check the quotes on {line_ref}. "
            "Every string needs both an opening and a closing quote of the same type."
        ),
        "indentation_error": (
            f"{line_ref} — is this line's indentation consistent with the block it belongs to? "
            "Check the line above it too."
        ),
        "unexpected_indent": (
            f"{line_ref} is indented, but Python didn't expect a new block to start here. "
            "Does the line above end with a colon?"
        ),
        "undefined_variable": (
            f"You used a name on {line_ref} — was it assigned a value before this point? "
            "Check for typos too."
        ),
        "wrong_operator": (
            f"{line_ref} — is that the right operator for what you're trying to compare or calculate?"
        ),
        "comparison_vs_assignment": (
            f"Inside a condition on {line_ref}, `=` assigns a value — "
            "did you mean `==` to compare two values?"
        ),
        "type_confusion": (
            f"Look at {line_ref} — what type is on each side of the operation? "
            "Can a string and a number work together like that in Python?"
        ),
        "index_error": (
            f"On {line_ref}, check the index you're using. "
            "How many items are actually in the list? Remember Python counts from 0."
        ),
        "infinite_loop_risk": (
            f"Look at {line_ref} — what value changes inside this loop "
            "so that the condition can eventually become False?"
        ),
        "off_by_one_error": (
            f"Check the boundary on {line_ref}. "
            "Is the range running one step too many or one step too few?"
        ),
        "wrong_loop_condition": (
            f"On {line_ref} — will this condition actually stop the loop at the right moment? "
            "Trace what happens on the first and last iteration."
        ),
        "wrong_formula": (
            f"Trace {line_ref} step by step. "
            "Is the order of operations matching what you intend to calculate?"
        ),
        "concept_gap": (
            f"Step back from {line_ref}. "
            "In plain English, what should this line accomplish? "
            "Does your code actually do that?"
        ),
        "multiple_errors": (
            f"There's more than one issue starting around {line_ref}. "
            "Fix the first error completely before looking at the rest."
        ),
        "unknown_ambiguous": (
            "Keep typing — the pattern isn't clear enough yet for a specific hint. "
            "I'll step in when it is."
        ),
        "incomplete_statement": (
            f"{line_ref} — what should come after this keyword? "
            "The statement isn't complete yet."
        ),
        "no_error": (
            "Code looks good so far — keep going."
        ),
    }

    base = bases.get(error_type, f"Review {line_ref} carefully before moving on.")

    # Silent/no-error types — return immediately, no prefix
    if error_type in ("no_error", "unknown_ambiguous"):
        return base

    # Escalation for repeat struggles
    if total_errors >= 10:
        return (
            f"You've hit {total_errors} errors this session — slow down and read carefully. "
            f"{base}"
        )
    if total_errors >= 6 and style in ("direct", "firm"):
        return f"This pattern keeps coming up. {base}"

    # Hint style framing
    prefixes = {
        "socratic":   "What do you notice? ",
        "direct":     "",
        "nudge":      "Here's a clue: ",
        "gentle":     "No rush — ",
        "supportive": "Take a look: ",
        "silent":     "",
    }
    prefix = prefixes.get(style, "")
    return f"{prefix}{base}"


def _build_evidence(error_type: str, line_content: str, compile_features: dict) -> str:
    """
    Build a specific evidence string for the expanded tooltip panel.
    All 20 error types get a meaningful, non-generic explanation.
    """
    line_ref = f'`{line_content}`' if line_content else "this line"

    evidences = {
        "missing_colon": (
            f"Python requires a colon `:` at the end of every block-starting statement "
            f"(if, for, while, def, class, elif, else, try, except). "
            f"The line {line_ref} starts a block but is missing it."
        ),
        "spelling_error": (
            f"A Python keyword or built-in name appears to be misspelled on {line_ref}. "
            "Python is case-sensitive and does not auto-correct keyword typos — "
            "even one wrong character makes the name unrecognizable."
        ),
        "missing_parenthesis": (
            f"Every opening `(` must be closed with a matching `)`. "
            f"On {line_ref}, at least one parenthesis is unclosed. "
            "This causes Python to think the expression continues onto the next line."
        ),
        "missing_bracket": (
            f"Every opening `[` must be closed with a matching `]`. "
            f"On {line_ref}, a bracket is unclosed. "
            "This is common in list literals or when indexing into a list."
        ),
        "missing_quote": (
            f"A string literal on {line_ref} is not properly closed. "
            "Strings must start and end with the same quote character — "
            "either both single `'` or both double `\"`."
        ),
        "indentation_error": (
            f"Python uses indentation (spaces/tabs) to define code blocks. "
            f"The indentation of {line_ref} doesn't match the block structure above it. "
            "Check that all lines in the same block use the same number of spaces."
        ),
        "unexpected_indent": (
            f"{line_ref} is indented, but there is no block-starting statement (ending in `:`) "
            "on the line above it. Python doesn't expect a new block here."
        ),
        "undefined_variable": (
            f"A name is used on {line_ref} that hasn't been assigned a value yet in this scope. "
            "In Python, you must assign a variable before you can read it. "
            "Also check for typos — `count` and `coutn` are different names."
        ),
        "wrong_operator": (
            f"The operator used on {line_ref} may not be correct for this comparison or calculation. "
            "Common mistakes: using `=` instead of `==`, `>` instead of `>=`, "
            "or `!` instead of `!=`."
        ),
        "comparison_vs_assignment": (
            f"Inside the condition on {line_ref}, `=` is being used. "
            "In Python, `=` assigns a value to a variable — it doesn't compare two values. "
            "Use `==` to check whether two values are equal."
        ),
        "type_confusion": (
            f"On {line_ref}, an operation is being performed between incompatible types. "
            "For example, you cannot add a string and an integer directly in Python. "
            "Use `int()`, `str()`, or `float()` to convert types before combining them."
        ),
        "index_error": (
            f"On {line_ref}, the index used to access a list or string is out of range. "
            "Remember: Python lists start at index 0, so a list with 3 items has "
            "valid indices 0, 1, and 2. Negative indices count from the end."
        ),
        "infinite_loop_risk": (
            f"The loop starting near {line_ref} may run forever. "
            "For a loop to stop, the condition must eventually become False — "
            "which means some value inside the loop must change with each iteration."
        ),
        "off_by_one_error": (
            f"The range or boundary on {line_ref} is off by one. "
            "This is a very common error: `range(n)` gives 0 to n-1, not 0 to n. "
            "Check whether you need `<` or `<=`, and whether range starts at 0 or 1."
        ),
        "wrong_loop_condition": (
            f"The loop condition on {line_ref} doesn't match the intended logic. "
            "Common mistakes: using `while False` (never runs), "
            "wrong variable in condition, or `and`/`or` confusion."
        ),
        "wrong_formula": (
            f"The arithmetic expression on {line_ref} has an incorrect operation or order. "
            "Python follows standard operator precedence (PEMDAS). "
            "Use parentheses to force the order you intend: `(a + b) * c` vs `a + b * c`."
        ),
        "concept_gap": (
            f"The code on {line_ref} is syntactically valid but doesn't match the logic "
            "needed to solve the problem. This isn't a typo — it's a conceptual mismatch. "
            "Re-read the task and describe in plain English what this line needs to do."
        ),
        "multiple_errors": (
            f"There are multiple issues in this code block. "
            "Starting near {line_ref}, fix the first error completely — "
            "subsequent errors often disappear once the first is resolved."
        ),
        "unknown_ambiguous": (
            "The code pattern is ambiguous — Cognis is not confident enough to give a specific hint. "
            "Keep typing and a clearer diagnosis will appear."
        ),
        "incomplete_statement": (
            f"{line_ref} begins a statement that isn't finished yet. "
            "Python keywords like `for`, `if`, `while`, and `def` need more parts "
            "before the statement is valid."
        ),
    }

    # If compiler also caught something, add that context
    if compile_features and compile_features.get("compile_error_type") not in ("no_error", None, error_type):
        compiler_type = compile_features["compile_error_type"]
        compiler_note = f" (Python's compiler also flagged: {compiler_type.replace('_', ' ')})"
    else:
        compiler_note = ""

    return evidences.get(error_type, f"Review {line_ref} carefully.") + compiler_note


def _select_message(error_type: str, student_history: dict, compile_features: dict = None,
                    code: str = "", cursor: dict = None) -> str:
    """
    Build the primary tooltip message. Now code-aware and specific for all 20 error types.
    The raw_message / msg_map path is bypassed — we build directly from code context.
    """
    style = str(student_history.get("hint_style", "supportive")).lower()
    total_errors = sum(
        int(v) for v in student_history.get("past_errors", {}).values()
        if isinstance(v, (int, float))
    )
    line_content = _get_line_content(code, cursor)

    # Handle multiple_errors specially — use compiler's specific finding if available
    if error_type == "multiple_errors" and compile_features:
        comp_err = compile_features.get("compile_error_type", "no_error")
        if comp_err not in ("no_error", "multiple_errors", None):
            # Delegate to the specific error type the compiler found
            return _build_context_message(comp_err, line_content, style, total_errors)

    return _build_context_message(error_type, line_content, style, total_errors)


def _adjust_intervention_and_tone(intervention: str, tone: str, student_history: dict,
                                   error_type: str, confidence: float) -> tuple[str, str]:
    hint_style = str(student_history.get("hint_style", "supportive")).lower()
    total_errors = sum(
        int(v) for v in student_history.get("past_errors", {}).values()
        if isinstance(v, (int, float))
    )

    if error_type == "no_error":
        return "silence", "silent"

    if total_errors >= 10:
        return "warning", "firm"
    if total_errors >= 6 and hint_style in ("direct", "firm"):
        return "warning", "firm"
    if hint_style == "nudge":
        return "nudge", "encouraging"
    if hint_style == "direct":
        return "warning", "firm"
    if hint_style == "socratic":
        return "hint", "supportive"
    if hint_style == "gentle":
        return "hint", "gentle"
    if hint_style == "supportive":
        return "hint", "supportive"

    return intervention, tone


def predict_precision_diagnostic(student_id, code, cursor, key_pressed, student_history):
    bundle = _get_bundle()
    preprocessor = bundle["preprocessor"]
    models = bundle["models"]

    compile_features = _compile_error_features(code or "")
    X_input = [{
        "code": code or "",
        "key_pressed": key_pressed,
        "cursor": cursor,
        "student_history": student_history,
        "compile_features": compile_features,
    }]

    input_expander = bundle["input_expander"]
    X_expanded = input_expander.transform(X_input)
    X_processed = preprocessor.transform(X_expanded)
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()

    # Model predictions
    risk_score = float(models["risk"].predict(X_processed)[0])
    risk_score = max(0.0, min(1.0, risk_score))

    should_speak = bool(models["speak"].predict(X_processed)[0] == 1)
    intervention = str(models["intervention"].predict(X_processed)[0])
    error_type = str(models["error_type"].predict(X_processed)[0])
    tone = str(models["tone"].predict(X_processed)[0])
    action = str(models["action"].predict(X_processed)[0])

    # Confidence from error_type model
    probs = models["error_type"].predict_proba(X_processed)[0]
    confidence = float(np.max(probs))

    # Build code-aware message and evidence — bypasses static msg_map entirely
    message = _select_message(error_type, student_history, compile_features, code=code, cursor=cursor)
    intervention, tone = _adjust_intervention_and_tone(intervention, tone, student_history, error_type, confidence)

    # Compiler rescue: if ML confidence is low but compiler is certain, trust compiler
    if confidence < 0.50 and compile_features.get("compile_error_type") not in ("no_error", None):
        error_type = compile_features["compile_error_type"]
        message = _select_message(error_type, student_history, compile_features, code=code, cursor=cursor)
        should_speak = True
        intervention = "hint"
        tone = "supportive"
        action = "fix_error"
        confidence = float(compile_features.get("compile_error_severity", 0.95))

    # Build evidence string — specific for all 20 error types
    line_content = _get_line_content(code, cursor)

    diagnosis = []
    if error_type != "no_error":
        diagnosis.append({
            "error_type": error_type,
            "line": cursor.get("lineNumber"),
            "column": cursor.get("column"),
            "confidence": round(confidence, 2),
            "evidence": _build_evidence(error_type, line_content, compile_features),
            "fix_priority": 1
        })
    else:
        diagnosis.append({
            "error_type": "no_error",
            "line": None,
            "column": None,
            "confidence": round(confidence, 2),
            "evidence": "Code looks good — keep going.",
            "fix_priority": 0
        })

    # Low confidence and no compiler rescue → stay silent
    if confidence < 0.50 and error_type != "no_error":
        diagnosis[0]["error_type"] = "unknown_ambiguous"
        message = ""
        should_speak = False
        intervention = "silence"

    return {
        "error_risk": round(risk_score, 2),
        "should_speak": should_speak,
        "intervention_type": intervention,
        "diagnosis": diagnosis,
        "message": message,
        "mentor_tone": tone,
        "next_best_action": action
    }