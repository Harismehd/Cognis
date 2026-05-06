from __future__ import annotations

from typing import Any, Dict, List, Tuple


SchemaStudent = Dict[str, Any]


def safe_div(n: float, d: float) -> float:
    return float(n) / float(d) if d else 0.0


def extract_features(student: SchemaStudent) -> Tuple[Dict[str, float], Dict[str, str]]:
    """
    Convert the fixed Cognis telemetry schema into flat, model-friendly features.

    Returns:
      - numeric_features: dict[str, float]
      - categorical_features: dict[str, str]
    """
    perf = student["performance"]
    mastery = student["concept_mastery"]
    lp = student["learning_patterns"]
    ea = student["error_analysis"]
    cl = student["cognitive_load"]
    ls = student["learning_style"]
    eng = student["engagement"]
    si = student["struggle_indicators"]
    ip = student["interaction_patterns"]
    cq = student["code_quality"]
    av = student["anti_vibe_coding"]
    ts = student["timestamps"]

    total_problems = float(perf["total_problems"])
    successful_runs = float(perf["successful_runs"])
    accuracy_pct = float(perf["accuracy"])
    completion_pct = float(perf["completion_rate"])

    syntax_err = float(ea["error_types"]["SyntaxError"])
    indent_err = float(ea["error_types"]["IndentationError"])
    type_err = float(ea["error_types"]["TypeError"])
    total_errors = syntax_err + indent_err + type_err

    hints_used = float(lp["hints_used"])
    revisions = float(lp["revision_count"])
    avg_time_ms = float(lp["avg_time"])

    paste_count = float(av["paste_count"])
    tab_switches = float(av["tab_switches"])
    repeated_errors = float(ea["repeated_errors"])

    beginner_cnt = float(cl["difficulty_distribution"]["beginner"])
    intermediate_cnt = float(cl["difficulty_distribution"]["intermediate"])
    diff_total = beginner_cnt + intermediate_cnt

    now_last_active = float(ts["last_active"])
    enrolled = float(ts["enrollment"])
    active_span_hours = safe_div(max(0.0, now_last_active - enrolled), 1000.0 * 60.0 * 60.0)

    struggling_concepts: List[str] = list(si["struggling_concepts"])
    loops_struggling = 1.0 if "loops" in struggling_concepts else 0.0

    numeric: Dict[str, float] = {
        # Core performance
        "total_problems": total_problems,
        "successful_runs": successful_runs,
        "accuracy_pct": accuracy_pct,
        "completion_pct": completion_pct,
        # Mastery
        "variables_mastery": float(mastery["variables"]),
        "loops_mastery": float(mastery["loops"]),
        # Time & behavior
        "avg_time_ms": avg_time_ms,
        "hints_used": hints_used,
        "revision_count": revisions,
        "mentor_questions": float(ip["mentor_questions"]),
        # Errors
        "syntax_errors": syntax_err,
        "indentation_errors": indent_err,
        "type_errors": type_err,
        "total_errors": total_errors,
        "repeated_errors": repeated_errors,
        "error_rate": safe_div(total_errors, total_problems),
        "syntax_error_rate": safe_div(syntax_err, total_problems),
        # Difficulty exposure
        "beginner_cnt": beginner_cnt,
        "intermediate_cnt": intermediate_cnt,
        "intermediate_ratio": safe_div(intermediate_cnt, diff_total),
        # Engagement
        "days_active": float(eng["days_active"]),
        "sessions": float(eng["sessions"]),
        "engagement_score": float(eng["engagement_score"]),
        # Learning style
        "confidence": float(ls["confidence"]),
        "recovery_capacity": float(si["recovery_capacity"]),
        # Anti-vibe / frustration signals (examples from prompt)
        "paste_count": paste_count,
        "tab_switches": tab_switches,
        "frustration_index": paste_count + tab_switches + repeated_errors,
        # Code quality (redundant but schema-fixed, keep both)
        "cq_indentation_errors": float(cq["indentation_errors"]),
        "cq_syntax_errors": float(cq["syntax_errors"]),
        # Struggle indicators (as a feature, not label)
        "loops_struggling": loops_struggling,
        # Time since enrollment proxy
        "active_span_hours": active_span_hours,
    }

    categorical: Dict[str, str] = {
        "peak_time": str(lp["peak_time"]),
        "optimal_difficulty": str(cl["optimal_difficulty"]),
        "pace": str(ls["pace"]),
        "prefers_hints": "true" if bool(ls["prefers_hints"]) else "false",
        "prefers_visualizations": "true" if bool(ls["prefers_visualizations"]) else "false",
    }

    return numeric, categorical

