import argparse
import json
import os
import random
import string
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


SchemaStudent = Dict[str, Any]


PERSONA_NAMES: Tuple[str, ...] = (
    "fast_guesser",
    "slow_steady",
    "copy_paster",
    "hint_dependent",
    "confident_builder",
    "struggler",
    "night_owl",
)

PEAK_TIMES: Tuple[str, ...] = ("morning", "afternoon", "evening", "night")

ERROR_TYPES: Tuple[str, ...] = ("SyntaxError", "IndentationError", "TypeError")
COMMON_MISTAKES: Tuple[str, ...] = (
    "missing_colon",
    "type_mismatch",
    "indentation_level",
    "unclosed_parenthesis",
    "name_error_variable",
)


@dataclass(frozen=True)
class PersonaConfig:
    pace: str
    prefers_hints: bool
    prefers_visualizations: bool
    avg_time_ms_range: Tuple[int, int]
    hints_used_range: Tuple[int, int]
    revision_count_range: Tuple[int, int]
    paste_count_range: Tuple[int, int]
    tab_switches_range: Tuple[int, int]
    repeated_errors_range: Tuple[int, int]
    sessions_range: Tuple[int, int]
    days_active_range: Tuple[int, int]
    mentor_questions_range: Tuple[int, int]
    confidence_range: Tuple[int, int]
    variables_mastery_range: Tuple[int, int]
    loops_mastery_range: Tuple[int, int]
    total_problems_range: Tuple[int, int]


PERSONAS: Dict[str, PersonaConfig] = {
    # High speed, lower care: quick attempts, middling accuracy, higher revisions.
    "fast_guesser": PersonaConfig(
        pace="fast",
        prefers_hints=False,
        prefers_visualizations=False,
        avg_time_ms_range=(10_000, 45_000),
        hints_used_range=(0, 6),
        revision_count_range=(10, 40),
        paste_count_range=(0, 2),
        tab_switches_range=(3, 20),
        repeated_errors_range=(1, 8),
        sessions_range=(6, 20),
        days_active_range=(3, 14),
        mentor_questions_range=(1, 8),
        confidence_range=(45, 80),
        variables_mastery_range=(50, 95),
        loops_mastery_range=(20, 70),
        total_problems_range=(20, 80),
    ),
    # Slow but consistent: longer time, good completion, low anti-vibe.
    "slow_steady": PersonaConfig(
        pace="slow",
        prefers_hints=True,
        prefers_visualizations=True,
        avg_time_ms_range=(40_000, 120_000),
        hints_used_range=(4, 18),
        revision_count_range=(8, 30),
        paste_count_range=(0, 1),
        tab_switches_range=(0, 8),
        repeated_errors_range=(0, 4),
        sessions_range=(4, 14),
        days_active_range=(5, 20),
        mentor_questions_range=(2, 12),
        confidence_range=(35, 70),
        variables_mastery_range=(60, 95),
        loops_mastery_range=(30, 80),
        total_problems_range=(15, 70),
    ),
    # Copies a lot: high paste & tab switching, higher repeated errors.
    "copy_paster": PersonaConfig(
        pace="fast",
        prefers_hints=False,
        prefers_visualizations=False,
        avg_time_ms_range=(8_000, 55_000),
        hints_used_range=(0, 8),
        revision_count_range=(5, 25),
        paste_count_range=(4, 20),
        tab_switches_range=(12, 60),
        repeated_errors_range=(4, 16),
        sessions_range=(3, 18),
        days_active_range=(2, 12),
        mentor_questions_range=(0, 6),
        confidence_range=(20, 60),
        variables_mastery_range=(40, 90),
        loops_mastery_range=(10, 60),
        total_problems_range=(10, 65),
    ),
    # Likes hints: uses many hints, lower confidence, decent engagement.
    "hint_dependent": PersonaConfig(
        pace="medium",
        prefers_hints=True,
        prefers_visualizations=False,
        avg_time_ms_range=(25_000, 80_000),
        hints_used_range=(10, 35),
        revision_count_range=(10, 35),
        paste_count_range=(0, 4),
        tab_switches_range=(2, 18),
        repeated_errors_range=(2, 10),
        sessions_range=(6, 22),
        days_active_range=(4, 18),
        mentor_questions_range=(4, 18),
        confidence_range=(20, 55),
        variables_mastery_range=(55, 92),
        loops_mastery_range=(20, 75),
        total_problems_range=(20, 90),
    ),
    # Confident: fewer errors, higher mastery, moderate time.
    "confident_builder": PersonaConfig(
        pace="medium",
        prefers_hints=False,
        prefers_visualizations=True,
        avg_time_ms_range=(18_000, 60_000),
        hints_used_range=(0, 6),
        revision_count_range=(6, 25),
        paste_count_range=(0, 2),
        tab_switches_range=(0, 10),
        repeated_errors_range=(0, 4),
        sessions_range=(6, 24),
        days_active_range=(6, 25),
        mentor_questions_range=(1, 8),
        confidence_range=(60, 95),
        variables_mastery_range=(75, 100),
        loops_mastery_range=(60, 98),
        total_problems_range=(30, 120),
    ),
    # Struggles: low mastery, more errors, low confidence.
    "struggler": PersonaConfig(
        pace="slow",
        prefers_hints=True,
        prefers_visualizations=False,
        avg_time_ms_range=(50_000, 150_000),
        hints_used_range=(8, 40),
        revision_count_range=(15, 60),
        paste_count_range=(1, 10),
        tab_switches_range=(6, 40),
        repeated_errors_range=(6, 25),
        sessions_range=(3, 16),
        days_active_range=(2, 14),
        mentor_questions_range=(4, 22),
        confidence_range=(5, 45),
        variables_mastery_range=(10, 70),
        loops_mastery_range=(0, 45),
        total_problems_range=(10, 70),
    ),
    # Night owl: same as medium but peak_time biased to night.
    "night_owl": PersonaConfig(
        pace="medium",
        prefers_hints=True,
        prefers_visualizations=True,
        avg_time_ms_range=(25_000, 90_000),
        hints_used_range=(4, 20),
        revision_count_range=(8, 35),
        paste_count_range=(0, 6),
        tab_switches_range=(2, 25),
        repeated_errors_range=(1, 12),
        sessions_range=(5, 18),
        days_active_range=(4, 18),
        mentor_questions_range=(2, 14),
        confidence_range=(25, 70),
        variables_mastery_range=(50, 95),
        loops_mastery_range=(25, 85),
        total_problems_range=(20, 90),
    ),
}


def _clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(x)))


def _rand_int(rng: random.Random, lo: int, hi: int) -> int:
    if lo > hi:
        lo, hi = hi, lo
    return rng.randint(lo, hi)


def _pick_common_mistakes(rng: random.Random, k_range: Tuple[int, int]) -> List[str]:
    k = _rand_int(rng, k_range[0], k_range[1])
    k = _clamp_int(k, 1, min(3, len(COMMON_MISTAKES)))
    return rng.sample(list(COMMON_MISTAKES), k=k)


def _student_id(prefix: str, idx: int) -> str:
    return f"{prefix}_{idx:03d}"


def _prefix_for_idx(rng: random.Random) -> str:
    letters = rng.choice(string.ascii_uppercase) + rng.choice(string.ascii_uppercase) + rng.choice(string.ascii_uppercase)
    return letters


def _weighted_peak_time(rng: random.Random, persona: str) -> str:
    if persona == "night_owl":
        # Bias heavily toward night/evening, still allow others.
        return rng.choices(
            population=list(PEAK_TIMES),
            weights=[1, 2, 4, 7],
            k=1,
        )[0]
    return rng.choice(list(PEAK_TIMES))


def _difficulty_distribution(
    rng: random.Random, total_problems: int, loops_mastery: int, persona: str
) -> Tuple[Dict[str, int], str]:
    # Keep schema fixed: beginner + intermediate counts.
    if total_problems <= 0:
        return {"beginner": 0, "intermediate": 0}, "beginner"

    # More intermediate if higher loops mastery and confident persona.
    base_intermediate_ratio = 0.2 + (loops_mastery / 100.0) * 0.55
    if persona in ("confident_builder", "fast_guesser"):
        base_intermediate_ratio += 0.10
    if persona in ("struggler", "slow_steady"):
        base_intermediate_ratio -= 0.10

    base_intermediate_ratio = max(0.05, min(0.85, base_intermediate_ratio))
    intermediate = _clamp_int(int(round(total_problems * base_intermediate_ratio)), 0, total_problems)
    beginner = total_problems - intermediate
    optimal = "intermediate" if intermediate >= beginner else "beginner"
    return {"beginner": beginner, "intermediate": intermediate}, optimal


def _error_counts(
    rng: random.Random, total_problems: int, persona: str, variables_mastery: int, loops_mastery: int
) -> Dict[str, int]:
    # Generate error counts that correlate with lower mastery & certain personas.
    mastery = (variables_mastery + loops_mastery) / 2.0
    base_error_rate = 0.55 - (mastery / 100.0) * 0.45  # roughly 0.10..0.55
    if persona == "copy_paster":
        base_error_rate += 0.10
    if persona == "confident_builder":
        base_error_rate -= 0.08
    if persona == "struggler":
        base_error_rate += 0.12

    base_error_rate = max(0.03, min(0.80, base_error_rate))
    total_errors = _clamp_int(int(round(total_problems * base_error_rate)), 0, max(0, total_problems * 2))

    # Split across fixed error types.
    # SyntaxError tends to dominate; IndentationError rises for strugglers.
    weights = [6, 2, 3]
    if persona == "struggler":
        weights = [5, 4, 3]
    if persona == "confident_builder":
        weights = [5, 1, 2]
    parts = rng.choices(population=[0, 1, 2], weights=weights, k=total_errors) if total_errors > 0 else []
    counts = {k: 0 for k in ERROR_TYPES}
    for p in parts:
        counts[ERROR_TYPES[p]] += 1
    return counts


def _derive_perf(
    rng: random.Random,
    total_problems: int,
    error_types: Dict[str, int],
    persona: str,
    avg_time_ms: int,
    hints_used: int,
    revisions: int,
) -> Dict[str, int]:
    total_errors = sum(int(v) for v in error_types.values())

    # Successful runs correlate with fewer errors, moderate hint use, and not-too-high revisions.
    # Keep within [0, total_problems].
    hint_bonus = 0.08 if persona in ("hint_dependent", "slow_steady", "struggler") else 0.04
    hint_factor = min(0.12, hints_used / max(1, total_problems) * hint_bonus)
    revision_penalty = min(0.20, revisions / max(1, total_problems) * 0.10)

    # Time factor: extremely fast can be sloppy; extremely slow can be careful but may time out.
    if avg_time_ms < 15_000:
        time_factor = -0.07
    elif avg_time_ms > 120_000:
        time_factor = -0.04
    else:
        time_factor = 0.03

    base_success_rate = 0.75 - (total_errors / max(1, total_problems)) * 0.35 + hint_factor - revision_penalty + time_factor
    if persona == "confident_builder":
        base_success_rate += 0.10
    if persona == "copy_paster":
        base_success_rate -= 0.08
    if persona == "struggler":
        base_success_rate -= 0.18

    base_success_rate = max(0.02, min(0.98, base_success_rate))
    successful_runs = _clamp_int(int(round(total_problems * base_success_rate)), 0, total_problems)

    # Accuracy and completion_rate in [0,100] ints.
    accuracy = _clamp_int(int(round((successful_runs / max(1, total_problems)) * 100)), 0, 100)

    # Completion rate: uses engagement proxies + persona bias.
    # (Still fixed keys as required.)
    completion_base = accuracy + (5 if persona in ("slow_steady", "confident_builder") else 0) - (8 if persona == "copy_paster" else 0)
    completion_rate = _clamp_int(int(round(completion_base + rng.uniform(-10, 10))), 0, 100)

    return {
        "total_problems": int(total_problems),
        "successful_runs": int(successful_runs),
        "accuracy": int(accuracy),
        "completion_rate": int(completion_rate),
    }


def generate_student(rng: random.Random, idx: int, persona: str) -> SchemaStudent:
    if persona not in PERSONAS:
        raise ValueError(f"Unknown persona: {persona}")
    p = PERSONAS[persona]

    total_problems = _rand_int(rng, *p.total_problems_range)
    variables_mastery = _rand_int(rng, *p.variables_mastery_range)
    loops_mastery = _rand_int(rng, *p.loops_mastery_range)

    avg_time = _rand_int(rng, *p.avg_time_ms_range)
    hints_used = _rand_int(rng, *p.hints_used_range)
    revision_count = _rand_int(rng, *p.revision_count_range)

    error_types = _error_counts(rng, total_problems, persona, variables_mastery, loops_mastery)
    repeated_errors = _rand_int(rng, *p.repeated_errors_range)
    common_mistakes = _pick_common_mistakes(rng, (1, 3))

    difficulty_distribution, optimal_difficulty = _difficulty_distribution(rng, total_problems, loops_mastery, persona)

    paste_count = _rand_int(rng, *p.paste_count_range)
    tab_switches = _rand_int(rng, *p.tab_switches_range)

    sessions = _rand_int(rng, *p.sessions_range)
    days_active = _rand_int(rng, *p.days_active_range)
    engagement_score = _clamp_int(int(round((sessions * 6 + days_active * 4) + rng.uniform(-12, 12))), 0, 100)

    mentor_questions = _rand_int(rng, *p.mentor_questions_range)

    confidence = _rand_int(rng, *p.confidence_range)

    peak_time = _weighted_peak_time(rng, persona)

    # Focus struggle concept: if loops mastery low, mark loops; else sometimes none or loops.
    struggling_concepts = ["loops"] if loops_mastery < 55 or persona in ("struggler",) else (["loops"] if rng.random() < 0.35 else [])
    recovery_capacity = _clamp_int(int(round(confidence + rng.uniform(-15, 15))), 0, 100)

    perf = _derive_perf(
        rng=rng,
        total_problems=total_problems,
        error_types=error_types,
        persona=persona,
        avg_time_ms=avg_time,
        hints_used=hints_used,
        revisions=revision_count,
    )

    # Ensure code_quality matches error counts (schema requires these exact keys).
    indentation_errors = int(error_types.get("IndentationError", 0))
    syntax_errors = int(error_types.get("SyntaxError", 0))

    now_ms = int(time.time() * 1000)
    enrollment = now_ms - _rand_int(rng, 3, 120) * 24 * 60 * 60 * 1000  # 3..120 days ago
    last_active = now_ms - _rand_int(rng, 0, 72) * 60 * 60 * 1000  # 0..72 hours ago
    if last_active < enrollment:
        last_active = enrollment + _rand_int(rng, 1, 72) * 60 * 60 * 1000

    # Student id format: AAA_001 etc, fixed string key.
    prefix = _prefix_for_idx(rng)
    student: SchemaStudent = {
        "student_id": _student_id(prefix, idx),
        "performance": {
            "total_problems": int(perf["total_problems"]),
            "successful_runs": int(perf["successful_runs"]),
            "accuracy": int(perf["accuracy"]),
            "completion_rate": int(perf["completion_rate"]),
        },
        "concept_mastery": {"variables": int(variables_mastery), "loops": int(loops_mastery)},
        "learning_patterns": {
            "avg_time": int(avg_time),
            "hints_used": int(hints_used),
            "revision_count": int(revision_count),
            "peak_time": str(peak_time),
        },
        "error_analysis": {
            "error_types": {
                "SyntaxError": int(error_types.get("SyntaxError", 0)),
                "IndentationError": int(error_types.get("IndentationError", 0)),
                "TypeError": int(error_types.get("TypeError", 0)),
            },
            "common_mistakes": list(common_mistakes),
            "repeated_errors": int(repeated_errors),
        },
        "cognitive_load": {
            "difficulty_distribution": {
                "beginner": int(difficulty_distribution["beginner"]),
                "intermediate": int(difficulty_distribution["intermediate"]),
            },
            "optimal_difficulty": str(optimal_difficulty),
        },
        "learning_style": {
            "pace": str(p.pace),
            "confidence": int(confidence),
            "prefers_hints": bool(p.prefers_hints),
            "prefers_visualizations": bool(p.prefers_visualizations),
        },
        "engagement": {"days_active": int(days_active), "sessions": int(sessions), "engagement_score": int(engagement_score)},
        "struggle_indicators": {"struggling_concepts": list(struggling_concepts), "recovery_capacity": int(recovery_capacity)},
        "interaction_patterns": {"mentor_questions": int(mentor_questions), "question_types": {"General": int(mentor_questions)}},
        "code_quality": {"indentation_errors": int(indentation_errors), "syntax_errors": int(syntax_errors)},
        "anti_vibe_coding": {"paste_count": int(paste_count), "tab_switches": int(tab_switches)},
        "timestamps": {"enrollment": int(enrollment), "last_active": int(last_active)},
    }
    return student


def _persona_for_idx(rng: random.Random) -> str:
    # Slightly favor common archetypes, but keep variety.
    return rng.choices(
        population=list(PERSONA_NAMES),
        weights=[14, 14, 10, 12, 12, 10, 8],
        k=1,
    )[0]


def generate_dataset(n_students: int, seed: int) -> List[SchemaStudent]:
    rng = random.Random(seed)
    students: List[SchemaStudent] = []
    for i in range(1, n_students + 1):
        persona = _persona_for_idx(rng)
        students.append(generate_student(rng, idx=i, persona=persona))
    return students


def _ensure_dir(path: str) -> None:
    if not path:
        return
    os.makedirs(path, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic Cognis student telemetry dataset (schema-fixed).")
    parser.add_argument("--n", type=int, default=1200, help="Number of students to generate (default: 1200).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42).")
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join("data", "students.jsonl"),
        help="Output JSONL path (default: data/students.jsonl).",
    )
    parser.add_argument(
        "--per-student-dir",
        type=str,
        default="",
        help="Optional directory to also write individual student JSON files (disabled by default).",
    )
    args = parser.parse_args()

    n_students = int(args.n)
    if n_students < 1000:
        n_students = 1000

    students = generate_dataset(n_students=n_students, seed=int(args.seed))

    out_path = os.path.normpath(args.out)
    _ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        for s in students:
            f.write(json.dumps(s, ensure_ascii=False))
            f.write("\n")

    if args.per_student_dir:
        per_dir = os.path.normpath(args.per_student_dir)
        _ensure_dir(per_dir)
        for s in students:
            sid = s["student_id"]
            with open(os.path.join(per_dir, f"{sid}.json"), "w", encoding="utf-8") as f:
                json.dump(s, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(students)} students to {out_path}")
    if args.per_student_dir:
        print(f"Also wrote per-student JSON files to {os.path.normpath(args.per_student_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
