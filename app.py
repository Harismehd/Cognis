import json
import uuid
import time
import os
import psutil
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
import re

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from cognis_features import extract_features
from cognis_expert import predict_precision_diagnostic

# ----------------------------
# Path Configuration
# ----------------------------
BASE_DIR = os.path.dirname(__file__)
LOG_FILE = os.path.join(BASE_DIR, "data", "interaction_logs.jsonl")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# ----------------------------
# In-Memory Observability State
# ----------------------------
MAX_STREAM_SIZE = 50
inference_stream = deque(maxlen=MAX_STREAM_SIZE)
session_history = {} # student_id -> deque of recent predictions
stats_window = deque(maxlen=200) # last 200 requests for latency/confidence trends

# Global Counters
stats_today = {
    "total_predictions": 0,
    "interventions": 0,
    "silences": 0,
    "start_time": datetime.utcnow().isoformat() + "Z"
}

# ----------------------------
# Pydantic Schemas
# ----------------------------

class Performance(BaseModel):
    total_problems: int
    successful_runs: int
    accuracy: int
    completion_rate: int

class ConceptMastery(BaseModel):
    variables: int
    loops: int

class LearningPatterns(BaseModel):
    avg_time: int
    hints_used: int
    revision_count: int
    peak_time: Literal["morning", "afternoon", "evening", "night"]

class ErrorTypes(BaseModel):
    SyntaxError: int
    IndentationError: int
    TypeError: int

class ErrorAnalysis(BaseModel):
    error_types: ErrorTypes
    common_mistakes: List[str]
    repeated_errors: int

class DifficultyDistribution(BaseModel):
    beginner: int
    intermediate: int

class CognitiveLoad(BaseModel):
    difficulty_distribution: DifficultyDistribution
    optimal_difficulty: Literal["beginner", "intermediate"]

class LearningStyle(BaseModel):
    pace: Literal["slow", "medium", "fast"]
    confidence: int
    prefers_hints: bool
    prefers_visualizations: bool

class Engagement(BaseModel):
    days_active: int
    sessions: int
    engagement_score: int

class StruggleIndicators(BaseModel):
    struggling_concepts: List[str]
    recovery_capacity: int

class QuestionTypes(BaseModel):
    General: int

class InteractionPatterns(BaseModel):
    mentor_questions: int
    question_types: QuestionTypes

class CodeQuality(BaseModel):
    indentation_errors: int
    syntax_errors: int

class AntiVibeCoding(BaseModel):
    paste_count: int
    tab_switches: int

class Timestamps(BaseModel):
    enrollment: int
    last_active: int

class RealtimeCursor(BaseModel):
    lineNumber: int
    column: int

class RealtimeHistory(BaseModel):
    past_errors: Dict[str, int]
    struggle_concepts: List[str]
    hint_style: str
    past_errors_total: Optional[int] = 0
    hints_ignored_today: Optional[int] = 0
    current_session_duration_minutes: Optional[int] = 0

class RealtimeDiagnosis(BaseModel):
    error_type: str
    line: Optional[int]
    column: Optional[int]
    confidence: float
    evidence: str
    fix_priority: int

class RealtimeInput(BaseModel):
    student_id: str
    code: str
    cursor: RealtimeCursor
    key_pressed: str
    student_history: RealtimeHistory
    mission_context: Optional[Dict[str, Any]] = None

class RuntimeErrorRequest(BaseModel):
    student_id: str
    code: str
    error_type: str        # "KeyError", "IndexError", "TypeError" etc.
    error_message: str     # raw Python error message
    error_line: Optional[int] = None
    student_history: dict = {}

class RealtimeResponse(BaseModel):
    prediction_id: str
    error_risk: float
    should_speak: bool
    intervention_type: str
    diagnosis: List[RealtimeDiagnosis]
    message: str
    mentor_tone: str
    next_best_action: str

class OutcomeInput(BaseModel):
    prediction_id: str
    student_id: str
    outcome_type: str
    student_action: str
    time_to_fix: float
    code_after: str

class StudentProfile(BaseModel):
    student_id: str
    performance: Performance
    concept_mastery: ConceptMastery
    learning_patterns: LearningPatterns
    error_analysis: ErrorAnalysis
    cognitive_load: CognitiveLoad
    learning_style: LearningStyle
    engagement: Engagement
    struggle_indicators: StruggleIndicators
    interaction_patterns: InteractionPatterns
    code_quality: CodeQuality
    anti_vibe_coding: AntiVibeCoding
    timestamps: Timestamps

# ----------------------------
# Helper Functions
# ----------------------------

def log_event(event: Dict[str, Any]):
    """Appends a single event to the interaction log with basic safety."""
    try:
        # Use a context manager to ensure the file closes quickly
        with open(LOG_FILE, "a", encoding="utf-8", buffering=1) as f:
            f.write(json.dumps(event) + "\n")
    except Exception as e:
        # Don't let logging errors crash or hang the main flow
        print(f"TELEMETRY LOG ERROR: {e}")

def update_observability(student_id, prediction, latency):
    stats_today["total_predictions"] += 1
    if prediction["intervention_type"] == "silence":
        stats_today["silences"] += 1
    else:
        stats_today["interventions"] += 1
    
    entry = {
        "id": prediction["prediction_id"],
        "student_id": student_id,
        "timestamp": datetime.utcnow().strftime("%H:%M:%S"),
        "code_preview": prediction["message"][:30] if prediction["message"] else "...",
        "error_type": prediction.get("diagnosis", [{}])[0].get("error_type", "no_error"),
        "confidence": prediction.get("diagnosis", [{}])[0].get("confidence", 0.0),
        "intervention": prediction["intervention_type"],
        "latency": round(latency * 1000, 1)
    }
    inference_stream.appendleft(entry)
    
    if student_id not in session_history:
        session_history[student_id] = deque(maxlen=100)
    session_history[student_id].append(entry)
    
    stats_window.append({
        "latency": entry["latency"],
        "confidence": entry["confidence"],
        "intervention": entry["intervention"]
    })

def _scaffolding_to_action(scaffolding: str, struggle_risk: float) -> str:
    if scaffolding == "high_support" or struggle_risk >= 0.75:
        return "increase_scaffolding"
    if scaffolding == "blank_canvas" and struggle_risk <= 0.30:
        return "reduce_scaffolding"
    return "adjust_hints"

def _focus_concept(payload: Dict[str, Any]) -> str:
    concepts = list(payload["struggle_indicators"]["struggling_concepts"])
    if concepts: return str(concepts[0])
    return "loops" if float(payload["concept_mastery"]["loops"]) < 60 else "variables"

def _mentor_prompt(payload: Dict[str, Any], anti_vibe: bool) -> str:
    errs = payload["error_analysis"]["error_types"]
    if anti_vibe:
        return f"Student relies heavily on pasting. Adopt a strict Socratic tone."
    return f"Student error profile: SyntaxError={errs['SyntaxError']}. Use Socratic questions."

# ----------------------------
# Output schema
# ----------------------------

class CognisAnalysis(BaseModel):
    predicted_struggle_risk: float = Field(..., ge=0.0, le=1.0)
    recommended_action: str
    scaffolding_level: Literal["high_support", "moderate_hints", "blank_canvas"]
    focus_concept: str
    anti_vibe_warning: bool
    mentor_prompt_injection: str

class PredictResponse(BaseModel):
    student_id: str
    cognis_analysis: CognisAnalysis

def _bundle_path() -> str:
    default_path = os.path.join(os.path.dirname(__file__), "models", "cognis_bundle.joblib")
    return os.path.normpath(os.environ.get("COGNIS_MODEL_PATH", default_path))

def _load_bundle() -> Dict[str, Any]:
    path = _bundle_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model bundle not found at {path}.")
    return joblib.load(path)

# ----------------------------
# API Initialization
# ----------------------------

app = FastAPI(title="Cognis ML", version="0.1.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_ARTIFACT: Optional[Dict[str, Any]] = None

@app.on_event("startup")
def warm_up_models():
    global MODEL_ARTIFACT
    # 1. Load the session-level struggle model
    try:
        MODEL_ARTIFACT = _load_bundle()
        print("COGNIS: Session model loaded.")
    except Exception as e:
        print(f"COGNIS: Session model warm-up skipped: {e}")
    
    # 2. Force precision model (expert) to load
    try:
        from cognis_expert import _get_bundle
        _get_bundle()
        print("COGNIS: Precision model loaded.")
    except Exception as e:
        print(f"COGNIS: Precision model warm-up skipped: {e}")

    # 3. Dummy inference to warm up JIT/CPU caches
    try:
        predict_precision_diagnostic(
            student_id="__warmup__",
            code="print('hello')",
            cursor={"lineNumber": 1, "column": 16},
            key_pressed="Enter",
            student_history={"past_errors": {}, "struggle_concepts": [], "hint_style": "supportive"}
        )
        print("COGNIS: Inference engine warmed up.")
    except Exception as e:
        print(f"COGNIS: Dummy inference failed: {e}")

import logging

# Silence Uvicorn's access logs for the dashboard polling to keep terminal clean
class LogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Exclude dashboard polling from terminal logs
        msg = record.getMessage()
        return "/dashboard/" not in msg and "/health" not in msg

logging.getLogger("uvicorn.access").addFilter(LogFilter())

# ----------------------------
# ENDPOINTS
# ----------------------------

# Runtime error type → Cognis internal error type mapping
RUNTIME_TO_COGNIS = {
    "KeyError":          "index_error",
    "IndexError":        "index_error",
    "NameError":         "undefined_variable",
    "TypeError":         "type_confusion",
    "ImportError":       "spelling_error",
    "ModuleNotFoundError": "spelling_error",
    "AttributeError":    "undefined_variable",
    "ZeroDivisionError": "wrong_formula",
    "ValueError":        "type_confusion",
    "RecursionError":    "infinite_loop_risk",
    "StopIteration":     "wrong_loop_condition",
}

@app.post("/diagnose_runtime")
async def diagnose_runtime(request: RuntimeErrorRequest):
    try:
        error_type = RUNTIME_TO_COGNIS.get(request.error_type, "concept_gap")
        
        # Extract line number from error message if not provided
        line_number = request.error_line
        if not line_number:
            match = re.search(r'line (\d+)', request.error_message)
            line_number = int(match.group(1)) if match else 1

        # Build cursor from line number
        cursor = {"lineNumber": line_number, "column": 1}

        # Use existing _select_message and _build_evidence from cognis_expert
        from cognis_expert import _select_message, _build_evidence, _get_line_content

        message = _select_message(
            error_type,
            request.student_history,
            compile_features=None,
            code=request.code,
            cursor=cursor
        )

        line_content = _get_line_content(request.code, cursor)
        evidence = _build_evidence(error_type, line_content, {})

        # Determine intervention from student history
        total_errors = sum(
            int(v) for v in request.student_history.get("past_errors", {}).values()
            if isinstance(v, (int, float))
        )
        intervention = "warning" if total_errors >= 6 else "hint"
        hint_style = str(request.student_history.get("hint_style", "supportive")).lower()
        tone = "firm" if hint_style == "direct" else "supportive"

        return {
            "should_speak": True,
            "intervention_type": intervention,
            "error_risk": 0.90,
            "message": message,
            "mentor_tone": tone,
            "next_best_action": "fix_error",
            "diagnosis": [{
                "error_type": error_type,
                "line": line_number,
                "column": 1,
                "confidence": 0.95,
                "evidence": evidence,
                "fix_priority": 1
            }],
            "is_runtime_error": True,
            "original_python_error": f"{request.error_type}: {request.error_message}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {
        "status": "Cognis AI Online",
        "version": "1.2.0",
        "dashboard": "http://127.0.0.1:8000/dashboard",
        "message": "Welcome to the Cognis ML Observability Backend"
    }

@app.post("/predict_realtime", response_model=RealtimeResponse)
def predict_realtime(data: RealtimeInput, background_tasks: BackgroundTasks):
    """
    Real-time precision diagnostic endpoint. Using 'def' to run in threadpool
    and avoid blocking the event loop during ML inference.
    """
    start_time = time.time()
    try:
        prediction_id = str(uuid.uuid4())
        prediction = predict_precision_diagnostic(
            student_id=data.student_id,
            code=data.code,
            cursor=data.cursor.model_dump(),
            key_pressed=data.key_pressed,
            student_history=data.student_history.model_dump()
        )
        prediction["prediction_id"] = prediction_id
        
        latency = time.time() - start_time
        update_observability(data.student_id, prediction, latency)
        
        log_payload = {
            "event_type": "prediction",
            "prediction_id": prediction_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "student_id": data.student_id,
            "diagnosis": prediction.get("diagnosis", [{}])[0],
            "intervention": prediction.get("intervention_type", "silence")
        }
        background_tasks.add_task(log_event, log_payload)
        
        return RealtimeResponse(**prediction)
    except Exception as e:
        print(f"ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/log_outcome")
def log_outcome(data: OutcomeInput, background_tasks: BackgroundTasks):
    background_tasks.add_task(log_event, {
        "event_type": "outcome",
        **data.model_dump(),
        "timestamp": datetime.utcnow().isoformat() + "Z"
    })
    return {"ok": True}

@app.get("/dashboard/stats")
async def get_dashboard_stats():
    latencies = [s["latency"] for s in stats_window]
    p95 = sorted(latencies)[int(len(latencies)*0.95)] if latencies else 0
    return {
        "today": stats_today,
        "p95_latency": p95,
        "intervention_rate": round((stats_today["interventions"] / max(1, stats_today["total_predictions"])) * 100, 1),
        "active_students": len(session_history)
    }

@app.get("/dashboard/recent")
async def get_recent():
    return list(inference_stream)

@app.get("/dashboard/student/{student_id}")
async def get_student_session(student_id: str):
    return list(session_history.get(student_id, []))

@app.get("/dashboard/model-health")
async def get_model_health():
    process = psutil.Process(os.getpid())
    return {
        "version": "v1.2",
        "size_kb": 8538,
        "memory_mb": round(process.memory_info().rss / (1024 * 1024), 1),
        "uptime": str(datetime.utcnow() - datetime.fromisoformat(stats_today["start_time"].replace("Z", ""))),
        "collecting_for_retrain": 0
    }

@app.get("/dashboard", response_class=HTMLResponse)
def serve_dashboard():
    dashboard_path = os.path.join(BASE_DIR, "dashboard.html")
    if not os.path.exists(dashboard_path):
        return HTMLResponse(content="<h1>Dashboard file not found.</h1>", status_code=404)
    with open(dashboard_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/predict", response_model=PredictResponse)
def predict(profile: StudentProfile) -> PredictResponse:
    global MODEL_ARTIFACT
    if MODEL_ARTIFACT is None: MODEL_ARTIFACT = _load_bundle()
    bundle = MODEL_ARTIFACT["bundle"]
    payload = profile.model_dump()
    numeric, categorical = extract_features(payload)
    row = {k: float(numeric.get(k, 0.0)) for k in bundle["numeric_feature_names"]}
    row.update({k: str(categorical.get(k, "")) for k in bundle["categorical_feature_names"]})
    X = pd.DataFrame([row])
    struggle_risk = max(0.0, min(1.0, float(bundle["struggle_model"].predict(X)[0])))
    scaffolding_level = str(bundle["scaffolding_model"].predict(X)[0])
    anti_vibe_warning = bool(bundle["intervention_model"].predict(X)[0] == 1)
    return PredictResponse(
        student_id=profile.student_id,
        cognis_analysis=CognisAnalysis(
            predicted_struggle_risk=struggle_risk,
            recommended_action=_scaffolding_to_action(scaffolding_level, struggle_risk),
            scaffolding_level=scaffolding_level,
            focus_concept=_focus_concept(payload),
            anti_vibe_warning=anti_vibe_warning,
            mentor_prompt_injection=_mentor_prompt(payload, anti_vibe_warning),
        ),
    )

@app.get("/health")
def health():
    return {"ok": True}

