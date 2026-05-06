# COGNIS — Complete Technical Documentation
### Cognis Precision Diagnostic Engine v1.2 | Logic Lens Mentoring System
**Last Updated:** 2026-05-02 | **Status:** Production (Active)

---

## Table of Contents
1. [What Is Cognis?](#1-what-is-cognis)
2. [Why Multiple Model Files?](#2-why-multiple-model-files)
3. [The 6-Model Precision Ensemble](#3-the-6-model-precision-ensemble)
4. [Training Pipeline — How Cognis Learns](#4-training-pipeline--how-cognis-learns)
5. [The Confidence Threshold System](#5-the-confidence-threshold-system)
6. [Every File Explained](#6-every-file-explained)
7. [Logic Lens Integration — The Handshake](#7-logic-lens-integration--the-handshake)
8. [Self-Learning Loop (Autonomous Improvement)](#8-self-learning-loop-autonomous-improvement)
9. [Live Observability Dashboard](#9-live-observability-dashboard)
10. [Current System Size & Performance](#10-current-system-size--performance)
11. [Today's Full Work Log](#11-todays-full-work-log)

---

## 1. What Is Cognis?

Cognis is a **keystroke-level, real-time diagnostic AI** built specifically for the Logic Lens Python IDE. Its job is to watch a student type code and decide — within 12ms — whether an error is forming, what type of error it is, how confident it is about that diagnosis, and how to communicate it in a way that guides the student without giving away the answer.

It is **not a linter**. Linters run after you finish writing. Cognis runs on every single keystroke, in the background, analysing partial code to catch errors before the student even realises they are making them.

**Core design philosophy:**  
- Identify the *root cause* of errors, not just their surface symptom  
- Speak in a Socratic, personalised voice tailored to each student's history  
- Stay silent when not confident — a wrong hint is worse than no hint  
- Never block the typing experience; all inference must complete in under 50ms  

---

## 2. Why Multiple Model Files?

There are **3 model bundle files** in the `models/` directory. Each one represents a different generation of Cognis:

| File | Size | Purpose | Status |
|---|---|---|---|
| `cognis_bundle.joblib` | 1.04 MB | Generation 1: Session-level struggle predictor. Predicts scaffolding level and anti-vibe risk from a student's full session profile. | **Active** (used by `/predict`) |
| `expert_cognis_bundle.joblib` | 2.39 MB | Generation 2: Pattern-based expert rules. Retired intermediate version. | **Retired** (kept for rollback) |
| `precision_cognis_bundle.joblib` | 8.76 MB | Generation 3: Full 6-model precision ensemble. The current production brain of Cognis. | **Active** (used by `/predict_realtime`) |

> The `precision_cognis_bundle.joblib` is the only one that matters for the keystroke pipeline. The `cognis_bundle.joblib` is still used by the student profile dashboard endpoint.

---

## 3. The 6-Model Precision Ensemble

When Logic Lens sends a keystroke to Cognis, the `precision_cognis_bundle.joblib` runs **six separate models in sequence**, each answering a different question:

```
Student types a character
         │
         ▼
[Input Expander] ── Extracts 9 feature columns from raw code + cursor + history
         │
         ▼
[Preprocessor] ── Character N-gram vectorizer (1–3 grams, 2500 features) + OHE + numeric passthrough
         │
         ├──► [Model 1: RISK]         → "How likely is this code to have an error?" (0.0–1.0 float)
         ├──► [Model 2: SPEAK]        → "Should Cognis say anything right now?" (yes/no)
         ├──► [Model 3: INTERVENTION] → "What type of intervention?" (silence/nudge/hint/warning)
         ├──► [Model 4: ERROR_TYPE]   → "Which of the 20 error categories is this?" (classification)
         ├──► [Model 5: TONE]         → "What mentor tone is appropriate?" (supportive/firm/gentle/encouraging)
         └──► [Model 6: ACTION]       → "What should the student do next?" (fix_error/review/continue)
```

### Model Details

| # | Model | Algorithm | Task | Key Config |
|---|---|---|---|---|
| 1 | `risk` | `HistGradientBoostingRegressor` | Predicts error probability (0–1) | `max_depth=6`, `max_iter=140`, `lr=0.07` |
| 2 | `speak` | `HistGradientBoostingClassifier` | Binary: intervene or stay silent | `max_depth=4`, `max_iter=120`, `lr=0.08` |
| 3 | `intervention` | `HistGradientBoostingClassifier` | 4-class: silence/nudge/hint/warning | `max_depth=4`, `max_iter=120`, `lr=0.08` |
| 4 | `error_type` | `HistGradientBoostingClassifier` | 20-class: identifies the specific error | `max_depth=8`, `max_iter=250`, `lr=0.08` — deepest, most complex |
| 5 | `tone` | `HistGradientBoostingClassifier` | 4-class: chooses mentor personality | `max_depth=4`, `max_iter=120`, `lr=0.08` |
| 6 | `action` | `HistGradientBoostingClassifier` | Recommends next student action | `max_depth=4`, `max_iter=120`, `lr=0.08` |

### Why HistGradientBoosting?

`HistGradientBoosting` is scikit-learn's most powerful tree-based classifier, based on the same algorithm as LightGBM. For this task it was chosen because:

- **Handles mixed data**: Works on sparse character n-gram vectors alongside numerical cursor data and categorical history fields without needing careful normalisation
- **Speed**: Uses histogram-based binning (128 bins) for fast training and sub-millisecond inference
- **Robustness**: Early stopping prevents overfitting on small error types like `unknown_ambiguous`
- **No null issues**: Natively handles missing values without imputation

---

## 4. Training Pipeline — How Cognis Learns

### Step 1: Data Generation (`generate_precision_data.py`)

Cognis does **not** learn from real student data (yet). It was trained on **82,900 synthetic examples** generated by a carefully designed Python script. This script creates realistic, student-like code snippets for all 20 error categories.

**20 Error Categories:**
```
no_error            spelling_error       missing_colon        missing_parenthesis
missing_bracket     missing_quote        indentation_error    unexpected_indent
undefined_variable  wrong_operator       comparison_vs_assign type_confusion
index_error         infinite_loop_risk   off_by_one_error     wrong_loop_condition
wrong_formula       concept_gap          multiple_errors      unknown_ambiguous
```

**How synthetic data is built:**
Each example is a JSON record with two keys:
- `INPUT`: The raw keystroke context — `{ code, key_pressed, cursor, student_history }`
- `TARGET`: What a perfect tutor would respond — `{ error_risk, should_speak, intervention_type, diagnosis, message, mentor_tone, next_best_action }`

**Key data quality decisions:**
- `spelling_error` has 55+ distinct typo variants (whille, pritn, defin, retrun…) so the model sees real human typing patterns
- `type_confusion` has 10 distinct real scenarios (input() + number, len() on int, loop over a number, etc.) not just `"5" + 5`
- `infinite_loop_risk` only flags *unintentional* infinite loops — `while True: print(x)` with no break is **not** flagged because it's a valid pattern
- `no_error` templates include deliberate `while True` loops, correct type conversions, and f-strings to prevent false positives
- Student history snapshots include 4 archetypes: `socratic`, `direct`, `gentle`, `nudge` — so the model learns personalisation

**Minimum count guardrails:**
```python
ERROR_MIN_COUNTS = {
    "no_error":              3000,
    "spelling_error":        2500,  # Increased today
    "type_confusion":        2000,  # Increased today
    "missing_colon":         1000,
    ...
}
```

### Step 2: Model Training (`train_precision_model.py`)

```
Load 82,900 JSONL examples
        │
        ▼
Sample 50,000 (stratified by error_type for balance)
        │
        ▼
PrecisionInputExtractor.transform() → 9-column DataFrame
        │
        ▼
ColumnTransformer (Preprocessor):
  ├── code        → CountVectorizer(char n-grams 1–3, 2500 features)
  ├── key         → OneHotEncoder
  ├── cursor      → JsonNumericalExtractor [lineNumber, column]
  ├── history     → JsonNumericalExtractor [past_errors, struggle_concepts]
  ├── history     → hint_style → OneHotEncoder
  ├── typing_state → OneHotEncoder
  ├── compile_type → OneHotEncoder
  └── compile_numeric → passthrough [flag, line, severity]
        │
        ▼
85% train / 15% test split (stratified)
        │
        ▼
Train 6 models independently on same feature matrix
        │
        ▼
Evaluate on holdout: per-type precision/recall/F1
        │
        ▼
Save bundle: preprocessor + input_expander + {models} → precision_cognis_bundle.joblib
```

### What the Character N-gram Vectorizer does

Instead of word-level tokens, Cognis looks at **character sequences 1–3 characters long**. This means:
- `"whille"` contains `"whi"`, `"hil"`, `"ill"`, `"lle"` — none of which appear in `"while"`, so the model immediately detects a typo
- `"if x >"` generates character patterns that differ from `"if x ="` — enabling `comparison_vs_assignment` detection
- Partial code like `"for i in range(5"` has a signature that the model recognises as `missing_parenthesis`

---

## 5. The Confidence Threshold System

The confidence logic in `cognis_expert.py` works in three layers:

```
ML Model produces confidence score
           │
     ┌─────┴─────┐
  ≥ 0.50       < 0.50
     │             │
  Trust ML    Check Python compiler
  output      (compile_features)
     │             │
     │         Has compile error?
     │         ┌────┴────┐
     │       Yes        No
     │         │         │
     │    Use compiler   Silence
     │    error type     model
     │    confidence=0.95
     │
  Final check:
  Is error_type == "no_error"?
  ├── Yes → Return silence, risk=0.x
  └── No  → Build Socratic message and return diagnosis
```

**Why 0.50 (changed from 0.60 today):**
The previous threshold of 0.60 created a conflict: the compiler fallback always sets confidence to 0.95, so the silence check at 0.60 would never fire correctly. Setting both to 0.50 creates a clean gap — the ML model is trusted above 50%, the compiler rescues below 50%, and silence is the default when neither is certain.

---

## 6. Every File Explained

### Core Runtime Files

| File | Role |
|---|---|
| `app.py` | FastAPI server. Serves all HTTP endpoints. Manages startup warm-up, in-memory observability state (deques), session history, and telemetry logging. Entry point for Logic Lens requests. |
| `cognis_expert.py` | The inference brain. Loads the precision bundle, runs the 6 models, applies confidence logic, selects personalised messages, and returns the structured diagnosis JSON. |
| `expert_transformers.py` | Custom sklearn transformers. `PrecisionInputExtractor` expands raw keystroke dicts into a 9-column DataFrame. `_compile_error_features` runs Python's built-in compiler on the code snippet to get a "ground truth" syntax check. |
| `cognis_features.py` | Feature extractor for the session-level `/predict` endpoint (Generation 1 model). Separate from the precision pipeline. |
| `dashboard.html` | The standalone monitoring dashboard. Served at `http://127.0.0.1:8000/dashboard`. Polls the backend every 2 seconds. Shows live inference feed, stat cards, model health, and alerts. |

### Training Files

| File | Role |
|---|---|
| `generate_precision_data.py` | Generates the 82,900 synthetic training examples. Defines all 20 error categories, spelling variants, type confusion scenarios, valid no-error codes, and student archetypes. Main source of truth for what Cognis "knows". |
| `train_precision_model.py` | Trains the 6-model precision ensemble. Handles preprocessing, 85/15 stratified split, HistGradientBoosting configuration, holdout evaluation, and saves the final bundle. |
| `train_model.py` | Trains Generation 1 (session-level struggle predictor). Legacy but still active for the student dashboard. |
| `train_expert_model.py` | Trains Generation 2 (expert pattern model). Retired, kept for reference. |
| `evaluate_precision_model.py` | Standalone evaluation script. Loads the precision bundle and runs the full classification report per error type. |

### Data Generation (Legacy/Supplementary)

| File | Role |
|---|---|
| `data_generator.py` | Original synthetic data generator for Generation 1 model. Still used for the session-level bundle. |
| `generate_expert_data.py` | Data generator for Generation 2. Retired. |
| `generate_misconceptions.py` | Generates data specifically for conceptual misconceptions (concept_gap type). Supplementary. |
| `append_csv.py` | Utility to merge real student session CSV files into the training dataset. |

### Model Files

| File | Size | Description |
|---|---|---|
| `models/precision_cognis_bundle.joblib` | **8.76 MB** | Current production brain. Contains preprocessor, input_expander, and all 6 trained models. |
| `models/cognis_bundle.joblib` | 1.04 MB | Generation 1 session predictor. Used by `/predict` for student profiles. |
| `models/expert_cognis_bundle.joblib` | 2.39 MB | Generation 2. Retired, kept for rollback. |
| `models/precision_diagnostics_report.json` | 9.4 KB | JSON holdout report with per-type precision/recall scores from the last training run. |

### Data Files

| Path | Content |
|---|---|
| `data/precision_diagnostics.jsonl` | 82,900 synthetic training examples (INPUT + TARGET pairs) |
| `data/interaction_logs.jsonl` | **Live telemetry log.** Append-only. Every prediction Cognis makes and every outcome Logic Lens reports is recorded here. Foundation for the self-learning loop. |

---

## 7. Logic Lens Integration — The Handshake

Cognis and Logic Lens communicate via HTTP. Here is the exact data flow:

```
Student types one character in Logic Lens IDE
                    │
                    ▼
  RealTimeMentorEngine.js (in Logic Lens)
  Collects: { code, cursor, key_pressed, student_history }
  Calls: CognisService.predictRealTimeGuidance()
                    │
                    ▼  HTTP POST /predict_realtime
  ┌─────────────────────────────────┐
  │     Cognis FastAPI Server       │
  │     (Port 8000)                 │
  │                                 │
  │  1. Generate UUID prediction_id │
  │  2. cognis_expert.predict()     │
  │  3. update_observability()      │
  │  4. log_event() [background]    │
  └─────────────────────────────────┘
                    │
                    ▼  Response JSON
  {
    "prediction_id": "uuid-v4",
    "error_risk": 0.88,
    "should_speak": true,
    "intervention_type": "hint",
    "diagnosis": [{ "error_type": "missing_colon", "confidence": 0.94, "evidence": "..." }],
    "message": "What do you notice here? Python needs a colon...",
    "mentor_tone": "supportive",
    "next_best_action": "fix_error"
  }
                    │
                    ▼
  RealTimeMentorEngine.js decides:
  ├── remoteConfidence >= 0.60?  → Use AI diagnosis (override local regex)
  └── remoteConfidence < 0.60?   → Fall back to local ErrorEngine.js
                    │
                    ▼
  App.jsx updates realTimeGuidance state
                    │
                    ▼
  RealTimeTooltip.jsx renders 1-line Socratic hint at cursor
```

### Priority Logic (Updated Today)

The key breakthrough was establishing AI priority over local regex scanning:

```javascript
// RealTimeMentorEngine.js — Relaxed Priority v1.2
const shouldOverrideLocal = remoteLooksSpecific && remoteConfidence >= 0.60;
// Previously this required complex structural matching — now AI wins at >= 0.60
```

**Why this matters:** The local `ErrorEngine.js` was aggressively flagging "Indentation Error" for every multi-line snippet. This was masking the AI's more precise diagnosis (missing colon, spelling error, etc.). The fix ensures that if Cognis says "missing_colon" with 94% confidence, that's what the student sees.

### Outcome Telemetry (Self-Learning Loop — Frontend Side)

Logic Lens also reports back when a student fixes an error:

```
Student types ':' after a missing_colon hint
                    │
                    ▼
  App.jsx detects: ErrorEngine now returns no error on that line
                    │
                    ▼  HTTP POST /log_outcome
  {
    "prediction_id": "uuid-v4",
    "student_id": "ali_001",
    "outcome_type": "fixed",
    "student_action": "typed_colon",
    "time_to_fix": 4.2,
    "code_after": "if x > 5:"
  }
```

---

## 8. Self-Learning Loop (Autonomous Improvement)

This was built today. The full loop is designed so Cognis improves from real student usage without manual retraining.

### The Interaction Log Schema

Every prediction and outcome is saved to `data/interaction_logs.jsonl` in a two-event system:

**Prediction Event** (logged on every `/predict_realtime` call):
```json
{
  "event_type": "prediction",
  "prediction_id": "uuid-v4",
  "timestamp": "2026-05-02T21:55:00.000Z",
  "student_id": "ali_001",
  "diagnosis": {
    "error_type": "missing_colon",
    "confidence": 0.94,
    "intervention": "hint",
    "message": "Python needs a colon..."
  }
}
```

**Outcome Event** (logged when student reacts):
```json
{
  "event_type": "outcome",
  "prediction_id": "uuid-v4",
  "timestamp": "2026-05-02T21:55:04.200Z",
  "student_id": "ali_001",
  "outcome_type": "fixed",
  "student_action": "typed_colon",
  "time_to_fix": 4.2,
  "code_after": "if x > 5:"
}
```

### Signal Weights (Future Retraining Use)

| Condition | Signal | Weight |
|---|---|---|
| Student fixed error within 20s of hint | Positive | +1.0 |
| Student typed the exact correction hinted | Strong Positive | +1.0 |
| Student ignored hint, kept typing | Negative | -1.0 |
| Student closed tooltip immediately | Negative | -1.0 |
| Same student frustrated (5+ hints in 2min) | Negative | -1.0 |
| High confidence (>0.9) + student fixed | Good calibration | flag |
| High confidence (>0.9) + student ignored | Poor calibration | flag |

### Retraining Trigger Rules

- **Minimum threshold:** 500 new verified signals before retraining triggers
- **Maximum wait:** 30 days (retrain anyway if data exists)
- **Manual override:** Always available via admin API
- **Gatekeeper:** Per-error-type precision must not drop below 0.85 on holdout — overall accuracy is not sufficient

---

## 9. Live Observability Dashboard

**URL:** `http://127.0.0.1:8000/dashboard`

The dashboard polls the backend every 2 seconds using 3 endpoints:

| Endpoint | Returns |
|---|---|
| `GET /dashboard/stats` | Total predictions today, P95 latency, intervention rate %, active student count |
| `GET /dashboard/recent` | Last 50 predictions as a list (error_type, confidence, latency, intervention) |
| `GET /dashboard/model-health` | Memory usage (MB), uptime, model version, retrain signals collected |

### What Each Panel Shows

**Stat Cards Row:**
- Total Predictions (today, resets on restart)
- P95 Latency in ms (computed from rolling 200-request window)
- Intervention Rate % (how often AI chose to speak)
- Active Students (unique student_ids seen)

**Live Feed Table:**
- Each row = one prediction, newest at top
- Colour coding: green confidence = 80%+, yellow = 60–80%, red = below 60%
- Badge colour: `hint` = blue, `warning` = red, `silence` = grey

**Model Health Panel:**
- Version: v1.2
- Memory: RSS memory of the Python process (via `psutil`)
- Model Size: 8.5 MB (precision bundle)
- Retrain Signals: Count of verified outcomes since last retrain

**Active Alerts:**
- High Latency: P95 > 100ms
- Memory Spike: Process > 500MB
- System OK: Shown in green when all clear

### Terminal Log Filtering

The dashboard polling creates noise in the terminal. A `LogFilter` class silences all `/dashboard/*` and `/health` GET logs, so only meaningful events (predictions, errors) appear.

---

## 10. Current System Size & Performance

### Model Sizes
```
precision_cognis_bundle.joblib    8.76 MB   ← Active precision engine
cognis_bundle.joblib              1.04 MB   ← Active session predictor
expert_cognis_bundle.joblib       2.39 MB   ← Retired
─────────────────────────────────────────
Total models on disk:            12.19 MB
```

### Training Data
```
precision_diagnostics.jsonl:  82,900 examples across 20 error types
Sampled for training:          50,000 (stratified)
Train/test split:              85% / 15%
```

### Runtime Performance (Post-Warm-Up)
```
First keystroke latency (before fix):  ~4300ms  (cold model load)
First keystroke latency (after fix):      ~15ms  (model pre-warmed at startup)
Sustained inference latency:              ~12ms  (target: <50ms)
Memory footprint:                       ~188MB   (Python process RSS)
```

### Classification Performance (Last Training Run)
```
error_type model holdout:
  comparison_vs_assignment:  precision 1.00  recall 1.00  f1 1.00
  missing_colon:             precision 1.00  recall 1.00  f1 1.00
  indentation_error:         precision 1.00  recall 1.00  f1 1.00
  concept_gap:               precision 0.90  recall 0.91  f1 0.91
  spelling_error:            precision 0.88  recall 0.89  f1 0.88
  type_confusion:            precision 0.87  recall 0.86  f1 0.87
  (overall accuracy > 95% across all 20 types)
```

---

## 11. Today's Full Work Log

### 11.1 Model Scaling: 4 → 20 Error Categories
- Rewrote `generate_precision_data.py` to generate 82,900 examples across 20 distinct error categories
- Added 55+ spelling typo variants (SPELLING_VARIANTS list)
- Added 10 realistic type confusion scenarios (TYPE_CONFUSION_SCENARIOS list)
- Added 15+ valid no-error codes to prevent false positives (VALID_NO_ERROR_CODES list)
- Increased `spelling_error` minimum count from 1500 → 2500
- Increased `type_confusion` minimum count from 800 → 2000
- Raised `no_error` confidence range to 0.85–0.98 (was 0.82–0.96)
- Fixed `infinite_loop_risk` to only flag unintentional patterns, not deliberate `while True`

### 11.2 Model Retraining
- Retrained all 6 models on 50,000 stratified samples
- Switched CountVectorizer to trigram (1,3) character n-grams with 2500 features
- Validated: per-type precision >0.85 on 20 categories
- Final bundle size: 8.76 MB

### 11.3 Logic Lens Handshake — Priority Fix
- Fixed `RealTimeMentorEngine.js`: AI confidence >= 0.60 now overrides local regex unconditionally
- Fixed `App.jsx`: local engine errors no longer prevent AI tooltip from clearing
- Added `prediction_id` tracking from backend to frontend for outcome reporting
- Added `socraticData` to guidance object for richer hint context
- Fixed `buildGuidance()`: `primaryEvidence` now correctly sources from `diagnosis.evidence` with full fallback chain

### 11.4 Self-Learning Loop Foundation
- Added `/log_outcome` endpoint with `student_action` granularity (7 action types)
- Added UUID `prediction_id` generation on every `/predict_realtime` call
- Implemented append-only `data/interaction_logs.jsonl` telemetry logger
- Added `student_history_snapshot` to prediction logs (past_errors_total, hints_ignored_today, session_duration)
- Updated `RealtimeHistory` schema with 3 new optional fields for frustration context

### 11.5 Observability Dashboard
- Built `dashboard.html`: a standalone real-time monitoring UI, dark theme, zero dependencies
- Serves from `http://127.0.0.1:8000/dashboard` (FastAPI `HTMLResponse`)
- In-memory `deque`-based state tracking: `inference_stream`, `session_history`, `stats_window`
- Added `GET /dashboard/stats`, `GET /dashboard/recent`, `GET /dashboard/model-health` endpoints
- Added `psutil` for memory monitoring (`pip install psutil`)
- Implemented `LogFilter` to silence dashboard polling from terminal logs

### 11.6 Performance Fixes
- **Model warm-up**: Added `@app.on_event("startup")` to pre-load both bundles and run a dummy inference at server boot — eliminated the 4.3-second first-request spike
- **Event loop fix**: Converted `async def predict_realtime` to `def` so ML inference runs in FastAPI's threadpool instead of blocking the async event loop
- **Confidence threshold**: Lowered both threshold checks in `cognis_expert.py` from 0.60 → 0.50 to eliminate the logic conflict between ML confidence and compiler fallback

### 11.7 Root URL Fix
- Added `GET /` root route returning a professional JSON welcome message with dashboard link

---

## API Quick Reference

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/predict_realtime` | Keystroke-level diagnostic (main Logic Lens endpoint) |
| `POST` | `/log_outcome` | Log student reaction to a hint (self-learning) |
| `POST` | `/predict` | Session-level struggle prediction (student dashboard) |
| `GET` | `/dashboard` | Serve the HTML monitoring dashboard |
| `GET` | `/dashboard/stats` | Aggregated today's stats |
| `GET` | `/dashboard/recent` | Last 50 predictions stream |
| `GET` | `/dashboard/student/{id}` | Full session history for one student |
| `GET` | `/dashboard/model-health` | Memory, uptime, version |
| `GET` | `/health` | Server health check |
| `GET` | `/docs` | Interactive Swagger UI with all schemas |

---

*Documentation last updated: 2026-05-02 | Cognis v1.2 | Logic Lens Python Mentoring System*
