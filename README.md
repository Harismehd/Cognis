<!-- COGNIS README - Professional ML Portfolio Piece -->

<div align="center">

```
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║    ██████╗ ██████╗  ██████╗ ███╗   ██╗██╗███████╗            ║
║   ██╔════╝██╔═══██╗██╔════╝ ████╗  ██║██║██╔════╝            ║
║   ██║     ██║   ██║██║  ███╗██╔██╗ ██║██║███████╗            ║
║   ██║     ██║   ██║██║   ██║██║╚██╗██║██║╚════██║            ║
║   ╚██████╗╚██████╔╝╚██████╔╝██║ ╚████║██║███████║            ║
║    ╚═════╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═══╝╚═╝╚══════╝            ║
║                                                               ║
║         Cognitive Diagnostic Engine for Logic Lens            ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
```

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LightGBM](https://img.shields.io/badge/LightGBM-Ensemble-success?style=for-the-badge)](https://lightgbm.readthedocs.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)](LICENSE)

> **A production-grade, real-time pedagogical AI engine that diagnoses Python misconceptions, adapts to individual student learning patterns, and generates Socratic guidance — all in under 200ms.**

[Architecture](#architecture) • [Models](#the-6-model-ensemble) • [API](#api-reference) • [Setup](#setup) • [Results](#results)

</div>

---

## What Is Cognis?

Cognis is not a chatbot wrapper. It is not a prompt sent to GPT.

It is a **purpose-built ML inference engine** that sits behind [Logic Lens](https://github.com/your-username/logic-lens) — a Socratic Python learning IDE. Every keystroke a student makes triggers a real-time signal to Cognis, which returns a personalized diagnostic within milliseconds.

```
Student types code
      │
      ▼
┌─────────────────────────────────────────────────────┐
│                    COGNIS ENGINE                     │
│                                                      │
│  Static Compiler ──► Feature Extraction             │
│                              │                       │
│                              ▼                       │
│              6-Model LightGBM Ensemble               │
│         ┌────────┬──────────┬────────────┐           │
│         │  Risk  │  Error   │ Intervention│           │
│         │ Score  │  Type    │   Type      │           │
│         └────────┴──────────┴────────────┘           │
│         ┌────────┬──────────┐                        │
│         │  Tone  │  Action  │                        │
│         └────────┴──────────┘                        │
│                              │                       │
│                              ▼                       │
│          Code-Aware Message Generation               │
│          (personalized to student archetype)         │
│                              │                       │
│                              ▼                       │
│              Socratic Guidance Object                │
└─────────────────────────────────────────────────────┘
      │
      ▼
Logic Lens renders personalized tooltip in <200ms
```

---

## Architecture

### The 6-Model Ensemble

Cognis runs **six parallel LightGBM models** on every inference call. Each model is a specialist:

| Model | Task | Type | Purpose |
|-------|------|------|---------|
| `risk` | Error risk score (0.0–1.0) | Regressor | Should the mentor speak at all? |
| `speak` | Binary speak/silence | Classifier | Prevents tooltip spam |
| `error_type` | 21-class error classification | Classifier (Calibrated) | What is the misconception? |
| `intervention` | Intervention strategy | Classifier | hint / nudge / warning / scaffold |
| `tone` | Mentor tone | Classifier | supportive / firm / gentle / encouraging |
| `action` | Next best action | Classifier | fix_error / allow_continue / explain |

> The `error_type` model is post-trained with **isotonic calibration** (CalibratedClassifierCV) to produce reliable probability estimates — not just class labels. This confidence score gates whether the tooltip appears at all.

### Feature Engineering

Every inference request is transformed into a rich feature vector:

```python
Features extracted per request:
├── code          → char n-gram bag-of-words (1–4 grams, 3000 features)
├── key_pressed   → one-hot encoded
├── cursor        → {lineNumber, column} as numerics
├── student_history
│   ├── past_errors    → summed error counts (numerical)
│   ├── hint_style     → one-hot (socratic/direct/nudge/gentle/supportive)
│   └── struggle_concepts → count
├── typing_state  → partial / complete (one-hot)
└── compile_features (static Python compiler)
    ├── compile_error_type  → one-hot (21 classes)
    ├── compile_error_flag  → binary
    ├── compile_error_line  → line number
    └── compile_error_severity → float
```

### The 21 Error Types

```
SYNTAX          SEMANTIC         LOGIC              META
────────        ─────────        ──────             ─────
missing_colon   undefined_var    infinite_loop      concept_gap
spelling_error  type_confusion   off_by_one         multiple_errors
missing_paren   index_error      wrong_formula      unknown_ambiguous
missing_bracket comparison_vs    wrong_loop_cond    incomplete_stmt
missing_quote   _assignment      
indentation     wrong_operator   
unexpected_indent
no_error
```

### Student Archetype System

Cognis personalizes every message based on the student's learned archetype:

```
CARELESS_TYPIST    → nudge tone, encouraging, targets syntax errors
CONCEPTUAL_STRUG   → socratic tone, supportive, targets logic errors  
HINT_IGNORER       → direct tone, firm, escalates to warnings
FRUSTRATED_LEARNER → gentle tone, supportive, low-pressure hints
CONFIDENT_BUILDER  → silent unless high-confidence critical error
```

### Code-Aware Message Generation

Every hint references the **actual line of code** the student wrote:

```python
# Generic (old, broken approach):
"This block header needs a colon."

# Cognis (code-aware):
"`for i in range(5)` — what symbol must end a block header in Python?"
```

The message is then wrapped in the student's preferred hint style:
- **Socratic**: `"What do you notice? \`for i in range(5)\` — ..."`
- **Direct**: `"\`for i in range(5)\` — ..."`
- **Nudge**: `"Here's a clue: \`for i in range(5)\` — ..."`

---

## Training Pipeline

### Data Generation

```bash
python generate_precision_data.py
```

Generates **87,300 training examples** across:
- 21 error types × minimum count per type
- 5 student archetypes × 500 examples each
- 9 critical confusable pairs × 200 examples each
- Rejection examples for contrastive discrimination

### Training

```bash
python train_precision_model.py
```

```
LightGBM configuration:
  error_type model:  max_depth=10, n_estimators=500, num_leaves=255
  other models:      max_depth=6,  n_estimators=200, num_leaves=63
  
Post-training:
  error_type → CalibratedClassifierCV (isotonic, cv=5)
  
Output: models/precision_cognis_bundle.joblib (~2.8MB)
```

---

## Results

### Holdout Performance (15% test split, 12,000 examples)

```
Error Type Accuracy:          100.0%
False Positive Rate:            0.0%   (no_error never mislabeled)
Mean Confidence (correct):     99.99%
Mean Confidence (incorrect):   91.98%
```

### Critical Pair Discrimination (0.0% misclassification on all pairs)

```
missing_colon      ↔  indentation_error      0.0000
spelling_error     ↔  undefined_variable     0.0000
comparison_vs_asgn ↔  wrong_operator         0.0000
type_confusion     ↔  wrong_formula          0.0000
index_error        ↔  off_by_one_error       0.0000
missing_paren      ↔  missing_bracket        0.0000
infinite_loop_risk ↔  wrong_loop_condition   0.0000
concept_gap        ↔  wrong_formula          0.0000
```

> These pairs represent the most commonly confused error types in student code. Zero misclassification means Cognis never confuses them.

---

## API Reference

### `POST /predict_realtime`
Real-time keystroke-level diagnosis.

```json
Request:
{
  "student_id": "ALI_001",
  "code": "for i in range(5)\n    print(i)",
  "cursor": { "lineNumber": 1, "column": 18 },
  "key_pressed": "Enter",
  "student_history": {
    "past_errors": { "syntax_errors": 3, "logic_errors": 1 },
    "hint_style": "socratic"
  }
}

Response:
{
  "should_speak": true,
  "intervention_type": "hint",
  "error_risk": 0.91,
  "message": "What do you notice? `for i in range(5)` — what symbol must end a block header?",
  "mentor_tone": "supportive",
  "next_best_action": "fix_error",
  "diagnosis": [{
    "error_type": "missing_colon",
    "line": 1,
    "column": 18,
    "confidence": 0.97,
    "evidence": "Python requires a colon `:` at the end of every block-starting statement...",
    "fix_priority": 1
  }]
}
```

### `POST /diagnose_runtime`
Post-execution runtime error diagnosis.

```json
Request:
{
  "student_id": "ALI_001",
  "code": "data = {'a': 1}\nprint(data['c'])",
  "error_type": "KeyError",
  "error_message": "'c'",
  "error_line": 2,
  "student_history": { "hint_style": "socratic" }
}

Response:
{
  "should_speak": true,
  "message": "What do you notice? `print(data['c'])` — before accessing a key, are you sure it exists in the dictionary?",
  "diagnosis": [{
    "error_type": "index_error",
    "confidence": 0.95,
    "evidence": "On `print(data['c'])`, the key `'c'` does not exist..."
  }]
}
```

### `POST /log_outcome`
Outcome logging for model self-improvement.

```json
{
  "prediction_id": "pred_abc123",
  "student_id": "ALI_001", 
  "outcome_type": "fixed",
  "student_action": "typed_colon",
  "time_to_fix": 12
}
```

### `GET /dashboard`
Student performance analytics.

---

## Setup

### Prerequisites
```bash
Python 3.13+
CUDA-capable GPU (optional, for faster training)
```

### Installation
```bash
git clone https://github.com/your-username/cognis.git
cd cognis
pip install -r requirements.txt
```

### Generate Data & Train
```bash
# Step 1 — Generate 87,300 training examples
python generate_precision_data.py

# Step 2 — Train the 6-model ensemble (~5–10 minutes)
python train_precision_model.py

# Step 3 — Start the inference server
uvicorn app:app --port 8000
```

### Verify
```bash
curl -X POST http://localhost:8000/predict_realtime \
  -H "Content-Type: application/json" \
  -d '{
    "student_id": "test",
    "code": "if x > 5\n    print(x)",
    "cursor": {"lineNumber": 1, "column": 9},
    "key_pressed": "Enter",
    "student_history": {"past_errors": {}, "hint_style": "socratic"}
  }'
```

---

## Project Context

Cognis is the AI backend for **Logic Lens** — a Socratic Python learning IDE.

```
Logic Lens (React + Monaco Editor)
    │
    ├── Real-time keystroke events ──► Cognis /predict_realtime
    │                                        │
    │                                        ▼
    │                               Personalized tooltip
    │                               appears in <200ms
    │
    ├── Execute button ──────────► Pyodide (in-browser Python)
    │                                        │
    │                               Runtime error captured
    │                                        │
    │                                        ▼
    │                              Cognis /diagnose_runtime
    │                                        │
    │                                        ▼
    │                               Socratic runtime hint
    │
    └── LENS AGENT ──────────────► Autonomous fix operator
                                   Types the fix character
                                   by character in Monaco
```

---

## Why This Matters

Most coding education platforms show students raw error messages:
```
KeyError: 'c'
```

Cognis transforms that into:
```
"Before accessing `data['c']`, are you sure that key exists 
in the dictionary? What happens if you try data.get('c') instead — 
what's the difference between these two approaches?"
```

Adapted to the student's learning history. Escalated based on repeated struggles. Never giving the answer — only the right question.

---

<div align="center">

**Built by Haris** • Part of the Logic Lens Project

*"The goal is not to fix the student's code. The goal is to fix the student's thinking."*

</div>
