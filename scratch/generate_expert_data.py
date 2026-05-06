import csv
import random
import os
import json

# Configuration
OUTPUT_FILE = os.path.join("data", "expert_keystrokes.csv")
TARGET_EVENTS = 60000
SEED = 42

random.seed(SEED)

# Personas
PERSONAS = [
    {"id": "fast_learner", "pref": "text", "style": "direct", "speed": (50, 150), "error_rate": 0.05},
    {"id": "struggling_learner", "pref": "visual", "style": "socratic", "speed": (200, 600), "error_rate": 0.3},
    {"id": "impatient_learner", "pref": "text", "style": "direct", "speed": (30, 100), "error_rate": 0.2},
    {"id": "careful_learner", "pref": "visual", "style": "socratic", "speed": (300, 800), "error_rate": 0.02},
]

# Concepts and Scenarios
SCENARIOS = {
    "variables": [
        {"correct": "x = 5", "errors": [("x = 5", "x - 5", "Syntax: Invalid assignment"), ("x = 5", "x = '5'", "Semantic: Type mixing")]},
        {"correct": "total = total + 1", "errors": [("total = total + 1", "total = totla + 1", "Spelling: Variable name")]},
    ],
    "loops": [
        {"correct": "for i in range(10):", "errors": [("for i in range(10):", "for i in range(10)", "Syntax: Missing colon"), ("for i in range(10):", "for i in rang(10):", "Spelling: range")]},
        {"correct": "while x < 10:", "errors": [("while x < 10:", "while x < 10", "Syntax: Missing colon")]},
    ],
    "conditionals": [
        {"correct": "if x == 5:", "errors": [("if x == 5:", "if x = 5:", "Syntax: Assignment in if"), ("if x == 5:", "if x == 5", "Syntax: Missing colon")]},
    ],
    "functions": [
        {"correct": "def my_func():", "errors": [("def my_func():", "def my_func:", "Syntax: Missing parentheses"), ("def my_func():", "def my_fun():", "Spelling: def name")]},
    ],
    "lists": [
        {"correct": "my_list = [1, 2, 3]", "errors": [("my_list = [1, 2, 3]", "my_list = (1, 2, 3)", "Syntax: Wrong list symbol")]},
    ]
}

ERROR_CATEGORIES = ["syntax", "logic", "concept gap", "spelling", "none"]
INTERVENTIONS = ["silence", "nudge", "hint", "warning"]

def generate_keystrokes():
    events = []
    while len(events) < TARGET_EVENTS:
        persona = random.choice(PERSONAS)
        concept = random.choice(list(SCENARIOS.keys()))
        scenario = random.choice(SCENARIOS[concept])
        
        is_error_scenario = random.random() < 0.4
        if is_error_scenario:
            base_code, target_code, error_label = random.choice(scenario["errors"])
        else:
            target_code = scenario["correct"]
            error_label = None

        current_code = ""
        for i, char in enumerate(target_code):
            # Simulation of a keystroke event
            keystroke = char
            cursor = {"lineNumber": 1, "column": i + 1}
            
            # Labels
            risk_score = 0.0
            intervention = "silence"
            error_cat = "none"
            hint_id = "none"
            confidence = round(random.uniform(0.85, 0.99), 2)
            reasoning = "Student is typing correctly."

            # Check if this keystroke introduces or completes an error
            if error_label:
                # We simplified simulation: if current char is part of the "error" part
                # For real data we'd be more granular. 
                # Let's say if we are at the end of the error snippet
                if i == len(target_code) - 1:
                    risk_score = round(random.uniform(0.75, 1.0), 2)
                    intervention = "warning"
                    error_cat = error_label.split(":")[0].lower()
                    hint_id = f"{concept}_{error_cat}_template"
                    reasoning = f"Detected {error_label}"
                elif i > len(target_code) // 2:
                    risk_score = round(random.uniform(0.4, 0.7), 2)
                    intervention = "hint"
                    error_cat = "none" # not an error yet but high risk
                    hint_id = f"{concept}_proactive_template"
                    reasoning = f"Likely heading towards {error_label}"

            # History
            history = {
                "past_errors": {concept: random.randint(0, 5)},
                "struggle_concepts": [concept] if random.random() < 0.3 else [],
                "hint_style": persona["style"]
            }

            events.append({
                "student_id": f"std_{persona['id']}_{random.randint(100, 999)}",
                "code": current_code,
                "cursor": json.dumps(cursor),
                "key_pressed": keystroke,
                "student_history": json.dumps(history),
                "risk_score": risk_score,
                "intervention_type": intervention,
                "error_category": error_cat,
                "hint_id": hint_id,
                "confidence": confidence,
                "reasoning": reasoning
            })
            
            current_code += char
            if len(events) >= TARGET_EVENTS:
                break
                
    return events

def main():
    print(f"Generating {TARGET_EVENTS} expert keystroke events...")
    events = generate_keystrokes()
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=events[0].keys())
        writer.writeheader()
        writer.writerows(events)
    
    print(f"Successfully saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
