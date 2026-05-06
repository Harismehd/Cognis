import csv
import json
import time
import os

def main():
    csv_path = os.path.join("data", "students_summary_2026-04-28.csv")
    jsonl_path = os.path.join("data", "students.jsonl")
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    with open(jsonl_path, "a", encoding="utf-8") as f:
        for row in rows:
            # Create a schema-compliant object using defaults for missing fields
            now_ms = int(time.time() * 1000)
            student = {
                "student_id": row["student_id"],
                "performance": {
                    "total_problems": int(row["total_problems"]),
                    "successful_runs": int(row["successful_runs"]),
                    "accuracy": int(row["accuracy"]),
                    "completion_rate": int(row["accuracy"]) # default
                },
                "concept_mastery": {
                    "variables": 50,
                    "loops": 50
                },
                "learning_patterns": {
                    "avg_time": int(row["avg_time_ms"]),
                    "hints_used": int(row["hints_used"]),
                    "revision_count": 5,
                    "peak_time": "morning"
                },
                "error_analysis": {
                    "error_types": {
                        "SyntaxError": int(row["syntax_errors"]),
                        "IndentationError": int(row["indentation_errors"]),
                        "TypeError": 0
                    },
                    "common_mistakes": ["missing_colon"],
                    "repeated_errors": 5
                },
                "cognitive_load": {
                    "difficulty_distribution": {
                        "beginner": int(row["total_problems"]) // 2,
                        "intermediate": int(row["total_problems"]) // 2
                    },
                    "optimal_difficulty": "beginner"
                },
                "learning_style": {
                    "pace": "medium",
                    "confidence": 50,
                    "prefers_hints": True,
                    "prefers_visualizations": True
                },
                "engagement": {
                    "days_active": 5,
                    "sessions": 5,
                    "engagement_score": 50
                },
                "struggle_indicators": {
                    "struggling_concepts": ["loops"],
                    "recovery_capacity": 50
                },
                "interaction_patterns": {
                    "mentor_questions": int(row["mentor_questions"]),
                    "question_types": {
                        "General": int(row["mentor_questions"])
                    }
                },
                "code_quality": {
                    "indentation_errors": int(row["indentation_errors"]),
                    "syntax_errors": int(row["syntax_errors"])
                },
                "anti_vibe_coding": {
                    "paste_count": int(row["paste_count"]),
                    "tab_switches": int(row["tab_switches"])
                },
                "timestamps": {
                    "enrollment": now_ms - (7 * 24 * 60 * 60 * 1000),
                    "last_active": now_ms
                }
            }
            f.write(json.dumps(student) + "\n")
    print(f"Appended {len(rows)} records from {csv_path} to {jsonl_path}")

if __name__ == "__main__":
    main()
