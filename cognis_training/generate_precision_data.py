import copy
import json
import os
import random
from typing import Any, Dict, List, Tuple

OUTPUT_FILE = os.path.join("data", "precision_diagnostics.jsonl")
SEED = 42
random.seed(SEED)

ERROR_MIN_COUNTS = {
    "no_error": 3000,
    "spelling_error": 1500,
    "missing_colon": 1000,
    "missing_parenthesis": 1000,
    "missing_bracket": 800,
    "missing_quote": 800,
    "indentation_error": 1000,
    "unexpected_indent": 800,
    "undefined_variable": 1000,
    "wrong_operator": 800,
    "comparison_vs_assignment": 1000,
    "type_confusion": 800,
    "index_error": 800,
    "infinite_loop_risk": 800,
    "off_by_one_error": 800,
    "wrong_loop_condition": 800,
    "wrong_formula": 600,
    "concept_gap": 1000,
    "multiple_errors": 500,
    "unknown_ambiguous": 500,
    "incomplete_statement": 800,
}

CRITICAL_PAIRS = [
    ("missing_colon", "indentation_error"),
    ("spelling_error", "undefined_variable"),
    ("comparison_vs_assignment", "wrong_operator"),
    ("type_confusion", "wrong_formula"),
    ("index_error", "off_by_one_error"),
    ("missing_parenthesis", "missing_bracket"),
    ("infinite_loop_risk", "wrong_loop_condition"),
    ("concept_gap", "wrong_formula"),
    ("incomplete_statement", "spelling_error"),
]

CONFUSABLE_MAP = {
    "no_error": ["spelling_error", "unknown_ambiguous", "indentation_error"],
    "spelling_error": ["undefined_variable", "no_error", "unknown_ambiguous"],
    "missing_colon": ["indentation_error", "unexpected_indent", "multiple_errors"],
    "missing_parenthesis": ["missing_bracket", "syntax_error", "multiple_errors"],
    "missing_bracket": ["missing_parenthesis", "index_error", "syntax_error"],
    "missing_quote": ["syntax_error", "type_confusion", "unknown_ambiguous"],
    "indentation_error": ["missing_colon", "unexpected_indent", "multiple_errors"],
    "unexpected_indent": ["indentation_error", "missing_colon", "multiple_errors"],
    "undefined_variable": ["spelling_error", "type_confusion", "no_error"],
    "wrong_operator": ["comparison_vs_assignment", "wrong_formula", "type_confusion"],
    "comparison_vs_assignment": ["wrong_operator", "syntax_error", "multiple_errors"],
    "type_confusion": ["wrong_formula", "undefined_variable", "missing_quote"],
    "index_error": ["off_by_one_error", "undefined_variable", "multiple_errors"],
    "infinite_loop_risk": ["wrong_loop_condition", "concept_gap", "no_error"],
    "off_by_one_error": ["index_error", "wrong_loop_condition", "wrong_formula"],
    "wrong_loop_condition": ["infinite_loop_risk", "off_by_one_error", "no_error"],
    "wrong_formula": ["type_confusion", "concept_gap", "wrong_operator"],
    "concept_gap": ["wrong_formula", "wrong_loop_condition", "unknown_ambiguous"],
    "multiple_errors": ["syntax_error", "missing_colon", "missing_parenthesis"],
    "unknown_ambiguous": ["syntax_error", "no_error", "concept_gap"],
    "incomplete_statement": ["spelling_error", "unknown_ambiguous", "missing_colon"],
}

ARCHETYPES = {
    "careless_typist": {
        "hint_style": "nudge",
        "past_errors": {"syntax_errors": 7, "logic_errors": 2, "type_errors": 1},
        "struggle_concepts": [],
        "tone": "encouraging",
    },
    "conceptual_struggler": {
        "hint_style": "socratic",
        "past_errors": {"syntax_errors": 3, "logic_errors": 8, "type_errors": 4},
        "struggle_concepts": ["loops", "conditionals"],
        "tone": "supportive",
    },
    "hint_ignorer": {
        "hint_style": "direct",
        "past_errors": {"syntax_errors": 5, "logic_errors": 6, "type_errors": 3},
        "struggle_concepts": ["variables", "functions"],
        "tone": "firm",
    },
    "frustrated_learner": {
        "hint_style": "supportive",
        "past_errors": {"syntax_errors": 9, "logic_errors": 5, "type_errors": 6},
        "struggle_concepts": ["loops", "lists"],
        "tone": "gentle",
    },
    "confident_builder": {
        "hint_style": "silent",
        "past_errors": {"syntax_errors": 1, "logic_errors": 0, "type_errors": 0},
        "struggle_concepts": [],
        "tone": "silent",
    },
}

ERROR_MESSAGES = {
    "no_error": "Code is syntactically valid and no likely misconception is detected.",
    "spelling_error": "Check the spelling closely; small typos can change Python keywords or variable names.",
    "missing_colon": "This block header needs a colon at the end to start the indented body.",
    "missing_parenthesis": "A function call or grouping expression is missing a closing parenthesis.",
    "missing_bracket": "A list or indexing operation is missing a closing bracket.",
    "missing_quote": "This string literal is not closed with a matching quote.",
    "indentation_error": "The line indentation is inconsistent or does not match the surrounding block.",
    "unexpected_indent": "There is indentation where Python does not expect a new block.",
    "undefined_variable": "This name has not been assigned yet or may be misspelled.",
    "wrong_operator": "The comparison operator is likely the wrong one for this condition.",
    "comparison_vs_assignment": "Inside this condition, '=' assigns rather than compares — use '==' instead.",
    "type_confusion": "A type mismatch is occurring; check whether values are strings, numbers, or lists.",
    "index_error": "The code is accessing a list or string index that does not exist.",
    "infinite_loop_risk": "This loop as written can repeat forever because the exit condition never changes.",
    "off_by_one_error": "The loop or range boundary is off by one, so it will execute too few or too many times.",
    "wrong_loop_condition": "The loop condition is incorrect, so the loop may never start or will stop too early.",
    "wrong_formula": "The arithmetic expression has the wrong operation or order of operations.",
    "concept_gap": "This isn't a typo — it looks like you're thinking about the problem in a way that won't work in Python. Let's step back.",
    "multiple_errors": "There's more than one issue here. Start with the first underlined error and fix it before moving on.",
    "unknown_ambiguous": "I can't tell what you're trying to do yet. Keep typing and I'll help when it's clearer.",
    "incomplete_statement": "You've started a statement but it's not complete. What should come after this keyword?",
}

NO_ERROR_TEMPLATES = [
    "if count == len(items):",
    "result = numbers[0] + numbers[-1]",
    "print(f\"Value is {value}\")",
    "if x in data and data[x] == target:",
]

SUSPECT_TYPES = ["int", "str", "list", "dict"]


def _random_name() -> str:
    return random.choice(["count", "value", "result", "index", "total", "item", "score"])


def _random_number() -> str:
    return str(random.randint(1, 9))


def cursor_after(code: str) -> Dict[str, int]:
    lines = code.splitlines() or [""]
    return {"lineNumber": len(lines), "column": len(lines[-1]) + 1}


def _infer_typing_state(code: str) -> str:
    text = str(code or "").rstrip()
    if not text or text.endswith((":", "(", "[", "{", "+", "-", "*", "/", "=")):
        return "partial"
    if "\n" in text and text.splitlines()[-1].strip() == "":
        return "partial"
    return "complete"


def _make_input(code: str, key: str, student_history: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "code": code,
        "cursor": cursor_after(code),
        "key_pressed": key,
        "student_history": student_history,
        "typing_state": _infer_typing_state(code),
        "current_errors": [],
    }


def _make_target(error_type: str, line: int, column: int, message: str, intervention_type: str, should_speak: bool, tone: str, next_best_action: str, confidence: float, error_risk: float) -> Dict[str, Any]:
    diagnosis = [{
        "error_type": error_type,
        "line": line,
        "column": column,
        "confidence": round(confidence, 2),
        "evidence": message,
        "fix_priority": 1 if error_type != "no_error" else 0,
    }]
    return {
        "error_risk": round(error_risk, 2),
        "should_speak": should_speak,
        "intervention_type": intervention_type,
        "diagnosis": diagnosis,
        "message": message if should_speak else "",
        "mentor_tone": tone,
        "next_best_action": next_best_action,
    }


def _student_history(archetype: str) -> Dict[str, Any]:
    arche = ARCHETYPES[archetype]
    return {
        "past_errors": arche["past_errors"],
        "struggle_concepts": arche["struggle_concepts"],
        "hint_style": arche["hint_style"],
    }


def _intervention_for(error_type: str, archetype: str) -> Tuple[str, bool, str, str, float]:
    arche = ARCHETYPES[archetype]
    base_risk = 0.88 if error_type != "no_error" else 0.05
    if archetype == "careless_typist":
        intervention = "nudge" if error_type in ["spelling_error", "missing_colon", "missing_parenthesis"] else "warning"
        should_speak = error_type != "no_error"
        tone = "encouraging"
    elif archetype == "conceptual_struggler":
        intervention = "hint" if error_type not in ["no_error", "unknown_ambiguous"] else "silence"
        should_speak = error_type not in ["no_error", "unknown_ambiguous"]
        tone = "supportive"
    elif archetype == "hint_ignorer":
        intervention = "warning" if error_type not in ["no_error", "unknown_ambiguous"] else "silence"
        should_speak = error_type not in ["no_error", "unknown_ambiguous"]
        tone = "firm"
    elif archetype == "frustrated_learner":
        intervention = "hint" if error_type not in ["no_error", "unknown_ambiguous"] else "silence"
        should_speak = error_type not in ["no_error", "unknown_ambiguous"]
        tone = "gentle"
    else:
        intervention = "silence" if error_type == "no_error" else "warning"
        should_speak = error_type not in ["no_error", "unknown_ambiguous"]
        tone = "silent" if error_type == "no_error" else "supportive"

    if error_type == "incomplete_statement":
        intervention = "nudge"
        should_speak = True
        tone = "encouraging"
        next_best_action = "allow_student_to_continue"
        error_risk = 0.50
    else:
        next_best_action = "allow_student_to_continue" if intervention == "silence" else "fix_error"
        error_risk = base_risk + random.uniform(-0.05, 0.07) if error_type != "no_error" else 0.05
    
    return intervention, should_speak, tone, next_best_action, max(0.0, min(1.0, error_risk))


def _choose_confidence(error_type: str) -> float:
    if error_type == "no_error":
        return random.uniform(0.82, 0.96)
    if error_type == "unknown_ambiguous":
        return random.uniform(0.40, 0.58)
    if error_type == "incomplete_statement":
        return random.uniform(0.82, 0.92)
    return random.uniform(0.88, 0.98)


def generate_code(error_type: str) -> Tuple[str, str, str, str]:
    name = _random_name()
    other = _random_name()
    num = _random_number()

    def multiline(*lines: str) -> str:
        return "\n".join(lines)

    if error_type == "no_error":
        templates = [
            multiline(f"for i in range({num}):", "    print(i)", "print(\"done\")"),
            multiline(f"def {name}():", f"    return {num}", f"print({name}())"),
            multiline(f"items = [{num}, {num}, {num}]", "for item in items:", "    print(item)"),
            multiline(f"if {name} > 0:", f"    total = {name}", "    print(total)"),
            multiline(f"result = [x for x in range({num}) if x % 2 == 0]", "print(result)", "print(\"done\")"),
            multiline(f"{name} = {num}", f"if {name} == {num}:", "    print('match')"),
            multiline(f"data = {{'key': {num}}}", f"print(data.get('key'))"),
            multiline(f"import math", f"print(math.sqrt({num}))"),
            multiline(f"def greet(n):", f"    print('hello ' + str(n))", "greet('user')"),
            multiline(f"x = {num}", "y = x * 2", "print(y)"),
            # Partially typed code that is NOT yet an error
            f"if {name} == {num}:",
            f"for i in range({num}):",
            f"def {name}():",
            f"while {name} < {num}:",
            f"elif {name} > 0:",
            f"else:",
            f"try:",
            f"except Exception as e:",
        ]
        return random.choice(templates), "Enter", ERROR_MESSAGES[error_type], "no_error"

    if error_type == "spelling_error":
        variants = [
            multiline(f"whille {name} < {num}:", "    print({name})"),
            f"pritn(\"{name}\")",
            f"defin {name}():",
            f"returnr {name}",
            f"iif {name} == {num}:",
            f"forr i in range({num}):",
            f"lamba x: x + 1",
            f"tryy:",
            f"excepte Exception:",
            f"impot math",
        ]
        return random.choice(variants), "Enter", ERROR_MESSAGES[error_type], "spelling_error"

    if error_type == "missing_colon":
        variants = [
            multiline(f"if {name} > {num}", "    print(\"yes\")", "print(\"done\")"),
            multiline(f"for {name} in range({num})", "    total += {name}", "print(total)"),
            multiline(f"while {name} <= {num}", "    {name} += 1", "print({name})"),
            multiline(f"def {name}()", "    return {name}", "print({name})"),
            multiline(f"elif {name} < 0", "    print('neg')"),
            multiline(f"else", "    print('default')"),
            multiline(f"with open('file.txt', 'r') as f", "    pass"),
            multiline(f"class {name.capitalize()}", "    pass"),
            multiline(f"try", "    x = 1"),
            multiline(f"except ValueError", "    print('err')"),
        ]
        return random.choice(variants), "Enter", ERROR_MESSAGES[error_type], "missing_colon"

    if error_type == "missing_parenthesis":
        variants = [
            multiline(f"print(\"{name}\"", "if {name} > 0:", "    print({name})"),
            multiline(f"max({num}, {num}", "result = 0", "print(result)"),
            multiline(f"len([{num}, {num}]", "print(\"done\")"),
            multiline(f"round({num} / 3", "print(\"ok\")"),
            f"sum([1, 2, 3]",
            f"abs(-5",
            f"input('enter name'",
            f"int('123'",
            f"str({num}",
            f"def func(a, b",
        ]
        return random.choice(variants), "Enter", ERROR_MESSAGES[error_type], "missing_parenthesis"

    if error_type == "missing_bracket":
        variants = [
            multiline(f"numbers = [{num}, {num}, {num}", "print(numbers[0])", "print(len(numbers))"),
            multiline(f"result = values[{num}", "print(result)", "print(\"done\")"),
            multiline(f"matrix = [[1, 2], [3, 4]", "print(matrix[1][1])"),
            multiline(f"item = data[{num}", "print(item)", "print(\"ok\")"),
            f"lst = [1, 2",
            f"d = {{'a': 1",
            f"val = data['key'",
            f"arr = [x for x in range(5)",
            f"print(colors[0",
            f"my_dict = {{\"id\": 1",
        ]
        return random.choice(variants), "Enter", ERROR_MESSAGES[error_type], "missing_bracket"

    if error_type == "missing_quote":
        variants = [
            multiline(f"text = \"Hello {name}", "print(text)", "print(\"done\")"),
            multiline(f"message = '{name}", "print(message)", "print(\"end\")"),
            f"print(\"She said \\\"hello\\\"\")",
            f"quote = '''{name}",
            f"s = \"incomplete",
            f"tag = '<div'",
            f"path = 'C:\\Users\\",
            f"f\"Value: {{val}}",
            f"r\"raw string",
            f"'single quote mismatch\"",
        ]
        return random.choice(variants), "Enter", ERROR_MESSAGES[error_type], "missing_quote"

    if error_type == "indentation_error":
        variants = [
            multiline(f"if {name} > {num}:", "    print({name})", "  print(\"done\")"),
            multiline(f"def {name}():", "\tprint({name})", "  return {name}"),
            multiline(f"for {name} in range({num}):", "    if {name} % 2 == 0:", " print({name})"),
            multiline(f"while True:", "print('loop')"),
            multiline(f"def test():", "  x = 1", "    y = 2"),
            multiline(f"if True:", "    pass", "   pass"),
        ]
        return random.choice(variants), "Enter", ERROR_MESSAGES[error_type], "indentation_error"

    if error_type == "unexpected_indent":
        variants = [
            multiline("    print(\"unexpected\")", "print(\"done\")"),
            multiline(f"\treturn {name}", "print(\"next\")"),
            multiline(f"    if {name} > {num}:", "print({name})"),
            multiline(" x = 5", "print(x)"),
            multiline("  # comment", "print(1)"),
            multiline("    pass"),
        ]
        return random.choice(variants), "Enter", ERROR_MESSAGES[error_type], "unexpected_indent"

    if error_type == "undefined_variable":
        variants = [
            multiline(f"total = {name} + {num}", "print(total)", f"print({other})"),
            multiline(f"if {name} > 0:", "    result = {name} + 1", "print(result)"),
            multiline(f"x = {num}", "print(y)", "print(\"done\")"),
            multiline(f"for i in range(5):", "    total += i", "print(total)"),
            f"print(undefined_var)",
            f"res = a * b",
        ]
        return random.choice(variants), "Enter", ERROR_MESSAGES[error_type], "undefined_variable"

    if error_type == "wrong_operator":
        variants = [
            multiline(f"if {name} >= {num}:", "    print(\"yes\")", "print(\"done\")"),
            multiline(f"if {name} < {num}:", "    print(\"low\")", "print(\"done\")"),
            multiline(f"if {name} != {num}:", "    print(\"check\")", "print(\"done\")"),
            f"x => 5",
            f"y =< 10",
            f"if a ! b:",
        ]
        return random.choice(variants), "Enter", ERROR_MESSAGES[error_type], "wrong_operator"

    if error_type == "comparison_vs_assignment":
        variants = [
            f"if {name} = {num}:\n    print(\"oops\")",
            f"while {name} = 0:\n    print({name})",
            f"if {name} = {other}:\n    print(\"check\")",
            f"elif x = 10:",
            f"if True and y = 5:",
        ]
        return random.choice(variants), "Enter", ERROR_MESSAGES[error_type], "comparison_vs_assignment"

    if error_type == "type_confusion":
        variants = [
            multiline(f"result = \"{num}\" + {num}", "print(result)", "print(type(result))"),
            multiline(f"total = len({num})", "print(total)", "print(\"done\")"),
            multiline(f"value = [1, 2] + \"{name}\"", "print(value)", "print(\"done\")"),
            multiline(f"for x in {num}:", "    print(x)", "print(\"done\")"),
            f"'hello' - 5",
            f"5 / '2'",
            f"None + 1",
        ]
        return random.choice(variants), "Enter", ERROR_MESSAGES[error_type], "type_confusion"

    if error_type == "index_error":
        variants = [
            multiline(f"items = [{num}, {num}]", "print(items[2])", "print(\"done\")"),
            multiline(f"letters = [\"a\", \"b\"]", "print(letters[-3])", "print(\"done\")"),
            multiline(f"names = [\"a\", \"b\", \"c\"]", "print(names[5])", "print(\"done\")"),
            f"text = 'hi'",
            f"print(text[10])",
        ]
        return random.choice(variants), "Enter", ERROR_MESSAGES[error_type], "index_error"

    if error_type == "infinite_loop_risk":
        variants = [
            multiline("while True:", f"    print({name})", "# no exit"),
            multiline(f"while {name} < {name}:", f"    {name} += 1", "print(\"done\")"),
            multiline("i = 0", "while i == 0:", "    print(i)"),
            multiline(f"while {name} > 0:", "    print('active')"),
        ]
        return random.choice(variants), "Enter", ERROR_MESSAGES[error_type], "infinite_loop_risk"

    if error_type == "off_by_one_error":
        variants = [
            multiline(f"for i in range(1, {num}):", "    print(i)", "print(\"done\")"),
            multiline("for i in range(len(items)):", "    print(items[i+1])", "print(\"done\")"),
            multiline(f"if {name} <= {num}:", "    print({name})", "print(\"done\")"),
            f"range(0, 10)",
            f"range(1, 11)",
        ]
        return random.choice(variants), "Enter", ERROR_MESSAGES[error_type], "off_by_one_error"

    if error_type == "wrong_loop_condition":
        variants = [
            multiline(f"while {num} > {name}:", f"    {name} += 1", "print(\"done\")"),
            multiline(f"while {name} < 0:", "    print({name})", "print(\"done\")"),
            multiline("for i in range(0):", "    print(i)", "print(\"done\")"),
            f"while False:",
        ]
        return random.choice(variants), "Enter", ERROR_MESSAGES[error_type], "wrong_loop_condition"

    if error_type == "wrong_formula":
        variants = [
            multiline(f"area = {name} + {other}", "print(area)", "print(\"done\")"),
            multiline("average = total / count * 2", "print(average)", "print(\"done\")"),
            multiline("discount = price - price * rate + 10", "print(discount)", "print(\"done\")"),
            f"x = 1 2",
            f"y = (5 + 2",
        ]
        return random.choice(variants), "Enter", ERROR_MESSAGES[error_type], "wrong_formula"

    if error_type == "concept_gap":
        variants = [
            multiline(f"if {name} > 0:", "    print(\"done\")", "else:\n    print(\"done\")"),
            multiline("for i in range(5):", "    total = i", "print(total)"),
            multiline(f"result = [x for x in range({num}) if x % 2 == 0]", "print(result)", "print(\"done\")"),
            multiline("x = 5", "if x == 5:", "    x = 5"),
        ]
        return random.choice(variants), "Enter", ERROR_MESSAGES[error_type], "concept_gap"

    if error_type == "multiple_errors":
        variants = [
            multiline(f"if {name} = {num}", "print(\"oops\")"),
            multiline(f"pritn(\"hello\"", "print(\"done\")"),
            multiline(f"for {name} in range({num})", f"print({other})", "print(\"done\")"),
            f"def f() print x",
        ]
        return random.choice(variants), "Enter", ERROR_MESSAGES[error_type], "multiple_errors"

    if error_type == "unknown_ambiguous":
        variants = [
            multiline(f"result = {name} * {other}", "print(result)", "print(\"done\")"),
            multiline(f"output = data.get({name})", "print(output)", "print(\"done\")"),
            f"print({name})",
            f"x = 1",
            f"# just a comment",
        ]
        return random.choice(variants), "Enter", ERROR_MESSAGES[error_type], "unknown_ambiguous"

    if error_type == "incomplete_statement":
        variants = [
            "for:",
            "if:",
            "while:",
            "def:",
            "elif:",
            "for i",
            "if x",
            "while True",
            "def greet",
            "for i in",
            "if x >",
            "while count",
            "for i in range(5):\n",
            "if x > 5:\n",
            "while True:\n",
            "def greet():\n",
        ]
        code = random.choice(variants)
        message = ERROR_MESSAGES["incomplete_statement"]
        if code.endswith(":\n"):
            message = "Good start! Now what should happen inside this block? Remember to indent."
        return code, "Enter", message, "incomplete_statement"

    return "", "Enter", "", error_type


def _make_entry(error_type: str, archetype: str = None, mark_confusion: bool = False) -> Dict[str, Any]:
    code, key, message, label = generate_code(error_type)
    student_history = _student_history(archetype) if archetype else {
        "past_errors": {"syntax_errors": random.randint(0, 3), "logic_errors": random.randint(0, 2), "type_errors": random.randint(0, 2)},
        "struggle_concepts": [] if error_type == "confident_builder" else [random.choice(["loops", "conditionals", "variables"])],
        "hint_style": random.choice(["socratic", "direct", "supportive", "silent"]),
    }
    intervention_type, should_speak, tone, next_best_action, error_risk = _intervention_for(error_type, archetype or "confident_builder")
    confidence = _choose_confidence(error_type)
    target = _make_target(error_type, cursor_after(code)["lineNumber"], cursor_after(code)["column"], message, intervention_type, should_speak, tone, next_best_action, confidence, error_risk)
    if mark_confusion:
        target["confusing_pair_context"] = True
    return {"INPUT": _make_input(code, key, student_history), "TARGET": target}


def _make_rejection_examples(example: Dict[str, Any], error_type: str) -> List[Dict[str, Any]]:
    rejections = []
    rejected_types = CONFUSABLE_MAP.get(error_type, [])[:3]
    for rejected_type in rejected_types:
        rejection = copy.deepcopy(example)
        reason = f"This is {error_type}, not {rejected_type}."
        rejection["TARGET"]["message"] = reason
        rejection["TARGET"]["mentor_tone"] = rejection["TARGET"]["mentor_tone"]
        rejection["TARGET"]["diagnosis"][0]["evidence"] = reason
        rejection["TARGET"]["rejected_type"] = rejected_type
        rejection["TARGET"]["rejection_confidence"] = round(random.uniform(0.04, 0.14), 2)
        rejections.append(rejection)
    return rejections


def generate_examples_by_type() -> List[Dict[str, Any]]:
    dataset: List[Dict[str, Any]] = []
    for error_type, min_count in ERROR_MIN_COUNTS.items():
        for _ in range(min_count):
            entry = _make_entry(error_type)
            dataset.append(entry)
            dataset.extend(_make_rejection_examples(entry, error_type))
        if error_type == "incomplete_statement":
            for _ in range(200):
                # Force the multi-line pattern for these 200 examples
                variants = [
                    "for i in range(5):\n",
                    "if x > 5:\n",
                    "while True:\n",
                    "def greet():\n",
                ]
                code = random.choice(variants)
                message = "Good start! Now what should happen inside this block? Remember to indent."
                # We need to manually construct the entry or slightly modify generate_code
                # But to stay clean, I'll just ensure generate_code handles it if we pass a flag, 
                # or just trust the increased count if I increase ERROR_MIN_COUNTS.
                # The user asked for "Add 200 examples of this pattern".
                # I will just manually add them here to be sure.
                entry = _make_entry(error_type)
                # Override the code and message for these 200
                entry["INPUT"]["code"] = code
                entry["INPUT"]["cursor"] = cursor_after(code)
                entry["TARGET"]["message"] = message
                entry["TARGET"]["diagnosis"][0]["evidence"] = message
                entry["TARGET"]["diagnosis"][0]["line"] = cursor_after(code)["lineNumber"]
                entry["TARGET"]["diagnosis"][0]["column"] = cursor_after(code)["column"]
                dataset.append(entry)
                dataset.extend(_make_rejection_examples(entry, error_type))
    return dataset


def generate_critical_pair_examples() -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    for a, b in CRITICAL_PAIRS:
        for error_type in (a, b):
            for _ in range(200):
                entry = _make_entry(error_type)
                entry["TARGET"]["confusable_with"] = b if error_type == a else a
                entry["TARGET"]["message"] = f"This is {error_type}; do not confuse it with {entry['TARGET'][ 'confusable_with']} ."
                examples.append(entry)
    return examples


def generate_archetype_examples() -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    archetype_types = list(ARCHETYPES.keys())
    for archetype in archetype_types:
        for _ in range(500):
            error_type = random.choice(list(ERROR_MIN_COUNTS.keys()))
            entry = _make_entry(error_type, archetype=archetype)
            entry["TARGET"]["archetype"] = archetype
            examples.append(entry)
    return examples


def main() -> None:
    print("Generating precision diagnostic examples for 20 error types...")
    data = generate_examples_by_type()
    data.extend(generate_critical_pair_examples())
    data.extend(generate_archetype_examples())

    random.shuffle(data)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")

    counts = {}
    for entry in data:
        et = entry["TARGET"]["diagnosis"][0]["error_type"]
        counts[et] = counts.get(et, 0) + 1
    print(f"Saved {len(data)} examples to {OUTPUT_FILE}")
    print("Counts by error type:")
    for et in sorted(counts):
        print(f"  {et}: {counts[et]}")

if __name__ == "__main__":
    main()
