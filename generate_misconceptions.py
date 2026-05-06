import csv
import random
import os

# Configuration
OUTPUT_FILE = os.path.join("data", "misconceptions.csv")
TARGET_COUNT = 600
SEED = 42

random.seed(SEED)

# Data pools
VAR_NAMES = ["x", "y", "z", "val", "count", "data", "item", "total", "score", "n", "i", "j", "my_var", "result"]
VALUES = [0, 1, 5, 10, 100, -1, "''", "'hello'", "'test'", "True", "False", "None"]
KEYWORDS = ["if", "for", "while", "def", "class", "import", "return", "pass", "break", "continue"]

# Templates: (snippet_template, error_type, misconception_label, base_intensity)
TEMPLATES = {
    "variables": [
        ("{var1} = {val1}\nprint({var2})", "Semantic", "Using undefined variable", 7),
        ("{var1} + {val1} = {var2}", "Syntax", "Invalid assignment target", 6),
        ("{val1} = {var1}", "Syntax", "Assigning to literal", 5),
        ("{var1}-name = {val1}", "Syntax", "Invalid variable name (hyphen)", 4),
        ("{keyword} = {val1}", "Syntax", "Using reserved keyword as variable name", 6),
        ("{var1} = '{val1}'\n{var2} = {var1} + {val2}", "Semantic", "Type mismatch in addition (str + int)", 8),
        ("{var1} = {val1}\n{var1}()", "Semantic", "Attempting to call a non-callable variable", 7),
    ],
    "loops": [
        ("while True:\n    print({val1})", "Logic", "Infinite loop due to missing break condition", 9),
        ("for {var1} in range({val1}):\nprint({var1})", "Syntax", "Indentation error in for loop", 4),
        ("for {var1} in {val1}:\n    print({var1})", "Semantic", "Iterating over non-iterable object", 7),
        ("while {var1} < {val1}:\n    print({var1})", "Logic", "Infinite loop due to missing increment", 9),
        ("for {var1} in range(len({var2})):\n    print({var2}[{var1}+1])", "Logic", "Index out of bounds (off-by-one)", 8),
        ("for {var1} in range({val1}, 0):\n    print({var1})", "Logic", "Loop body never executes (wrong range direction)", 5),
        ("while {var1} != 0:\n    {var1} -= 2", "Logic", "Infinite loop due to step skipping condition", 8),
    ],
    "conditionals": [
        ("if {var1} = {val1}:", "Syntax", "Using = instead of == in if-statement", 7),
        ("if {var1} > {val1}\n    print({var1})", "Syntax", "Missing colon in if-statement", 3),
        ("if {var1} > {val1} and < {val2}:", "Syntax", "Incomplete logical expression", 6),
        ("if {var1} == {val1} or {val2}:", "Logic", "Always true condition (truthy check on constant)", 8),
        ("if {val1} < {var1} < {val2} == False:", "Logic", "Confusing comparison chaining with boolean check", 7),
        ("if {var1} == {val1}:\n{var2} = {val2}", "Syntax", "Missing indentation after if-statement", 4),
        ("if {var1} is {val1}:", "Semantic", "Using 'is' for value comparison instead of '=='", 5),
    ],
    "functions": [
        ("def {var1}:\n    print({val1})", "Syntax", "Missing parameters/parentheses in definition", 5),
        ("def {var1}():\nprint({val1})", "Syntax", "Indentation error in function body", 4),
        ("{var1}", "Semantic", "Referencing function without calling it", 6),
        ("def {var1}(a, b):\n    return a + b\n{var1}({val1})", "Semantic", "Missing required positional argument", 7),
        ("def {var1}():\n    local_v = {val1}\n{var1}()\nprint(local_v)", "Semantic", "Accessing local variable outside scope", 8),
        ("def {var1}(x):\n    x = x + 1\n{var1}({val1})\nprint(x)", "Semantic", "Misunderstanding pass-by-value/local reassignment", 7),
        ("return {val1}", "Syntax", "'return' outside function", 9),
    ],
    "data_types": [
        ("{var1} = [{val1}, {val2}]\nprint({var1}[2])", "Logic", "Index out of range", 8),
        ("{var1} = 'hello'\n{var1}[0] = 'H'", "Semantic", "Attempting to mutate immutable string", 7),
        ("{var1} = {{'a': 1}}\nprint({var1}['b'])", "Semantic", "Accessing non-existent dictionary key", 6),
        ("{var1} = '{val1}' * '{val2}'", "Semantic", "Invalid operand types for multiplication (str * str)", 9),
        ("{var1}.append({val1}, {val2})", "Semantic", "Incorrect number of arguments for list.append", 5),
        ("{var1} = ({val1}, {val2})\n{var1}.append({val3})", "Semantic", "Attempting to use list method on tuple", 7),
        ("{var1} = {val1} + '{val2}'", "Semantic", "Unsupported operand types (int + str)", 8),
    ]
}

def generate_data(count):
    rows = []
    concepts = list(TEMPLATES.keys())
    
    for i in range(count):
        concept = random.choice(concepts)
        template, error_type, label, base_intensity = random.choice(TEMPLATES[concept])
        
        # Fill placeholders
        var1, var2, var3 = random.sample(VAR_NAMES, 3)
        val1, val2, val3 = random.sample(VALUES, 3)
        keyword = random.choice(KEYWORDS)
        
        snippet = template.format(
            var1=var1, var2=var2, var3=var3,
            val1=val1, val2=val2, val3=val3,
            keyword=keyword
        )
        
        # Add some random "noise" to intensity
        intensity = max(1, min(10, base_intensity + random.randint(-1, 1)))
        
        rows.append({
            "code_snippet": snippet.replace("\n", "\\n"), # Keep it on one line for CSV simplicity if needed, but user said clean CSV
            "concept_id": concept,
            "error_type": error_type,
            "misconception_label": label,
            "struggle_intensity": intensity
        })
    
    return rows

def main():
    print(f"Generating {TARGET_COUNT} records...")
    data = generate_data(TARGET_COUNT)
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["code_snippet", "concept_id", "error_type", "misconception_label", "struggle_intensity"])
        writer.writeheader()
        writer.writerows(data)
    
    print(f"Successfully saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
