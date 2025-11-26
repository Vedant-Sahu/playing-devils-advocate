
from pathlib import Path
import sys
import dspy

# Make sure the local src/ package root is importable so that cloudpickle can
# locate the `dspy_pipeline` module when loading the compiled program.
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Path to the GEPA-compiled program (directory created by compile_dspy_program.py)
compiled_dir = "results\dspy_gepa_from_agents"  # or whatever you passed to --output

program = dspy.load(compiled_dir)

print("\n=== TEACHER PROMPT ===\n")
print(program.teacher.signature.instructions)

print("\n=== FINAL-ANSWER PROMPT ===\n")
print(program.final_answerer.signature.instructions)

print("\n=== STUDENT PROMPTS BY PERSONA ===")
for persona, module in program.student_modules.items():
    print(f"\n--- Student persona: {persona} ---\n")
    print(module.signature.instructions)
