"""Run every analysis in order: python run_all.py (from this folder).
06_check_manuscript_values.py runs only inside the manuscript repository (it reads the .tex sources)."""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)
SCRIPTS = ["01_time_models.py", "02_count_models.py", "03_force_models.py", "04_questionnaire_table1.py",
           "05_figures.py", "06_check_manuscript_values.py"]

for s in SCRIPTS:
    if s.startswith("06") and not os.path.isdir(os.path.join(HERE, "..", "..", "sections")):
        print("06_check_manuscript_values: skipped (manuscript sources not present); "
              "see output/manuscript_value_audit.csv")
        continue
    try:
        runpy.run_path(os.path.join(HERE, s), run_name="__main__")
    except SystemExit as e:
        if e.code not in (0, None):
            sys.exit(f"{s} failed (exit {e.code})")
print("all analyses complete")
