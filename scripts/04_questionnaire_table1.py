"""Questionnaire (Section IV-C, Table I): item summaries, SUS, and the tests quoted in the text.

Writes output/table1_qual_master.tex from table1_template.tex. Values are mean +/- SD (n-1) on the
1-10 scale. Novice-vs-expert comparisons use Welch's t-test; same-participant comparisons across
controllers use paired t-tests, or a one-way repeated-measures ANOVA for three controllers."""
import os
import re

import numpy as np
import pandas as pd
from scipy import stats

from common import HERE, OUT, Results, load_questionnaire

R = Results("04_questionnaire_table1")
q = load_questionnaire()
CTRL = ["manual", "haptic_on", "haptic_off", "joystick"]
ROB = ["haptic_on", "haptic_off", "joystick"]
DIMS = ["mental", "physical", "temporal", "performance", "effort", "frustration"]
ITEMS = (["realism_vessel_image", "realism_device_image", "realism_phantom"] + [f"intuitive_{c}" for c in CTRL]
         + ["training_teaching_tool", "training_practice_tool"] + [f"tlx_{c}_{d}" for c in CTRL for d in DIMS])
GROUPS = {"all": q, "novice": q[q.expertise == "Novice"], "expert": q[q.expertise == "Expert"]}

# ---- SUS: each 1-10 answer rescaled to the standard 0-4 item score, then x2.5
odd = [f"sus_{i:02d}" for i in (1, 3, 5, 7, 9)]
even = [f"sus_{i:02d}" for i in (2, 4, 6, 8, 10)]
q["sus_score"] = 2.5 * (((q[odd] - 1) * 4 / 9).sum(axis=1) + ((10 - q[even]) * 4 / 9).sum(axis=1))
GROUPS = {"all": q, "novice": q[q.expertise == "Novice"], "expert": q[q.expertise == "Expert"]}

# ---- item summaries + Welch novice vs expert
for item in ITEMS + ["sus_score"]:
    for g, d in GROUPS.items():
        x = d[item].dropna()
        R.add(f"q.{item}.{g}.mean", x.mean()), R.add(f"q.{item}.{g}.sd", x.std(ddof=1)), R.add(f"q.{item}.{g}.n", len(x))
    a, b = GROUPS["novice"][item].dropna(), GROUPS["expert"][item].dropna()
    t = stats.ttest_ind(a, b, equal_var=False)
    R.add(f"q.{item}.welch_novice_vs_expert.t", t.statistic), R.add(f"q.{item}.welch_novice_vs_expert.p", t.pvalue)

# ---- preferred robotic controller
PREF = {"Robot Controller (Haptics On)": "H", "Robot Controller (Haptics Off)": "O", "Joystick Controller": "J"}
pref_text = {}
for g, d in GROUPS.items():
    counts = d["most_intuitive_robotic_controller"].map(PREF).value_counts()
    for k in "HOJ":
        R.add(f"q.preferred.{g}.{k}", counts.get(k, 0))
    pref_text[g] = ", ".join(f"\\textbf{{{k}}} = {counts.get(k, 0)}" for k in "HOJ")

# ---- paired comparisons (all participants answered every robotic item)
q["intuitive_device_mimicking"] = q[["intuitive_haptic_on", "intuitive_haptic_off"]].mean(axis=1)
t = stats.ttest_rel(q["intuitive_device_mimicking"], q["intuitive_joystick"])
R.add("q.intuitive.device_mimicking_vs_joystick.p", t.pvalue), R.add("q.intuitive.device_mimicking_vs_joystick.t", t.statistic)
for dim in DIMS:
    t = stats.ttest_rel(q[f"tlx_haptic_on_{dim}"], q[f"tlx_haptic_off_{dim}"])
    R.add(f"q.tlx_{dim}.haptic_on_vs_off.p", t.pvalue)
R.add("q.tlx.haptic_on_vs_off.min_p", min(stats.ttest_rel(q[f"tlx_haptic_on_{d}"], q[f"tlx_haptic_off_{d}"]).pvalue
                                         for d in DIMS))
dm_perf = q[["tlx_haptic_on_performance", "tlx_haptic_off_performance"]].mean(axis=1)
R.add("q.tlx_performance.device_mimicking.mean", dm_perf.mean()), R.add("q.tlx_performance.device_mimicking.sd", dm_perf.std(ddof=1))
R.add("q.tlx_performance.device_mimicking_vs_joystick.p", stats.ttest_rel(dm_perf, q["tlx_joystick_performance"]).pvalue)


def rm_anova(wide):
    """One-way repeated-measures ANOVA; rows = participants, columns = conditions."""
    y = wide.to_numpy(float)
    n, k = y.shape
    grand = y.mean()
    ss_cond = n * ((y.mean(axis=0) - grand) ** 2).sum()
    ss_subj = k * ((y.mean(axis=1) - grand) ** 2).sum()
    ss_err = ((y - grand) ** 2).sum() - ss_cond - ss_subj
    df1, df2 = k - 1, (k - 1) * (n - 1)
    f = (ss_cond / df1) / (ss_err / df2)
    return f, df1, df2, stats.f.sf(f, df1, df2)


for g in ("all", "novice", "expert"):
    f, d1, d2, p = rm_anova(GROUPS[g][[f"tlx_{c}_effort" for c in ROB]])
    R.add(f"q.tlx_effort.rm_anova_robotic.{g}.F", f), R.add(f"q.tlx_effort.rm_anova_robotic.{g}.df1", d1)
    R.add(f"q.tlx_effort.rm_anova_robotic.{g}.df2", d2), R.add(f"q.tlx_effort.rm_anova_robotic.{g}.p", p)

# ---- Table I
tpl = open(os.path.join(HERE, "table1_template.tex"), encoding="utf-8").read()


def cell(m):
    col, g = m.group(1), m.group(2)
    if col == "preferred":
        return pref_text[g]
    x = GROUPS[g][col].dropna()
    return f"{x.mean():.2f} \\(\\pm\\) {x.std(ddof=1):.2f}"


table = re.sub(r"@@(\w+):(\w+)@@", cell, tpl)
tests = pd.DataFrame([r for r in R.rows if re.search(r"\.(p|t|F|df1|df2)$", r["key"])])[["key", "value"]]
R.log("Questionnaire tests (paired t, Welch t, repeated-measures ANOVA)", tests.to_string(index=False))
with open(os.path.join(OUT, "table1_qual_master.tex"), "w", encoding="utf-8", newline="") as fh:
    fh.write(table)
manuscript = os.path.join(HERE, "..", "..", "sections", "qual_master_table.tex")  # present only inside the repo
if os.path.exists(manuscript):
    same = open(manuscript, encoding="utf-8").read() == table
    print("Table I: output/table1_qual_master.tex is " + ("IDENTICAL to" if same else "DIFFERENT from")
          + " sections/qual_master_table.tex")
R.save()
