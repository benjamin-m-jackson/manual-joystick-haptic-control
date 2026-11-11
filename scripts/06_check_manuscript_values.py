r"""Check every number reported in the manuscript text against output/results_values.csv.

Each entry below quotes a passage of the manuscript with each reported number replaced by @. The
script formats the computed values as printed (decimals, sign, words) and checks that the passage
appears verbatim in the manuscript source. Formats:
  1f/2f/3f/4f  fixed decimals       d     integer          neg1f/neg2f  value with its sign flipped
  lt0.001      printed "0.001", checks value < 0.001     le0.009  value <= 0.009    gt0.16  value > 0.16
  word/Word    number as an English word                  sci1     a.b\times10^{-n}
A format starting with "!" is a check only (no @ in the passage), e.g. "!ns" = p >= 0.05.
Table I is checked separately (04_questionnaire_table1.py regenerates it and compares it byte for byte).

Writes output/manuscript_value_audit.csv. Needs the manuscript sources (../../sections/), i.e. it
runs inside the manuscript repository; in the public deposit the audit output is included instead."""
import math
import os
from decimal import ROUND_HALF_UP, Decimal
import re
import sys

import pandas as pd

from common import OUT, PKG

SECTIONS = os.path.join(PKG, "..", "sections")
WORDS = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen " \
        "sixteen seventeen".split()

T, HON, HOFF, J, M = "time.lmm.", "Haptic On", "Haptic Off", "Joystick", "Manual"
F = "force.lmm."
C, P = "cath.gee_md.", "prolapse.gee_md."
CHECKS = [
    # ---------------------------------------------------------------- Abstract
    ("abstract", r"significantly faster (mean @ s) than all robotic modalities: haptic on (@ s), haptic off (@ s), and joystick (@ s) (all p$<$@)",
     [("time.Manual.mean", "1f"), ("time.Haptic On.mean", "1f"), ("time.Haptic Off.mean", "1f"),
      ("time.Joystick.mean", "1f"), (T + "Haptic On_vs_Manual.p", "lt0.001"), (T + "Haptic Off_vs_Manual.p", "!lt0.001"),
      (T + "Joystick_vs_Manual.p", "!lt0.001")]),
    ("abstract", r"when compared directly (p$<$@ and p=@)", [(T + "Haptic On_vs_Joystick.p", "lt0.001"), (T + "Haptic Off_vs_Joystick.p", "3f")]),
    ("abstract", r"when the @ navigations stopped at the eight-minute limit were treated as censored (p=@)",
     [("time.all.n_stopped_at_limit", "d"), ("time.cox.Haptic Off_vs_Joystick.p", "3f")]),
    ("abstract", r"more prolapse events than manual navigation (p$\leq$@, with small-sample-corrected standard errors)",
     [(P + "Haptic On_vs_Manual.p", "le0.009"), (P + "Haptic Off_vs_Manual.p", "!le0.009"), (P + "Joystick_vs_Manual.p", "!le0.009")]),
    ("abstract", r"prolapse did not differ significantly between the robotic controllers",
     [(P + "Haptic On_vs_Joystick.p", "!ns"), (P + "Haptic Off_vs_Joystick.p", "!ns"), (P + "Haptic Off_vs_Haptic On.p", "!ns")]),
    ("abstract", r"Navigation time, incorrect catheterizations, and prolapse events did not differ significantly by experience",
     [(T + "expert_vs_novice.lr_p", "!ns"), (C + "expert_vs_novice.p", "!ns"), (P + "expert_vs_novice.p", "!ns")]),
    ("abstract", r"System Usability Scale scores averaged @ (0--100 scale)", [("q.sus_score.all.mean", "1f")]),
    # ---------------------------------------------------------------- Methods
    ("methods", r"Of the 72 planned navigations, @ were performed", [("time.all.n", "d")]),
    ("methods", r"Force logs were available for @ of the @ navigations", [("force.n_navigations", "d"), ("time.all.n", "d")]),
    ("methods", r"@ navigations reached the limit: joystick @ of @, haptic off @ of @, haptic on @ of @, and manual @ of @.",
     [("time.all.n_stopped_at_limit", "Word"), ("time.Joystick.n_stopped_at_limit", "d"), ("time.Joystick.n", "d"),
      ("time.Haptic Off.n_stopped_at_limit", "d"), ("time.Haptic Off.n", "d"), ("time.Haptic On.n_stopped_at_limit", "d"),
      ("time.Haptic On.n", "d"), ("time.Manual.n_stopped_at_limit", "d"), ("time.Manual.n", "d")]),
    # ---------------------------------------------------------------- Results IV-A1 time
    ("results", r"mean completion time of $@\pm@$\,s ($n=@$ navigations)", [("time.Manual.mean", "1f"), ("time.Manual.sd", "1f"), ("time.Manual.n", "d")]),
    ("results", r"haptic on averaged $@\pm@$\,s ($n=@$), haptic off $@\pm@$\,s ($n=@$), and joystick navigation $@\pm@$\,s ($n=@$)",
     [(f"time.{c}.{s}", f) for c in (HON, HOFF, J) for s, f in (("mean", "1f"), ("sd", "1f"), ("n", "d"))]),
    ("results", r"(likelihood-ratio test, $\chi^2(3) = @$, $p = @$)", [(T + "omnibus.chi2", "2f"), (T + "omnibus.p", "sci1")]),
    ("results", r"faster than each robotic controller ($p < @$ for all three)",
     [(T + "Haptic On_vs_Manual.p", "lt0.001"), (T + "Haptic Off_vs_Manual.p", "!lt0.001"), (T + "Joystick_vs_Manual.p", "!lt0.001")]),
    ("results", r"(haptic on: $@$\,min, $p<@$; haptic off: $@$\,min, $p=@$)",
     [(T + "Haptic On_vs_Joystick.coef_min", "2f"), (T + "Haptic On_vs_Joystick.p", "lt0.001"),
      (T + "Haptic Off_vs_Joystick.coef_min", "2f"), (T + "Haptic Off_vs_Joystick.p", "3f")]),
    ("results", r"was not significant ($+@$\,min with haptic off, $p=@$)", [(T + "Haptic Off_vs_Haptic On.coef_min", "2f"), (T + "Haptic Off_vs_Haptic On.p", "3f")]),
    ("results", r"navigation was on average $@$\,min faster on the repeat attempt, which was not significant ($p=@$)",
     [(T + "repeat.coef_min", "neg2f"), (T + "repeat.p", "3f")]),
    ("results", r"faster than the joystick (hazard ratio @, 95\% CI @--@, $p=@$)",
     [("time.cox.Haptic On_vs_Joystick." + k, f) for k, f in (("hr", "1f"), ("hr_lo", "1f"), ("hr_hi", "1f"), ("p", "3f"))]),
    ("results", r"not significant in this analysis (hazard ratio @, 95\% CI @--@, $p=@$), nor was the difference between haptic on and haptic off ($p=@$)",
     [("time.cox.Haptic Off_vs_Joystick." + k, f) for k, f in (("hr", "1f"), ("hr_lo", "1f"), ("hr_hi", "1f"), ("p", "3f"))]
     + [("time.cox.Haptic Off_vs_Haptic On.p", "2f")]),
    # ---------------------------------------------------------------- IV-A2 force
    ("results", r"lowest maximum forces across all sensors ($@\pm@$\,mN)", [("force.Manual.mean", "1f"), ("force.Manual.sd", "1f")]),
    ("results", r"(haptic on $@\pm@$\,mN; haptic off $@\pm@$\,mN; joystick $@\pm@$\,mN; means $\pm$ SD over $n=@$, $@$, and $@$ sensor readings)",
     [("force.Haptic On.mean", "1f"), ("force.Haptic On.sd", "1f"), ("force.Haptic Off.mean", "2f"), ("force.Haptic Off.sd", "2f"),
      ("force.Joystick.mean", "2f"), ("force.Joystick.sd", "2f"), ("force.Haptic On.n", "d"), ("force.Haptic Off.n", "d"), ("force.Joystick.n", "d")]),
    ("results", r"with maxima of @ and @\,mN", [("force.Haptic Off.max", "1f"), ("force.Joystick.max", "1f")]),
    ("results", r"(likelihood-ratio test, $\chi^2(3)=@$, $p<@$): each robotic controller", [(F + "omnibus.chi2", "2f"), (F + "omnibus.p", "lt0.0001")]),
    ("results", r"(haptic on $+@$\,mN; haptic off $+@$\,mN; joystick $+@$\,mN; all $p<@$",
     [(F + "Haptic On_vs_Manual.coef_mN", "1f"), (F + "Haptic Off_vs_Manual.coef_mN", "1f"), (F + "Joystick_vs_Manual.coef_mN", "1f"),
      (F + "Haptic On_vs_Manual.p", "lt0.001"), (F + "Haptic Off_vs_Manual.p", "!lt0.001"), (F + "Joystick_vs_Manual.p", "!lt0.001")]),
    ("results", r"with no significant differences among the three robotic controllers",
     [(F + "Haptic On_vs_Joystick.p", "!ns"), (F + "Haptic Off_vs_Joystick.p", "!ns"), (F + "Haptic Off_vs_Haptic On.p", "!ns")]),
    ("results", r"Peak forces were also lower on the repeat attempt ($@$\,mN, $p<@$)", [(F + "repeat.coef_mN", "1f"), (F + "repeat.p", "lt0.001")]),
    ("results", r"Both force sensitivity analyses (Section~\ref{sec:statistics}) gave the same conclusions.",
     [(f"force.{s}.{c}_vs_Manual.p", "!lt0.001") for s in ("window", "covered") for c in (HON, HOFF, J)]
     + [(f"force.{s}.{k}", "!ns") for s in ("window", "covered") for k in ("Haptic On_vs_Joystick.p", "Haptic Off_vs_Joystick.p", "Haptic Off_vs_Haptic On.p")]
     + [(f"force.{s}.repeat.p", "!lt0.001") for s in ("window", "covered")]),
    # ---------------------------------------------------------------- IV-A3 catheterizations
    ("results", r"with a mean of $@\pm@$ incorrect catheterizations per navigation", [("cath.Manual.mean", "2f"), ("cath.Manual.sd", "2f")]),
    ("results", r"haptic on showed the fewest errors ($@\pm@$), followed by joystick navigation ($@\pm@$), while haptic off yielded the highest mean rate at $@\pm@$.",
     [(f"cath.{c}.{s}", "2f") for c in (HON, J, HOFF) for s in ("mean", "sd")]),
    ("results", r"(Wald test, $\chi^2(3) = @$, $p = @$). In pairwise contrasts, joystick navigation had more incorrect catheterizations than manual control ($p = @$)",
     [(C + "omnibus.chi2", "2f"), (C + "omnibus.p", "2f"), (C + "Joystick_vs_Manual.p", "3f")]),
    ("results", r"neither haptic mode differed significantly from manual, nor from the joystick",
     [(C + f"{c}_vs_{r}.p", "!ns") for c in (HON, HOFF) for r in (M, J)]),
    ("results", r"Incorrect catheterizations were lower on the repeat attempt ($p=@$)", [(C + "repeat.p", "3f")]),
    # ---------------------------------------------------------------- IV-A4 prolapse
    ("results", r"at only $@\pm@$ events per navigation", [("prolapse.Manual.mean", "2f"), ("prolapse.Manual.sd", "2f")]),
    ("results", r"with $@\pm@$ events per navigation.", [("prolapse.Joystick.mean", "2f"), ("prolapse.Joystick.sd", "2f")]),
    ("results", r"\textit{haptic off} yielded $@\pm@$ events, and \textit{haptic on} showed a lower mean of $@\pm@$ events",
     [("prolapse.Haptic Off.mean", "2f"), ("prolapse.Haptic Off.sd", "2f"), ("prolapse.Haptic On.mean", "2f"), ("prolapse.Haptic On.sd", "2f")]),
    ("results", r"(Wald test, $\chi^2(3) = @$, $p = @$). All three robotic", [(P + "omnibus.chi2", "2f"), (P + "omnibus.p", "4f")]),
    ("results", r"(haptic on $p=@$; haptic off $p=@$; joystick $p<@$)",
     [(P + "Haptic On_vs_Manual.p", "3f"), (P + "Haptic Off_vs_Manual.p", "3f"), (P + "Joystick_vs_Manual.p", "lt0.001")]),
    ("results", r"neither haptic on ($p=@$) nor haptic off ($p=@$) differed significantly. The trial-repeat effect was not significant ($p=@$)",
     [(P + "Haptic On_vs_Joystick.p", "3f"), (P + "Haptic Off_vs_Joystick.p", "2f"), (P + "repeat.p", "2f")]),
    # ---------------------------------------------------------------- IV-B novices vs experts
    ("results", r"were accounted for (Section~\ref{sec:statistics}; $p=@$)", [(T + "expert_vs_novice.lr_p", "3f")]),
    ("results", r"novices averaged $@\pm@$\,s and experts $@\pm@$\,s",
     [("time.Novice.mean", "1f"), ("time.Novice.sd", "1f"), ("time.Expert.mean", "1f"), ("time.Expert.sd", "1f")]),
    ("results", r"(novice mean $@\pm@$\,mN vs.\ expert mean $@\pm@$\,mN",
     [("force.Novice.mean", "1f"), ("force.Novice.sd", "1f"), ("force.Expert.mean", "1f"), ("force.Expert.sd", "1f")]),
    ("results", r"(experts $@$\,mN lower, $p=@$)", [(F + "expert_vs_novice.coef_mN", "neg1f"), (F + "expert_vs_novice.lr_p", "4f")]),
    ("results", r"significant at sensors 1, 2, and 5 ($p_{\mathrm{adj}} = @$, $@$, $@$) but not at sensors 3, 4, or 6--8",
     [(F + f"sensor{s}.expert_vs_novice.p_bonf", "3f") for s in (1, 2, 5)]
     + [(F + f"sensor{s}.expert_vs_novice.p_bonf", "!ns") for s in (3, 4, 6, 7, 8)]),
    ("results", r"averaging $@\pm@$ versus $@\pm@$ errors per navigation",
     [("cath.Novice.mean", "2f"), ("cath.Novice.sd", "2f"), ("cath.Expert.mean", "2f"), ("cath.Expert.sd", "2f")]),
    ("results", r"this difference was not significant ($p=@$), unlike the nominally significant result", [(C + "expert_vs_novice.p", "2f")]),
    ("results", r"averaging $@\pm@$ versus $@\pm@$ events per navigation",
     [("prolapse.Novice.mean", "2f"), ("prolapse.Novice.sd", "2f"), ("prolapse.Expert.mean", "2f"), ("prolapse.Expert.sd", "2f")]),
    ("results", r"Adjusting for controller type and trial repeat (Section~\ref{sec:statistics}), this difference was not significant ($p=@$).",
     [(P + "expert_vs_novice.p", "3f")]),
    # ---------------------------------------------------------------- IV-C questionnaire
    ("results", r"@ participants chose haptic on, @ haptic off, and @ the joystick",
     [("q.preferred.all.H", "word"), ("q.preferred.all.O", "word"), ("q.preferred.all.J", "word")]),
    ("results", r"most intuitive control method (mean $@ \pm @$)", [("q.intuitive_manual.all.mean", "2f"), ("q.intuitive_manual.all.sd", "2f")]),
    ("results", r"rated equally ($@$ each) and somewhat higher than the joystick ($@$), but not significantly (paired $t$-test, device-mimicking average vs.\ joystick, $p = @$)",
     [("q.intuitive_haptic_on.all.mean", "2f"), ("q.intuitive_joystick.all.mean", "2f"),
      ("q.intuitive.device_mimicking_vs_joystick.p", "2f")]),
    ("results", r"(means $@$--$@$)", [("q.realism_vessel_image.all.mean", "2f"), ("q.realism_device_image.all.mean", "2f")]),
    ("results", r"vascular anatomy higher than novices ($@ \pm @$ vs.\ $@ \pm @$)",
     [("q.realism_vessel_image.expert.mean", "2f"), ("q.realism_vessel_image.expert.sd", "2f"),
      ("q.realism_vessel_image.novice.mean", "2f"), ("q.realism_vessel_image.novice.sd", "2f")]),
    ("results", r"teaching tool ($@ \pm @$) and practice tool ($@ \pm @$)",
     [("q.training_teaching_tool.all.mean", "2f"), ("q.training_teaching_tool.all.sd", "2f"),
      ("q.training_practice_tool.all.mean", "2f"), ("q.training_practice_tool.all.sd", "2f")]),
    ("results", r"(paired $t$-tests, all $p>@$)", [("q.tlx.haptic_on_vs_off.min_p", "gt0.16")]),
    ("results", r"on Performance ($@\pm@$ vs.\ $@\pm@$), but not significantly ($p=@$)",
     [("q.tlx_performance.device_mimicking.mean", "2f"), ("q.tlx_performance.device_mimicking.sd", "2f"),
      ("q.tlx_joystick_performance.all.mean", "2f"), ("q.tlx_joystick_performance.all.sd", "2f"),
      ("q.tlx_performance.device_mimicking_vs_joystick.p", "2f")]),
    ("results", r"The overall SUS score was $@ \pm @$ ($n=@$)", [("q.sus_score.all.mean", "1f"), ("q.sus_score.all.sd", "1f"), ("q.sus_score.all.n", "d")]),
    ("results", r"Experts and novices scored similarly ($@ \pm @$ vs.\ $@ \pm @$)",
     [("q.sus_score.expert.mean", "1f"), ("q.sus_score.expert.sd", "1f"), ("q.sus_score.novice.mean", "1f"), ("q.sus_score.novice.sd", "1f")]),
    ("results", r"(RM-ANOVA, $F(@,@)=@$, $p=@$)",
     [("q.tlx_effort.rm_anova_robotic.expert." + k, f) for k, f in (("df1", "d"), ("df2", "d"), ("F", "2f"), ("p", "2f"))]),
    # ---------------------------------------------------------------- Discussion
    ("discussion", r"the difference between them was not significant ($p=@$; Section~\ref{subsubsec:cv_time})", [(T + "Haptic Off_vs_Haptic On.p", "3f")]),
    ("discussion", r"only the haptic-on advantage over the joystick remained significant",
     [("time.cox.Haptic On_vs_Joystick.p", "!lt0.05"), ("time.cox.Haptic Off_vs_Joystick.p", "!ns")]),
    ("discussion", r"(haptic on vs.\ joystick prolapse, $p=@$;", [(P + "Haptic On_vs_Joystick.p", "3f")]),
    ("discussion", r"differed only in an exploratory joystick-versus-manual contrast ($p=@$; overall $p=@$)",
     [(C + "Joystick_vs_Manual.p", "3f"), (C + "omnibus.p", "2f")]),
    ("discussion", r"similar between novices and experts ($p=@$)", [(T + "expert_vs_novice.lr_p", "3f")]),
    ("discussion", r"both overall ($p=@$)", [(F + "expert_vs_novice.lr_p", "4f")]),
    ("discussion", r"Prolapse events ($p=@$) and incorrect catheterizations ($p=@$) did not differ significantly",
     [(P + "expert_vs_novice.p", "3f"), (C + "expert_vs_novice.p", "2f")]),
    ("discussion", r"(@--@\,mN by controller)", [("force.Manual.mean", "1f"), ("force.Haptic On.mean", "1f")]),
    ("discussion", r"the highest single reading (@\,mN)", [("force.all.max", "0f")]),
]


def render(v, fmt):
    """Return (text or None, ok)."""
    if fmt in ("d",):
        return str(int(round(v))), True
    m = re.fullmatch(r"(neg)?(\d)f", fmt)
    if m:  # conventional half-up rounding of the value's shortest decimal representation
        x = Decimal(repr(float(-v if m.group(1) else v)))
        q = Decimal(1).scaleb(-int(m.group(2)))
        return str(x.quantize(q, rounding=ROUND_HALF_UP)), True
    m = re.fullmatch(r"(lt|le|gt)([\d.]+)", fmt)
    if m:
        th = float(m.group(2))
        ok = {"lt": v < th, "le": v <= th, "gt": v > th}[m.group(1)]
        return m.group(2), ok
    if fmt in ("word", "Word"):
        w = WORDS[int(round(v))]
        return (w.capitalize() if fmt == "Word" else w), True
    if fmt == "sci1":
        e = math.floor(math.log10(v))
        return f"{v / 10 ** e:.1f}\\times10^{{{e}}}", True
    if fmt == "ns":
        return None, v >= 0.05
    if fmt == "eq0":
        return None, v == 0
    raise ValueError(fmt)


def norm(s):
    return re.sub(r"\s+", " ", s)


def main():
    if not os.path.isdir(SECTIONS):
        sys.exit("manuscript sources (../../sections) not found; this check runs inside the manuscript repository")
    res = pd.read_csv(os.path.join(OUT, "results_values.csv")).set_index("key")["value"]
    text = {f: norm(open(os.path.join(SECTIONS, f + ".tex"), encoding="utf-8").read())
            for f in {c[0] for c in CHECKS}}
    rows, bad = [], 0
    for file, template, items in CHECKS:
        parts, problems = [], []
        for key, fmt in items:
            if key not in res.index:
                problems.append(f"missing key {key}")
                continue
            s, ok = render(res[key], fmt.lstrip("!"))
            if not ok:
                problems.append(f"{key} = {res[key]:.4g} fails {fmt}")
            if not fmt.startswith("!"):
                parts.append(s)
        pieces = template.split("@")
        if len(pieces) - 1 != len(parts):
            problems.append(f"template has {len(pieces) - 1} @ but {len(parts)} printed values")
            expected = template
        else:
            expected = pieces[0] + "".join(p + q for p, q in zip(parts, pieces[1:]))
        if not problems and norm(expected) not in text[file]:
            problems.append("passage not found as expected in the manuscript")
        status = "OK" if not problems else "MISMATCH"
        bad += status != "OK"
        rows.append({"file": f"sections/{file}.tex", "expected_passage": expected, "status": status,
                     "problems": "; ".join(problems), "keys": "; ".join(k for k, _ in items)})
        if problems:
            print(f"MISMATCH [{file}] {expected[:110]}\n          {'; '.join(problems)}")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "manuscript_value_audit.csv"), index=False, lineterminator="\n")
    n_values = sum(len(c[2]) for c in CHECKS)
    print(f"06_check_manuscript_values: {len(CHECKS) - bad}/{len(CHECKS)} passages OK "
          f"({n_values} values checked) -> output/manuscript_value_audit.csv")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
