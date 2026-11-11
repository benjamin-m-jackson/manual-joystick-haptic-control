"""Peak vessel-wall force (Sections IV-A2, IV-B2): descriptives and linear mixed models with a participant
random intercept and a sensor-site variance component.

Primary (key 'force.lmm'): peak force over the whole log, controller + repeat (+ expertise for the
novice-expert comparison); per-sensor novice-expert models with controller + repeat, Bonferroni x8.
Sensitivity analyses:
  'force.window'  peak within the navigation window only (log start to t_ICA + 5 s);
  'force.covered' excluding the three navigations whose force log ended before the navigation did;
  'force.r1spec'  the R1 model specification (no repeat term; per-sensor models without controller),
                  for the record of what changed between R1 and R2."""
import pandas as pd

from common import (CONTROLLERS, EXP_TERM, EXPERTISE, Results, ctrl, ctrl_term, describe, load_force,
                    load_navigation, lmm, lr_test)

R = Results("03_force_models")
VC = {"sensor": "0 + C(sensor)"}
EXPF = "C(expertise, Treatment('Novice'))"


def fit_all(prefix, d, repeat=True, per_sensor_controller=True):
    rep = " + repeat" if repeat else ""
    full = lmm(f"force ~ {ctrl('Manual')}{rep}", d, VC)
    reduced = lmm(f"force ~ 1{rep}", d, VC)
    lr, df, p = lr_test(full, reduced)
    R.add(f"{prefix}.omnibus.chi2", lr), R.add(f"{prefix}.omnibus.df", df), R.add(f"{prefix}.omnibus.p", p)
    if repeat:
        R.add(f"{prefix}.repeat.coef_mN", full.params["repeat"]), R.add(f"{prefix}.repeat.p", full.pvalues["repeat"])
    for ref in ["Manual", "Joystick", "Haptic On"]:
        m = full if ref == "Manual" else lmm(f"force ~ {ctrl(ref)}{rep}", d, VC)
        for lev in CONTROLLERS:
            if lev != ref:
                R.add(f"{prefix}.{lev}_vs_{ref}.coef_mN", m.params[ctrl_term(lev, ref)])
                R.add(f"{prefix}.{lev}_vs_{ref}.p", m.pvalues[ctrl_term(lev, ref)])
    ne = lmm(f"force ~ {ctrl('Manual')}{rep} + {EXPF}", d, VC)
    lr, df, p = lr_test(ne, full)
    R.add(f"{prefix}.expert_vs_novice.coef_mN", ne.params[EXP_TERM])
    R.add(f"{prefix}.expert_vs_novice.p", ne.pvalues[EXP_TERM]), R.add(f"{prefix}.expert_vs_novice.lr_p", p)
    # per sensor: participant random intercept only (one sensor), Bonferroni across the eight sensors
    fe = f"{ctrl('Manual')}{rep} + {EXPF}" if per_sensor_controller else EXPF
    for s in range(1, 9):
        sub = d[d.sensor == s]
        try:
            m = lmm(f"force ~ {fe}", sub)
            coef, p = m.params[EXP_TERM], m.pvalues[EXP_TERM]
        except (RuntimeError, ValueError, pd.errors.PerformanceWarning, Exception) as e:  # noqa: B014
            print(f"  {prefix} sensor {s}: model failed ({type(e).__name__}); recorded as missing")
            coef = p = float("nan")
        R.add(f"{prefix}.sensor{s}.expert_vs_novice.coef_mN", coef)
        R.add(f"{prefix}.sensor{s}.expert_vs_novice.p", p)
        R.add(f"{prefix}.sensor{s}.expert_vs_novice.p_bonf", min(1.0, 8 * p) if p == p else p)
    return full, ne


# ---- primary: whole-log peaks
f = load_force("peak_force_mN")
describe(R, "force.all", f.force)
for c in CONTROLLERS:
    describe(R, f"force.{c}", f.loc[f.controller == c, "force"])
for e in EXPERTISE:
    describe(R, f"force.{e}", f.loc[f.expertise == e, "force"])
for s in range(1, 9):
    describe(R, f"force.sensor{s}", f.loc[f.sensor == s, "force"])
R.add("force.n_navigations", len(f.drop_duplicates(["participant", "controller", "repeat"])))
full, ne = fit_all("force.lmm", f)
R.log("LMM force ~ controller (ref Manual) + repeat; participant RE + sensor VC", full.summary())
R.log("LMM force ~ controller + repeat + expertise", ne.summary())

# ---- sensitivity 1: navigation-window peaks
w = load_force("peak_force_nav_window_mN")
describe(R, "force.window.all", w.force)
for c in CONTROLLERS:
    describe(R, f"force.window.{c}", w.loc[w.controller == c, "force"])
for e in EXPERTISE:
    describe(R, f"force.window.{e}", w.loc[w.expertise == e, "force"])
fit_all("force.window", w)

# ---- sensitivity 2: only navigations whose force log covers the whole navigation
nav = load_navigation()
covered = nav.loc[nav.force_log_covers_navigation == 1, ["participant", "controller", "repeat"]]
fc = f.merge(covered, on=["participant", "controller", "repeat"])
R.add("force.covered.n_navigations", len(covered))
fit_all("force.covered", fc)

# ---- record: R1 model specification on the (corrected) R2 data
fit_all("force.r1spec", f, repeat=False, per_sensor_controller=False)

R.save()
