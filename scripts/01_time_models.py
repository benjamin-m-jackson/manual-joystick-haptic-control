"""Navigation time (Section IV-A1, IV-B1): descriptives, trials stopped at the eight-minute limit,
linear mixed model, and a censoring-aware Cox sensitivity analysis."""
import numpy as np
from statsmodels.duration.hazard_regression import PHReg

from common import (CONTROLLERS, EXP_TERM, EXPERTISE, LIMIT_S, ROBOTIC, Results, ctrl, ctrl_term, describe,
                    load_navigation, lmm, lr_test)

R = Results("01_time_models")
nav = load_navigation()

# ---- descriptives (s) and the eight-minute limit
describe(R, "time.all", nav.t_ICA_s)
for c in CONTROLLERS:
    s = nav[nav.controller == c]
    describe(R, f"time.{c}", s.t_ICA_s)
    R.add(f"time.{c}.n_stopped_at_limit", (s.reached_ICA == 0).sum(), "navigations stopped at 480 s")
    R.add(f"time.{c}.n_not_reaching_CCA", (s.t_CCA_s >= LIMIT_S).sum())
for e in EXPERTISE:
    describe(R, f"time.{e}", nav[nav.expertise == e].t_ICA_s)
R.add("time.all.n_stopped_at_limit", (nav.reached_ICA == 0).sum())
R.add("time.all.n_not_reaching_CCA", (nav.t_CCA_s >= LIMIT_S).sum())

# ---- linear mixed model: time (min) ~ controller + repeat, participant random intercept.
# Navigations stopped at the limit enter at 480 s (a lower bound on their completion time).
full = lmm(f"t_ICA_min ~ {ctrl('Manual')} + repeat", nav)
reduced = lmm("t_ICA_min ~ repeat", nav)
lr, df, p = lr_test(full, reduced)
R.add("time.lmm.omnibus.chi2", lr), R.add("time.lmm.omnibus.df", df), R.add("time.lmm.omnibus.p", p)
R.add("time.lmm.repeat.coef_min", full.params["repeat"]), R.add("time.lmm.repeat.p", full.pvalues["repeat"])
R.log("LMM time ~ controller (ref Manual) + repeat", full.summary())
for ref in ["Manual", "Joystick", "Haptic On"]:
    m = full if ref == "Manual" else lmm(f"t_ICA_min ~ {ctrl(ref)} + repeat", nav)
    for lev in CONTROLLERS:
        if lev != ref:
            R.add(f"time.lmm.{lev}_vs_{ref}.coef_min", m.params[ctrl_term(lev, ref)])
            R.add(f"time.lmm.{lev}_vs_{ref}.p", m.pvalues[ctrl_term(lev, ref)])

# novice vs expert, adjusted for controller and repeat
ne = lmm(f"t_ICA_min ~ {ctrl('Manual')} + repeat + C(expertise, Treatment('Novice'))", nav)
lr, df, p = lr_test(ne, full)
R.add("time.lmm.expert_vs_novice.lr_p", p), R.add("time.lmm.expert_vs_novice.coef_min", ne.params[EXP_TERM])
R.add("time.lmm.expert_vs_novice.p", ne.pvalues[EXP_TERM])
R.log("LMM time ~ controller + repeat + expertise", ne.summary())

# ---- sensitivity: Cox model treating navigations stopped at 480 s as censored.
# Robotic navigations only (no manual navigation reached the limit); stratified by participant
# (each participant has their own baseline hazard), controller + repeat as covariates.
# Hazard ratio > 1 = faster completion.
rob = nav[nav.controller.isin(ROBOTIC)].copy()
rob["controller"] = rob["controller"].cat.remove_unused_categories()
for ref in ["Joystick", "Haptic On"]:
    cox = PHReg.from_formula(f"t_ICA_s ~ {ctrl(ref)} + repeat", data=rob, status=rob["reached_ICA"].to_numpy(),
                             strata=rob["participant"].to_numpy(), ties="efron").fit()
    names = cox.model.exog_names
    for lev in ROBOTIC:
        if lev == ref:
            continue
        i = names.index(ctrl_term(lev, ref))
        lo, hi = np.exp(cox.conf_int()[i])
        R.add(f"time.cox.{lev}_vs_{ref}.hr", np.exp(cox.params[i]))
        R.add(f"time.cox.{lev}_vs_{ref}.hr_lo", lo), R.add(f"time.cox.{lev}_vs_{ref}.hr_hi", hi)
        R.add(f"time.cox.{lev}_vs_{ref}.p", cox.pvalues[i])
    if ref == "Joystick":
        i = names.index("repeat")
        R.add("time.cox.repeat.hr", np.exp(cox.params[i])), R.add("time.cox.repeat.p", cox.pvalues[i])
        R.log("Cox PH (robotic, strata = participant, 480 s censored), ref Joystick", cox.summary())
R.add("time.cox.n_events", rob.reached_ICA.sum()), R.add("time.cox.n", len(rob))

R.save()
