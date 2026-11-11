"""Incorrect catheterizations and prolapse events (Sections IV-A3, IV-A4, IV-B3, IV-B4): descriptives and
Poisson GEE (exchangeable working correlation, clustered by participant).

Every test is reported twice: with the Mancl & DeRouen (2001) bias-reduced covariance (key 'md',
primary in R2: small-sample correction for nine clusters) and with the robust sandwich used in R1
(key 'robust')."""
import numpy as np

from common import CONTROLLERS, EXP_TERM, EXPERTISE, Results, ctrl, ctrl_term, describe, gee, load_navigation, wald

R = Results("02_count_models")
nav = load_navigation()
COV = {"md": "bias_reduced", "robust": "robust"}

for outcome, short in [("incorrect_catheterizations", "cath"), ("prolapse_events", "prolapse")]:
    describe(R, f"{short}.all", nav[outcome])
    for c in CONTROLLERS:
        describe(R, f"{short}.{c}", nav.loc[nav.controller == c, outcome])
    for e in EXPERTISE:
        describe(R, f"{short}.{e}", nav.loc[nav.expertise == e, outcome])
    for key, cov in COV.items():
        full = gee(f"{outcome} ~ {ctrl('Manual')} + repeat", nav, cov)
        terms = [ctrl_term(c, "Manual") for c in CONTROLLERS[1:]]
        chi2, df, p = wald(full, terms)
        R.add(f"{short}.gee_{key}.omnibus.chi2", chi2), R.add(f"{short}.gee_{key}.omnibus.df", df)
        R.add(f"{short}.gee_{key}.omnibus.p", p)
        R.add(f"{short}.gee_{key}.repeat.irr", np.exp(full.params["repeat"]))
        R.add(f"{short}.gee_{key}.repeat.p", full.pvalues["repeat"])
        if key == "md":
            R.log(f"GEE {outcome} ~ controller (ref Manual) + repeat, Mancl-DeRouen covariance", full.summary())
        for ref in ["Manual", "Joystick", "Haptic On"]:
            m = full if ref == "Manual" else gee(f"{outcome} ~ {ctrl(ref)} + repeat", nav, cov)
            for lev in CONTROLLERS:
                if lev != ref:
                    R.add(f"{short}.gee_{key}.{lev}_vs_{ref}.irr", np.exp(m.params[ctrl_term(lev, ref)]))
                    R.add(f"{short}.gee_{key}.{lev}_vs_{ref}.p", m.pvalues[ctrl_term(lev, ref)])
        ne = gee(f"{outcome} ~ {ctrl('Manual')} + repeat + C(expertise, Treatment('Novice'))", nav, cov)
        R.add(f"{short}.gee_{key}.expert_vs_novice.irr", np.exp(ne.params[EXP_TERM]))
        R.add(f"{short}.gee_{key}.expert_vs_novice.p", ne.pvalues[EXP_TERM])
        if key == "md":
            R.log(f"GEE {outcome} ~ controller + repeat + expertise, Mancl-DeRouen covariance", ne.summary())

R.save()
