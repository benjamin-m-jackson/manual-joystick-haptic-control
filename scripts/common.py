"""Shared helpers for the analysis scripts. They read only the de-identified files in ../data."""
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.genmod.cov_struct as cov_struct
import statsmodels.genmod.families as families
from scipy import stats

warnings.filterwarnings("ignore")  # statsmodels convergence chatter; convergence is checked explicitly

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
DATA = os.path.join(PKG, "data")
OUT = os.path.join(PKG, "output")
SUMMARIES = os.path.join(OUT, "model_summaries")
FIGURES = os.path.join(OUT, "figures")
for d in (OUT, SUMMARIES, FIGURES):
    os.makedirs(d, exist_ok=True)

CONTROLLERS = ["Manual", "Haptic On", "Haptic Off", "Joystick"]
ROBOTIC = ["Haptic On", "Haptic Off", "Joystick"]
EXPERTISE = ["Novice", "Expert"]
LIMIT_S = 480


# ---------------------------------------------------------------------------- data
def load_navigation():
    d = pd.read_csv(os.path.join(DATA, "navigation_trials.csv"))
    d["controller"] = pd.Categorical(d["controller"], CONTROLLERS)
    d["t_ICA_min"] = d["t_ICA_s"] / 60.0
    return d


def load_force(column="peak_force_mN"):
    d = pd.read_csv(os.path.join(DATA, "force_peak_per_sensor.csv"))
    d["controller"] = pd.Categorical(d["controller"], CONTROLLERS)
    d["force"] = d[column]
    return d


def load_questionnaire():
    return pd.read_csv(os.path.join(DATA, "questionnaire_ratings.csv"))


# ---------------------------------------------------------------------------- results registry
class Results:
    """Collects every reported number under a stable key. Each script's rows are written to
    output/results_values.csv (replacing that script's previous rows) and are what
    06_check_manuscript_values.py compares with the manuscript."""

    def __init__(self, script):
        self.script, self.rows, self.text = script, [], []

    def add(self, key, value, description=""):
        self.rows.append({"key": key, "value": float(value), "description": description, "script": self.script})

    def log(self, title, obj):
        self.text.append(f"{'=' * 100}\n{title}\n{'=' * 100}\n{obj}\n")

    def save(self):
        path = os.path.join(OUT, "results_values.csv")
        new = pd.DataFrame(self.rows)
        if os.path.exists(path):
            old = pd.read_csv(path)
            new = pd.concat([old[old.script != self.script], new], ignore_index=True)
        new.to_csv(path, index=False, float_format="%.10g", lineterminator="\n")
        with open(os.path.join(SUMMARIES, f"{self.script}.txt"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(self.text))
        print(f"{self.script}: {len(self.rows)} values -> output/results_values.csv; "
              f"model output -> output/model_summaries/{self.script}.txt")


def describe(R, prefix, s):
    """mean, SD (n-1), n and max of a series."""
    R.add(f"{prefix}.mean", s.mean())
    R.add(f"{prefix}.sd", s.std(ddof=1))
    R.add(f"{prefix}.n", s.count())
    R.add(f"{prefix}.max", s.max())


# ---------------------------------------------------------------------------- models
def ctrl(ref):
    return f"C(controller, Treatment('{ref}'))"


def ctrl_term(level, ref):
    return f"{ctrl(ref)}[T.{level}]"


EXP_TERM = "C(expertise, Treatment('Novice'))[T.Expert]"


def lmm(formula, data, vc=None):
    """Linear mixed model, participant random intercept (optional variance component), ML fit."""
    res = smf.mixedlm(formula, data=data, groups=data["participant"], vc_formula=vc).fit(reml=False, method="powell")
    if not res.converged:
        raise RuntimeError(f"MixedLM did not converge: {formula}")
    return res


def lr_test(full, reduced):
    lr = 2.0 * (full.llf - reduced.llf)
    df = int(round(full.df_modelwc - reduced.df_modelwc))
    return lr, df, stats.chi2.sf(lr, df)


def gee(formula, data, cov_type):
    """Poisson GEE, exchangeable working correlation, clustered by participant.
    cov_type 'bias_reduced' = Mancl & DeRouen (2001) small-sample correction; 'robust' = sandwich."""
    model = smf.gee(formula, groups="participant", data=data, family=families.Poisson(),
                    cov_struct=cov_struct.Exchangeable())
    return model.fit(cov_type=cov_type)


def wald(res, terms):
    """Joint Wald test that the listed coefficients are all zero, using the fit's covariance."""
    b = res.params[terms].to_numpy()
    v = res.cov_params().loc[terms, terms].to_numpy()
    chi2 = float(b @ np.linalg.solve(v, b))
    return chi2, len(terms), stats.chi2.sf(chi2, len(terms))
