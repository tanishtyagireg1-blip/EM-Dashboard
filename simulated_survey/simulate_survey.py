"""SIMULATED student survey for the PlateWise report. ALL RESPONSES ARE SYNTHETIC.

The 'respondents' are drawn from the assumed distributions below. The output cannot validate
anything: every result is a restatement of the assumptions. Its use is to rehearse the survey
instrument and analysis, to size the real survey, and to fix decision rules in advance.

Run:  python3 simulate_survey.py      (fixed seed)
Writes: SIMULATED_survey_responses.csv, sim_survey_results.json, ../charts/fig_sim_survey.png
"""
import json, math
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED, N = 20261102, 60
rng = np.random.default_rng(SEED)

# ---- ASSUMED population distributions (the generating assumptions; see Appendix D) ----
SEEN  = ["Never", "Rarely", "Sometimes", "Often", "Always"];            P_SEEN  = [.03, .12, .30, .35, .20]
SHARE = ["Less than 5%", "5-10%", "10-20%", "20-30%", "More than 30%"]; P_SHARE = [.15, .30, .30, .17, .08]
MID   = [2.5, 7.5, 15.0, 25.0, 35.0]
RUN   = ["Never", "Rarely", "Sometimes", "Often"];                      P_RUN   = [.20, .35, .30, .15]
SKIP  = ["Never", "Rarely", "Sometimes", "Often", "Always"];            P_SKIP  = [.08, .22, .40, .22, .08]
SIGN  = ["No", "Maybe", "Yes"];                                         P_SIGN  = [.20, .40, .40]   # ordered low->high
ATT_MEAN, ATT_SD = 10.8, 2.2        # of 14 meals (77% attendance, as in the pilot)
RHO = 0.4                           # link: students who see more waste are more willing to sign up

def ordinal(z, probs):
    cuts = norm.ppf(np.cumsum(probs)[:-1])
    return np.searchsorted(cuts, z)

z_seen = rng.standard_normal(N)
z_sign = RHO * z_seen + math.sqrt(1 - RHO ** 2) * rng.standard_normal(N)
seen = ordinal(z_seen, P_SEEN)
sign = ordinal(z_sign, P_SIGN)
share = ordinal(rng.standard_normal(N), P_SHARE)
run = ordinal(rng.standard_normal(N), P_RUN)
skip = ordinal(rng.standard_normal(N), P_SKIP)
att = np.clip(np.round(rng.normal(ATT_MEAN, ATT_SD, N)), 0, 14).astype(int)

df = pd.DataFrame({
    "respondent": [f"SIM-{i+1:03d}" for i in range(N)],
    "attended_of_14": att,
    "waste_seen": [SEEN[i] for i in seen],
    "waste_share": [SHARE[i] for i in share],
    "runout": [RUN[i] for i in run],
    "skip_unannounced": [SKIP[i] for i in skip],
    "signup": [SIGN[i] for i in sign],
})
df.to_csv("SIMULATED_survey_responses.csv", index=False)

def cnt(vals, levels): return [int(sum(v == l for v in vals)) for l in levels]
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(100 * (c - h), 1), round(100 * (c + h), 1)

c_seen, c_share = cnt(df.waste_seen, SEEN), cnt(df.waste_share, SHARE)
c_run, c_skip, c_sign = cnt(df.runout, RUN), cnt(df.skip_unannounced, SKIP), cnt(df.signup, ["Yes", "Maybe", "No"])
k_seen = sum(c_seen[3:]); k_run = sum(c_run[2:]); k_skip = sum(c_skip[2:]); k_sign = c_sign[0] + c_sign[1]
mean_share = float(np.mean([MID[i] for i in share]))
res = dict(
    n=N, seed=SEED, SIMULATED=True,
    waste_seen=dict(zip(SEEN, c_seen)), waste_share=dict(zip(SHARE, c_share)), runout=dict(zip(RUN, c_run)),
    skip=dict(zip(SKIP, c_skip)), signup=dict(zip(["Yes", "Maybe", "No"], c_sign)),
    pct_waste_often_always=round(100 * k_seen / N, 1), ci_waste=wilson(k_seen, N),
    mean_perceived_waste_share=round(mean_share, 1),
    pct_runout_sometimes_plus=round(100 * k_run / N, 1), ci_runout=wilson(k_run, N),
    pct_skip_sometimes_plus=round(100 * k_skip / N, 1), ci_skip=wilson(k_skip, N),
    pct_signup_yes_or_maybe=round(100 * k_sign / N, 1), ci_signup=wilson(k_sign, N),
    pct_signup_yes=round(100 * c_sign[0] / N, 1),
    mean_attendance_pct=round(float(att.mean()) / 14 * 100, 1),
    half_width_n30=round(100 * 1.96 * math.sqrt(.25 / 30), 1), half_width_n60=round(100 * 1.96 * math.sqrt(.25 / 60), 1),
)
Path("sim_survey_results.json").write_text(json.dumps(res, indent=1))

BLUE, ORANGE, TEAL, INK = "#2a78d6", "#eb6834", "#118a60", "#0b0b0b"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c3c2b7", "axes.titlesize": 9.5, "axes.titleweight": "bold"})
fig, axs = plt.subplots(1, 3, figsize=(11, 3.7), facecolor="white")
panels = [("How often do you see food thrown away?", ["Never", "Rarely", "Sometimes", "Often", "Always"], c_seen, ORANGE),
          ("Perceived share of food thrown away", ["<5%", "5-10%", "10-20%", "20-30%", ">30%"], c_share, BLUE),
          ("Would you mark attendance?", ["Yes", "Maybe", "No"], c_sign, TEAL)]
for ax, (title, labels, vals, col) in zip(axs, panels):
    ys = list(range(len(labels)))
    ax.barh(ys, vals, color=col, height=0.6)
    ax.set_yticks(ys); ax.set_yticklabels(labels); ax.invert_yaxis()
    ax.set_title(title, loc="left")
    ax.tick_params(axis="x", colors="#898781"); ax.set_xlabel("Synthetic respondents", color="#52514e")
    mx = max(vals)
    for y, v in zip(ys, vals):
        ax.text(v + mx * 0.02, y, f"{v} ({100*v/N:.0f}%)", va="center", fontsize=8.5, color=INK)
    ax.set_xlim(0, mx * 1.45)
fig.suptitle(f"SIMULATED DATA: {N} synthetic respondents drawn from assumed distributions (not real students)",
             x=0.01, ha="left", fontsize=10, fontweight="bold", color=ORANGE)
fig.tight_layout(rect=(0, 0, 1, 0.93))
Path("../charts").mkdir(exist_ok=True)
fig.savefig("../charts/fig_sim_survey.png", dpi=200, bbox_inches="tight", facecolor="white")
print(json.dumps(res, indent=1))
