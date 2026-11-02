"""
PlateWise (working name) - Entrepreneurship WAI model.
Builds: market sizing, unit economics, 36-month plan, sensitivity, Monte Carlo,
and a SIMULATED pilot (synthetic data, labelled as such in the report).
Outputs: results.json + PNG charts in ./charts
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data"
CH = ROOT / "charts"
OUT.mkdir(parents=True, exist_ok=True)
CH.mkdir(parents=True, exist_ok=True)

# ---------------- palette (validated: dataviz skill, light mode) ----------------
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "savefig.dpi": 200, "axes.titlesize": 9,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False,
})


def save(fig, name):
    fig.savefig(CH / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------- ASSUMPTIONS (all labelled in the report) ----------------
A = dict(
    fx=88.0,                      # INR per USD (assumption)
    diners=1000, days=270,        # reference canteen: 1,000 diners, 270 mess days
    food_per_day=90.0,            # Rs raw-food spend / diner / day
    waste_pct=0.10, reduction=0.30,
    sub_per_diner_month=15.0, sub_months=9, gain_share=0.20,
    cogs_cloud=30000, cogs_support=45000, cogs_hw_dep=10000, hw_capex=30000,
    cac=200000, churn=0.20, disc=0.12,
    fixed_opex=[2400000, 4500000, 7000000, 9000000],   # Rs per year
    adds=[12, 33, 65, 55], add_mult=1.0, fixed_mult=1.0,
    contract_catering_usd_m=3884.99, food_cost_share=0.35,
    enrol=4.33e7, hostel_share=0.10, sam_reach=0.30,
    cost_per_portion=30.0,
)


def unit(a):
    gross = a["diners"] * a["days"] * a["food_per_day"] * a["waste_pct"] * a["reduction"]
    sub = a["diners"] * a["sub_per_diner_month"] * a["sub_months"]
    gain = a["gain_share"] * gross
    rev = sub + gain
    cogs = a["cogs_cloud"] + a["cogs_support"] + a["cogs_hw_dep"]
    gp = rev - cogs
    payback = a["cac"] / (gp / 12) if gp > 0 else float("inf")
    ltv = gp / (a["churn"] + a["disc"]) if gp > 0 else 0
    return dict(gross=gross, sub=sub, gain=gain, rev=rev, cogs=cogs, gp=gp,
                gm=gp / rev if rev else 0, payback=payback, ltv=ltv,
                ltv_cac=ltv / a["cac"], net_client=gross - rev,
                client_roi=gross / rev if rev else 0,
                portions=gross / a["cost_per_portion"])


def monthly_adds(a, months):
    out = np.zeros(months)
    rates = [x / 12 for x in a["adds"]]
    raw = np.interp(np.arange(months) + 0.5, [6, 18, 30, 42], rates)
    for y in range(months // 12):
        sl = slice(12 * y, 12 * y + 12)
        if y == 0:
            w_ = np.arange(1, 13) / 78.0           # launch ramp in year 1
            out[sl] = a["adds"][0] * w_
        else:
            out[sl] = raw[sl] * a["adds"][y] / raw[sl].sum()
    return out * a["add_mult"]


def simulate(a, months=48):
    u = unit(a)
    ADDS = monthly_adds(a, months)
    cust = 0.0
    cum = 0.0
    rows = []
    for m in range(months):
        y, k = m // 12, m % 12
        adds = ADDS[m]
        start = cust
        cust = cust - cust * a["churn"] / 12 + adds
        avg = (start + cust) / 2
        rev = avg * u["rev"] / 12
        cogs_cash = avg * (a["cogs_cloud"] + a["cogs_support"]) / 12
        cogs_acc = avg * u["cogs"] / 12
        hw = adds * a["hw_capex"]
        cac = adds * a["cac"]
        fixed = a["fixed_opex"][y] * a["fixed_mult"] / 12
        net = rev - cogs_cash - hw - cac - fixed
        cum += net
        rows.append(dict(m=m + 1, year=y + 1, adds=adds, cust=cust, avg=avg, rev=rev,
                         cogs_acc=cogs_acc, cogs_cash=cogs_cash, hw=hw, cac=cac,
                         fixed=fixed, net=net, cum=cum,
                         op=rev - cogs_acc - cac - fixed))
    return pd.DataFrame(rows), u


# ---------------- 1. UNIT ECONOMICS ----------------
U = unit(A)
df, _ = simulate(A)
annual = df.groupby("year").agg(rev=("rev", "sum"), cogs_acc=("cogs_acc", "sum"),
                                cac=("cac", "sum"), fixed=("fixed", "sum"),
                                hw=("hw", "sum"), cogs_cash=("cogs_cash", "sum"),
                                net=("net", "sum"), op=("op", "sum"),
                                adds=("adds", "sum"))
annual["gp"] = annual["rev"] - annual["cogs_acc"]
annual["cust_end"] = [df[df.year == y].cust.iloc[-1] for y in (1, 2, 3, 4)]
annual["cust_avg"] = [df[df.year == y].avg.mean() for y in (1, 2, 3, 4)]
trough = df.cum.min()
trough_m = int(df.loc[df.cum.idxmin(), "m"])
funding_need = -trough
arr_m36 = df.cust.iloc[35] * U["rev"]
cust_m36 = df.cust.iloc[35]
breakeven_customers = A["fixed_opex"][3] / U["gp"]
runrate_pos_month = next((int(r.m) for r in df.itertuples() if (r.rev - r.cogs_acc - r.fixed) > 0), None)
op_m36 = df.rev.iloc[-1] - df.cogs_acc.iloc[-1] - df.fixed.iloc[-1]  # monthly, before growth CAC
first_pos_net = next((int(r.m) for r in df.itertuples() if r.net > 0 and r.m > 12), None)
cum_m48 = df.cum.iloc[-1]
net_y = annual["net"].tolist()

# ---------------- 2. MARKET SIZING ----------------
cc_cr = A["contract_catering_usd_m"] * 1e6 * A["fx"] / 1e7        # ₹ crore
food_cr = cc_cr * A["food_cost_share"]
waste_cr = food_cr * A["waste_pct"]
reduc_cr = waste_cr * A["reduction"]
take = U["rev"] / U["gross"]
tam_cr = reduc_cr * take
hostel_diners = A["enrol"] * A["hostel_share"]
arpd = U["rev"] / A["diners"]                       # revenue per diner per year
bottomup_cr = hostel_diners * arpd / 1e7
sam_diners = hostel_diners * A["sam_reach"]
sam_cr = sam_diners * arpd / 1e7
som_cr = arr_m36 / 1e7
market = dict(cc_cr=cc_cr, food_cr=food_cr, waste_cr=waste_cr, reduc_cr=reduc_cr,
              take=take, tam_cr=tam_cr, hostel_diners=hostel_diners, arpd=arpd,
              bottomup_cr=bottomup_cr, sam_diners=sam_diners, sam_cr=sam_cr,
              som_cr=som_cr, som_share_sam=som_cr / sam_cr)

# ---------------- 3. SENSITIVITY (tornado on 36-month cumulative cash) ----------------
def ltvcac(a):
    return unit(a)["ltv_cac"]

base_lc = ltvcac(A)
sens_def = [
    ("Share of food purchases wasted today", "waste_pct", 0.05, 0.15, "5% / 15%"),
    ("Waste reduction achieved", "reduction", 0.15, 0.45, "15% / 45%"),
    ("Diners per canteen", "diners", 700, 1400, "700 / 1,400"),
    ("Raw-food spend per diner per day", "food_per_day", 70, 120, "₹70 / 120"),
    ("Customer acquisition cost", "cac", 300000, 150000, "₹3.0 / 1.5 lakh"),
    ("Gain-share rate", "gain_share", 0.15, 0.25, "15% / 25%"),
    ("Subscription (₹/diner/month)", "sub_per_diner_month", 10, 20, "₹10 / 20"),
    ("Annual churn", "churn", 0.30, 0.10, "30% / 10%"),
]
sens = []
for label, key, lo, hi, txt in sens_def:
    a1, a2 = dict(A), dict(A)
    a1[key], a2[key] = lo, hi
    c1, c2 = ltvcac(a1), ltvcac(a2)
    sens.append(dict(label=label, txt=txt, low=c1 - base_lc, high=c2 - base_lc,
                     swing=abs(c2 - c1)))
sens.sort(key=lambda s: -s["swing"])

# plan-level scenarios
def scen(name, **kw):
    a = dict(A); a.update(kw)
    d_, u_ = simulate(a)
    return dict(name=name, ltv_cac=u_["ltv_cac"], payback=u_["payback"], gm=u_["gm"],
                cust36=float(d_.cust.iloc[35]), arr36=float(d_.cust.iloc[35] * u_["rev"]),
                need=float(-d_.cum.min()), trough_m=int(d_.loc[d_.cum.idxmin(), "m"]),
                cum48=float(d_.cum.iloc[-1]),
                pos_month=next((int(r.m) for r in d_.itertuples() if r.net > 0 and r.m > 12), None))

scenarios = [
    scen("Pessimistic", reduction=0.20, waste_pct=0.08, add_mult=0.7, cac=260000, churn=0.28),
    scen("Base"),
    scen("Optimistic", reduction=0.40, waste_pct=0.12, add_mult=1.2, cac=160000, churn=0.12),
]

# ---------------- 4. MONTE CARLO (per-canteen economics) ----------------
rng = np.random.default_rng(2026)
n = 10000
w = rng.triangular(0.03, 0.10, 0.15, n)
r = rng.triangular(0.10, 0.30, 0.45, n)
f = rng.triangular(70, 90, 120, n)
d = rng.triangular(600, 1000, 1500, n)
gross = d * A["days"] * f * w * r
fees = d * A["sub_per_diner_month"] * A["sub_months"] + A["gain_share"] * gross
ratio = gross / fees
mc = dict(
    gross_p10=np.percentile(gross, 10), gross_p50=np.percentile(gross, 50),
    gross_p90=np.percentile(gross, 90), ratio_p10=np.percentile(ratio, 10),
    ratio_p50=np.percentile(ratio, 50), ratio_p90=np.percentile(ratio, 90),
    p_ratio_gt1=float((ratio > 1).mean()), p_ratio_gt2=float((ratio > 2).mean()),
    p_gt_1lakh=float((gross > 100000).mean()),
)

# ---------------- 5. SIMULATED PILOT (synthetic data) ----------------
def make_pilot(seed=7, N=800, days=56):
    rg = np.random.default_rng(seed)
    lunch = [0.80, 0.79, 0.78, 0.78, 0.74, 0.68, 0.60]
    dinner = [0.86, 0.87, 0.86, 0.85, 0.80, 0.74, 0.80]
    rows = []
    wk = 0.0
    for dd in range(days):
        dow = dd % 7
        week = dd // 7
        if dow == 0:
            wk = 0.5 * wk + rg.normal(0, 0.015)
        exam = 1 if week == 4 else 0
        fest = 1 if dd in (17, 18) else 0
        holiday = 1 if dd == 40 else 0
        special = 1 if (dow == 6 and week in (1, 3, 5, 7)) else 0
        for meal, base in ((0, lunch), (1, dinner)):
            fr = base[dow] + wk
            fr += exam * (0.05 if meal == 0 else 0.04)
            fr -= fest * (0.18 if dd == 17 else 0.15)
            fr -= holiday * 0.20
            fr += special * (0.06 if meal == 1 else 0.0)
            fr += rg.normal(0, 0.025)
            fr = float(np.clip(fr, 0.3, 0.98))
            demand = int(round(N * fr))
            if dow < 5:
                cook = round((0.90 if meal == 0 else 0.92) * N)
            else:
                cook = round((0.80 if meal == 0 else 0.88) * N)
            rows.append(dict(day=dd, dow=dow, meal=meal, exam=exam, ev=int(fest or holiday),
                             special=special, demand=demand, base_cook=cook))
    return pd.DataFrame(rows), N


P, NRES = make_pilot()
key = P.set_index(["day", "meal"])["demand"]


def lagv(day, meal, k):
    return key.get((day - k, meal), np.nan)


def feats(day, meal, row):
    x = [1.0 if row.dow == i else 0.0 for i in range(7)]
    x += [meal, row.exam, row.ev, row.special]
    x += [lagv(day, meal, 7) / NRES, lagv(day, meal, 1) / NRES]
    x += [np.mean([lagv(day, meal, k) for k in (7, 14, 21, 28) if not np.isnan(lagv(day, meal, k))]) / NRES]
    return x


EV0 = 28
recs = []
for day in range(EV0, 56):
    train = P[(P.day >= 7) & (P.day < day)]
    Xtr = np.array([feats(r_.day, r_.meal, r_) for r_ in train.itertuples()])
    ytr = train.demand.values / NRES
    ridge = Ridge(alpha=0.5).fit(Xtr, ytr)
    gb = GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05,
                                   random_state=0).fit(Xtr, ytr)
    for r_ in P[P.day == day].itertuples():
        x = np.array([feats(day, r_.meal, r_)])
        l7 = lagv(day, r_.meal, 7)
        ma4 = np.mean([lagv(day, r_.meal, k) for k in (7, 14, 21, 28)])
        recs.append(dict(day=day, meal=r_.meal, demand=r_.demand, base_cook=r_.base_cook,
                         naive=l7, ma4=ma4, ridge=ridge.predict(x)[0] * NRES,
                         gb=gb.predict(x)[0] * NRES))
E = pd.DataFrame(recs)


def mape(col):
    return float(np.mean(np.abs(E[col] - E.demand) / E.demand) * 100)


def mae(col):
    return float(np.mean(np.abs(E[col] - E.demand)))


acc = {k: dict(mape=mape(k), mae=mae(k)) for k in ("naive", "ma4", "gb", "ridge")}
best = min(("gb", "ridge"), key=lambda k: acc[k]["mape"])


def policy(cook, dem):
    waste = np.maximum(cook - dem, 0).sum() / cook.sum()
    unserved = np.maximum(dem - cook, 0).sum() / dem.sum()
    stock = float((cook < dem).mean())
    return waste, unserved, stock


base_w, base_u, base_s = policy(E.base_cook.values.astype(float), E.demand.values.astype(float))
buffers = np.arange(0, 0.201, 0.01)
curves = {}
for k in ("naive", best):
    pts = []
    for b in buffers:
        cook = np.round(E[k].values * (1 + b))
        wst, uns, stk = policy(cook, E.demand.values.astype(float))
        pts.append((b, wst, uns, stk))
    curves[k] = pts
target_unserved = max(base_u, 0.005)
op = next((p for p in curves[best] if p[2] <= target_unserved and p[3] <= 0.10), curves[best][-1])
op_naive = next((p for p in curves["naive"] if p[2] <= target_unserved and p[3] <= 0.10), curves["naive"][-1])
waste_red = 1 - op[1] / base_w
waste_red_naive = 1 - op_naive[1] / base_w
days_eval = E.day.nunique()
cook_base = E.base_cook.sum()
waste_p_base = base_w * cook_base
cook_op = np.round(E[best].values * (1 + op[0]))
waste_p_op = np.maximum(cook_op - E.demand.values, 0).sum()
portions_saved_per_day = (waste_p_base - waste_p_op) / days_eval
rs_saved_year_pilot = portions_saved_per_day * A["cost_per_portion"] * A["days"]
rs_per_diner_year_pilot = rs_saved_year_pilot / NRES
pilot = dict(
    N=NRES, days_total=56, eval_days=int(days_eval), train_days=EV0,
    acc=acc, best=best, base_waste=base_w, base_unserved=base_u, base_stockout=base_s,
    op_buffer=op[0], op_waste=op[1], op_unserved=op[2], op_stockout=op[3],
    target_unserved=target_unserved, waste_red=waste_red,
    naive_op_waste=op_naive[1], naive_op_buffer=op_naive[0], waste_red_naive=waste_red_naive,
    portions_saved_per_day=portions_saved_per_day, rs_saved_year=rs_saved_year_pilot,
    rs_per_diner_year=rs_per_diner_year_pilot,
    avg_attendance=float(P.demand.mean() / NRES), base_cook_avg=float(P.base_cook.mean() / NRES),
)

# ---------------- 6. IDEA SCREENING ----------------
crit = ["Market scale", "Data and measurability", "Low capital need", "Social / SDG impact",
        "Lean testability", "Policy tailwind"]
weights = np.array([0.20, 0.20, 0.15, 0.15, 0.20, 0.10])
ideas = {
    "EV battery-swapping for delivery riders": [5, 4, 1, 3, 2, 5],
    "Millet-based ready-to-eat foods": [3, 3, 3, 4, 3, 4],
    "Cold-chain micro-hubs for produce": [4, 3, 1, 5, 2, 5],
    "Food-waste analytics for canteens": [3, 5, 4, 4, 5, 4],
    "Vernacular placement-readiness platform": [4, 3, 4, 3, 4, 2],
}
w_alt = np.array([0.40, 0.15, 0.10, 0.10, 0.15, 0.10])
screen = []
for k, v in ideas.items():
    screen.append(dict(idea=k, scores=v, total=float(np.dot(weights, v)),
                       total_alt=float(np.dot(w_alt, v))))

# ================= CHARTS =================
def lakh(x, _=None):
    return f"{x/1e5:,.0f}"

# --- Fig: TAM / SAM / SOM
fig, ax = plt.subplots(figsize=(6.5, 2.3))
labs = ["TAM  revenue pool, contract and\ninstitutional catering (top-down)",
        "SAM  higher-ed hostel messes\nreachable in 3 years (bottom-up)",
        "SOM  year-3 recurring revenue\nat month 36 (model)"]
vals = [tam_cr, sam_cr, som_cr]
cols = [BLUE, BLUE, AQUA]
ax.barh(labs[::-1], vals[::-1], color=cols[::-1], height=0.52)
for i, v in enumerate(vals[::-1]):
    ax.text(v + max(vals) * 0.015, i, f"₹{v:,.1f} crore", va="center", fontsize=8.5, color=INK)
ax.set_xlim(0, max(vals) * 1.25)
ax.set_xlabel("Annual revenue pool, ₹ crore")
ax.grid(axis="y", visible=False)
save(fig, "fig_tam_sam_som.png")

# --- Fig: benchmarks of waste reduction
cases = [("Compass Group Canada site (Winnow)", 18), ("Marriott UK, Ireland, Nordics (53 hotels)", 25),
         ("Salem Health (Winnow)", 33), ("PwC Amsterdam canteen (Winnow)", 40),
         ("Carle Health, year 1 (Leanpath)", 40), ("Leanpath, average client claim", 50)]
fig, ax = plt.subplots(figsize=(6.5, 3.1))
ys = list(range(len(cases) + 2))
names = [c[0] for c in cases] + ["Winnow, typical range (6-12 months)", "Hilton Dubai / Marriott Dubai (Winnow)"]
vals = [c[1] for c in cases]
for i, v in enumerate(vals):
    ax.plot([0, v], [i, i], color=GRID, lw=1.2, zorder=1)
    ax.plot(v, i, "o", color=BLUE, ms=6, zorder=3)
    ax.text(v + 1.5, i, f"{v}%", va="center", fontsize=8, color=INK)
i = len(cases)
ax.plot([40, 70], [i, i], color=BLUE, lw=5, solid_capstyle="round", zorder=3)
ax.text(72, i, "40-70%", va="center", fontsize=8, color=INK)
ax.plot([0, 70], [i + 1, i + 1], color=GRID, lw=1.2, zorder=1)
ax.plot(70, i + 1, "o", color=BLUE, ms=6, zorder=3)
ax.text(71.5, i + 1, "about 70%", va="center", fontsize=8, color=INK)
ax.axvline(30, color=ORANGE, lw=1.6, zorder=2)
ax.text(29.4, -0.9, "Planning assumption: 30%", color=ORANGE, fontsize=8, fontweight="bold", va="center", ha="right")
ax.axvline(waste_red * 100, color=AQUA, lw=1.6, ls=(0, (4, 2)), zorder=2)
ax.text(waste_red * 100 + 0.6, len(cases) + 1.85, f"Simulated pilot: {waste_red*100:.0f}%", color="#0f7f58",
        fontsize=8, fontweight="bold", va="center")
ax.set_yticks(ys)
ax.set_yticklabels(names, fontsize=8)
ax.set_ylim(-1.4, len(cases) + 2.3)
ax.set_xlim(0, 85)
ax.set_xlabel("Reported reduction in food waste, %")
ax.invert_yaxis()
ax.grid(axis="y", visible=False)
save(fig, "fig_benchmarks.png")

# --- Fig: simulated attendance vs forecast vs baseline plan
daily = E.groupby("day").agg(demand=("demand", "sum"), base=("base_cook", "sum"),
                             fc=(best, "sum"), nv=("naive", "sum")).reset_index()
fig, ax = plt.subplots(figsize=(6.5, 2.9))
x = daily.day - EV0 + 1
ax.step(x, daily.base, where="mid", color=ORANGE, lw=1.6)
ax.plot(x, daily.demand, color=BLUE, lw=1.8, marker="o", ms=3)
ax.plot(x, daily.fc, color=AQUA, lw=1.6, ls=(0, (4, 2)))
ax.text(x.iloc[-1] + 0.4, daily.base.iloc[-1], "Current practice:\nfixed cooking norm", color=ORANGE, fontsize=7.5, va="center", fontweight="bold")
ax.text(x.iloc[-1] + 0.4, 1255, "Actual portions\nneeded", color=BLUE, fontsize=7.5, va="center", fontweight="bold")
ax.text(x.iloc[-1] + 0.4, 1085, "Model forecast\n(made day before)", color="#0f7f58", fontsize=7.5, va="center", fontweight="bold")
ax.set_xlim(1, 28)
ax.set_xlabel("Evaluation day (weeks 5-8 of the simulated pilot)")
ax.set_ylabel("Lunch + dinner portions, 800 residents")
ax.set_xlim(0.5, 33)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([0], [0], color=BLUE, lw=1.8, marker="o", ms=3, label="Actual portions needed"),
                   Line2D([0], [0], color=AQUA, lw=1.6, ls=(0, (4, 2)), label="Model forecast"),
                   Line2D([0], [0], color=ORANGE, lw=1.6, label="Current practice: fixed cooking norm")],
          loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=7.5)
save(fig, "fig_pilot_series.png")

# --- Fig: forecast accuracy
fig, ax = plt.subplots(figsize=(6.5, 2.1))
order = [("naive", "Same meal last week"), ("ma4", "4-week average, same weekday"),
         ("gb", "Gradient boosting + calendar"), ("ridge", "Ridge regression + calendar")]
vv = [acc[k]["mape"] for k, _ in order]
cc = [BLUE, BLUE, BLUE, BLUE]
cc[[k for k, _ in order].index(best)] = AQUA
ax.barh([l for _, l in order][::-1], vv[::-1], color=cc[::-1], height=0.5)
for i, v in enumerate(vv[::-1]):
    ax.text(v + 0.15, i, f"{v:.1f}%", va="center", fontsize=8.5)
ax.set_xlabel("Mean absolute percentage error, rolling-origin test (28 days, 56 meals)")
ax.set_xlim(0, max(vv) * 1.2)
ax.grid(axis="y", visible=False)
save(fig, "fig_mape.png")

# --- Fig: trade-off curve
fig, ax = plt.subplots(figsize=(6.5, 3.0))
for k, col, lab in (("naive", BLUE, "Last-week rule + buffer"), (best, AQUA, "Forecast model + buffer")):
    xs = [p[2] * 100 for p in curves[k]]
    ys_ = [p[1] * 100 for p in curves[k]]
    ax.plot(xs, ys_, color=col, lw=1.8)
ax.plot(base_u * 100, base_w * 100, "D", color=ORANGE, ms=7, zorder=4)
ax.annotate("Current practice\n(fixed cooking norm)", (base_u * 100, base_w * 100), xytext=(base_u * 100 + 1.2, base_w * 100 + 1.4),
            color=ORANGE, fontsize=8, fontweight="bold", arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.8))
ax.plot(op[2] * 100, op[1] * 100, "o", mfc="white", mec=AQUA, mew=2, ms=9, zorder=5)
ax.annotate(f"Chosen setting: +{op[0]*100:.0f}% buffer\nwaste {op[1]*100:.1f}%, unserved {op[2]*100:.1f}%",
            (op[2] * 100, op[1] * 100), xytext=(1.15, 9.0), color="#0f7f58", fontsize=8,
            fontweight="bold", arrowprops=dict(arrowstyle="-", color=AQUA, lw=0.8))
xs_n = [p[2] * 100 for p in curves["naive"]]
ys_n = [p[1] * 100 for p in curves["naive"]]
ax.text(xs_n[2] + 0.4, ys_n[2] + 0.9, "Last-week rule + buffer", color=BLUE, fontsize=8, fontweight="bold")
xs_b = [p[2] * 100 for p in curves[best]]
ys_b = [p[1] * 100 for p in curves[best]]
ax.text(0.2, 0.8, "Forecast model + buffer", color="#0f7f58", fontsize=8, fontweight="bold")
ax.set_xlabel("Portions unserved (shortage), % of demand")
ax.set_ylabel("Food wasted, % of portions cooked")
ax.set_xlim(left=-0.3)
ax.set_ylim(bottom=0)
ax.legend(handles=[Line2D([0], [0], color=BLUE, lw=1.8, label="Last-week rule + buffer"),
                   Line2D([0], [0], color=AQUA, lw=1.8, label="Forecast model + buffer"),
                   Line2D([0], [0], color=ORANGE, marker="D", ls="", ms=6, label="Current practice")],
          loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=7.5)
save(fig, "fig_tradeoff.png")

# --- Fig: unit economics (two panels, same unit)
fig, axs = plt.subplots(1, 2, figsize=(6.5, 3.0), sharey=True, gridspec_kw=dict(width_ratios=[1.35, 1]))
L = 1e5
g, s, gn = U["gross"] / L, U["sub"] / L, U["gain"] / L
steps = [("Gross food\nsaved", 0, g, BLUE), ("Subscription\nfee", g - s, s, ORANGE),
         ("Gain-share\nfee", g - s - gn, gn, ORANGE), ("Net saving\nto canteen", 0, g - s - gn, AQUA)]
ax = axs[0]
for i, (lab, bot, hgt, col) in enumerate(steps):
    ax.bar(i, hgt, bottom=bot, color=col, width=0.58)
    ax.text(i, bot + hgt + 0.15, f"{hgt:.2f}" if i in (0, 3) else f"-{hgt:.2f}", ha="center", fontsize=8.5)
ax.set_xticks(range(4))
ax.set_xticklabels([x[0] for x in steps], fontsize=7.5)
ax.set_ylabel("₹ lakh per year, 1,000-diner canteen")
ax.set_title("Client view", fontsize=9)
ax.grid(axis="x", visible=False)
ax = axs[1]
rv, cg, gpft = U["rev"] / L, U["cogs"] / L, U["gp"] / L
steps2 = [("Revenue", 0, rv, BLUE), ("Cloud, support,\nhardware", rv - cg, cg, ORANGE), ("Gross\nprofit", 0, gpft, AQUA)]
for i, (lab, bot, hgt, col) in enumerate(steps2):
    ax.bar(i, hgt, bottom=bot, color=col, width=0.58)
    ax.text(i, bot + hgt + 0.15, f"{hgt:.2f}" if i != 1 else f"-{hgt:.2f}", ha="center", fontsize=8.5)
ax.set_xticks(range(3))
ax.set_xticklabels([x[0] for x in steps2], fontsize=7.5)
ax.set_title("Venture view", fontsize=9)
ax.grid(axis="x", visible=False)
fig.tight_layout()
save(fig, "fig_unit_economics.png")

# --- Fig: price positioning
leanpath_inr = 1200 * 12 * A["fx"] / L
fig, ax = plt.subplots(figsize=(6.5, 1.7))
labs = ["Global incumbent list price\n(Leanpath, USD 1,200/month)", "This venture, per 1,000-diner\ncanteen (subscription + gain share)"]
vals = [leanpath_inr, U["rev"] / L]
ax.barh(labs[::-1], vals[::-1], color=[AQUA, BLUE], height=0.5)
for i, v in enumerate(vals[::-1]):
    ax.text(v + 0.2, i, f"₹{v:.1f} lakh / year", va="center", fontsize=8.5)
ax.set_xlim(0, max(vals) * 1.3)
ax.set_xlabel("Annual price per canteen, ₹ lakh (USD 1 = ₹88 assumed)")
ax.grid(axis="y", visible=False)
save(fig, "fig_price_gap.png")

# --- Fig: Monte Carlo
fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.7))
ax = axs[0]
ax.hist(gross / L, bins=40, color=BLUE, edgecolor="white", linewidth=0.4)
ymax = ax.get_ylim()[1]
ax.set_ylim(0, ymax * 1.2)
for q, lab in ((mc["gross_p10"], "P10"), (mc["gross_p50"], "P50"), (mc["gross_p90"], "P90")):
    ax.axvline(q / L, color=ORANGE, lw=1.3)
    ax.text(q / L, ymax * 1.17, lab, color=ORANGE, fontsize=7.5, va="top", ha="center", fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.2))
ax.set_xlabel("Gross food saved, ₹ lakh / canteen / year")
ax.set_ylabel("Simulation runs")
ax.set_title("Savings per canteen", fontsize=9)
ax = axs[1]
ax.hist(np.clip(ratio, 0, 8), bins=40, color=BLUE, edgecolor="white", linewidth=0.4)
ax.axvline(1, color=ORANGE, lw=1.4)
ax.text(1.1, ax.get_ylim()[1] * 0.97, "Break-even for\nthe canteen (1x)", color=ORANGE, fontsize=7.5, va="top", fontweight="bold")
ax.set_xlabel("Food saved per Re 1 of fees (x)")
ax.set_title("Benefit-cost ratio", fontsize=9)
fig.tight_layout()
save(fig, "fig_montecarlo.png")

# --- Fig: 48 month plan
fig, axs = plt.subplots(2, 1, figsize=(6.5, 4.2), sharex=True)
ax = axs[0]
ax.plot(df.m, df.cust, color=BLUE, lw=2)
ax.set_ylabel("Active customers\n(canteens)")
for y in (12, 24, 36, 48):
    v = df.cust.iloc[y - 1]
    ax.plot(y, v, "o", color=BLUE, ms=5)
    ax.text(y - 0.5, v + 5, f"{v:.0f}", ha="right", fontsize=8.5, fontweight="bold")
ax = axs[1]
ax.plot(df.m, df.cum / L, color=BLUE, lw=2)
ax.axhline(0, color=AXIS, lw=1)
ax.plot(trough_m, trough / L, "o", color=ORANGE, ms=6)
ax.annotate(f"Deepest point: -₹{abs(trough)/L:,.0f} lakh (month {trough_m})", (trough_m, trough / L),
            xytext=(trough_m - 24, trough / L - 38), color=ORANGE, fontsize=8, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.8))
ax.set_ylim(trough / L * 1.35, 25)
ax.set_ylabel("Cumulative net cash,\n₹ lakh")
ax.set_xlabel("Month")
ax.set_xticks([1, 6, 12, 18, 24, 30, 36, 42, 48])
fig.tight_layout()
save(fig, "fig_plan48.png")

# --- Fig: tornado (LTV:CAC)
fig, ax = plt.subplots(figsize=(6.5, 3.3))
sx = sens[::-1]
for i, s_ in enumerate(sx):
    lo, hi = s_["low"], s_["high"]
    ax.barh(i, lo, color=ORANGE, height=0.55)
    ax.barh(i, hi, color=BLUE, height=0.55)
    ax.text(min(lo, hi) - 0.08, i, f"{s_['txt']}", ha="right", va="center", fontsize=7.2, color=INK2)
ax.axvline(0, color=INK2, lw=1)
ax.set_yticks(range(len(sx)))
ax.set_yticklabels([s_["label"] for s_ in sx], fontsize=8)
ax.set_xlabel(f"Change in LTV:CAC versus the base case ({base_lc:.1f}x).  Orange = adverse end, blue = favourable end")
lo_all = min(min(s_["low"], s_["high"]) for s_ in sens)
hi_all = max(max(s_["low"], s_["high"]) for s_ in sens)
ax.set_xlim(lo_all * 1.8, hi_all * 1.15)
ax.grid(axis="y", visible=False)
save(fig, "fig_tornado.png")

# ---------------- OUTPUT ----------------
res = dict(
    A=A, U=U, market=market, annual=annual.reset_index().to_dict("records"),
    monthly=df[["m", "cust", "rev", "net", "cum"]].to_dict("records"),
    trough=trough, trough_m=trough_m, funding_need=funding_need, arr_m36=arr_m36, cust_m36=cust_m36,
    breakeven_customers=breakeven_customers, base_lc=base_lc, first_pos_net=first_pos_net,
    runrate_pos_month=runrate_pos_month, cum_m48=cum_m48, net_y=net_y, scenarios=scenarios,
    sens=sens, mc=mc, pilot=pilot, screen=screen, crit=crit, weights=list(weights),
    w_alt=list(w_alt), leanpath_inr=leanpath_inr * L,
    pilot_table=E.assign(mape_naive=np.abs(E.naive - E.demand) / E.demand).groupby("meal")
    .agg(mape_naive=("mape_naive", "mean")).reset_index().to_dict("records"),
)
with open(OUT / "results.json", "w") as fh:
    json.dump(res, fh, indent=1, default=float)
E.to_csv(OUT / "pilot_eval_simulated.csv", index=False)
P.to_csv(OUT / "pilot_full_simulated.csv", index=False)
df.to_csv(OUT / "plan_48_month.csv", index=False)

print("UNIT", {k: round(v, 3) for k, v in U.items()})
print("MARKET", {k: round(v, 2) for k, v in market.items()})
print("ANNUAL"); print(annual.round(0))
print("trough", round(trough / L, 1), "month", trough_m, "need", round(funding_need / L, 1), "ARR36", round(arr_m36 / L, 1), "cust36", round(cust_m36,1), "BE cust", round(breakeven_customers, 1), "cum48", round(cum_m48 / L, 1), "first pos net", first_pos_net, "runrate pos", runrate_pos_month)
print("SCEN"); [print(" ", {k: (round(v,2) if isinstance(v,float) else v) for k,v in sc.items()}) for sc in scenarios]
print("CURVE best", [(round(p[0],2), round(p[1]*100,1), round(p[2]*100,2), round(p[3]*100,1)) for p in curves[best]][:12])
print("MC", {k: round(v, 3) for k, v in mc.items()})
print("PILOT", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in pilot.items() if k != "acc"})
print("ACC", {k: {a_: round(b_, 2) for a_, b_ in v.items()} for k, v in acc.items()})
print("SENS base", round(base_lc,2)); [print(" ", s_["label"], round(s_["low"], 2), round(s_["high"], 2)) for s_ in sens]
print("SCREEN", [(s_["idea"], round(s_["total"], 2), round(s_["total_alt"], 2)) for s_ in screen])
