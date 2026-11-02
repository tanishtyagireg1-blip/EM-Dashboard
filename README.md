# PlateWise: dashboard data pack

Data, charts and model code behind the **PlateWise** Working With AI (WAI) report for *The Entrepreneurial Manager* (IIM Ranchi, MBA 2026-28).
PlateWise is a concept venture: AI-driven food-waste analytics and demand forecasting for institutional canteens and messes.

> **Read this first.** The pilot data (`pilot_*_simulated.csv`) is **synthetic**. It was generated from assumptions, not measured in a real mess. Every dashboard view that uses it must carry a visible "Simulated data" label. Market figures come from published estimates cited in the report; low-confidence assumptions are listed in the report's Appendix B.

## Folder layout

```
index.html the dashboard (open it, or publish on GitHub Pages)
data/      all numbers the dashboard reads (data.js + CSVs + one JSON)
charts/    10 static PNG charts from the report (for reference / fallback images)
model/     model.py, the script that regenerates everything in data/ and charts/
report/    the Word report (source of truth for wording and numbers)
```

## Data dictionary (`data/`)

| File | What it holds |
|---|---|
| `results.json` | Everything: assumptions (`A`), unit economics (`U`), market sizing, annual plan, 48 monthly rows, scenarios, tornado (`sens`), Monte Carlo (`mc`), pilot accuracy (`pilot`), cooking-policy comparison (`pol`), use of funds (`uof`), idea screening (`screen`) |
| `plan_48_month.csv` | Monthly base-case plan: customers, revenue, CAC, fixed cost, net cash, cumulative cash (INR) |
| `annual_plan.csv` | Year 1-4 summary of the plan (INR) |
| `scenarios.csv` | Pessimistic / base / optimistic outcomes |
| `sensitivity_tornado.csv` | One-at-a-time change in LTV:CAC per driver |
| `market_sizing.csv` | Top-down chain, TAM / SAM / SOM, diners (INR crore) |
| `unit_economics.csv` | Per 1,000-diner site-year economics (INR) |
| `idea_screening.csv` | Five-idea scorecard, scores 1-5, with weighted totals |
| `pilot_eval_simulated.csv` | 28-day evaluation window: demand, fixed cooking norm, and forecasts (naive, 4-week average, ridge, gradient boosting). **Simulated** |
| `pilot_full_simulated.csv` | Full 56-day simulated demand and cooking norm. **Simulated** |

Units: money in `results.json` and the CSVs is in **rupees** unless a column says crore or lakh (1 lakh = 100,000; 1 crore = 10,000,000).
Note: `uof` and `pol` in `results.json` were added after the model run; re-running `model.py` regenerates the rest but not these two blocks.

## Key headline numbers (for tiles)

TAM ₹138 crore | SAM ₹36.5 crore | SOM ₹2.58 crore ARR (month 36) | LTV:CAC 3.06 | payback 12.3 months | gross margin 70% | funding need ₹2.05 crore (ask ₹2.5 crore) | first cash-positive month 41 | 92 customer sites at month 36 | 2.24 million portions saved per year at month 36

## The dashboard (`index.html`)

A single-page static site with no backend and no external libraries (charts are drawn as inline SVG, so it also works offline). Open `index.html` in a browser, or publish it (see below). It reads `data/data.js`, which is generated from `results.json` and the pilot CSV.

What it shows: scenario explorer with 8 sliders (the 48-month model is re-run in the browser and matches the report's numbers), headline tiles, cash J-curve, customer sites, market funnel and TAM/SAM/SOM, unit economics, tornado, a clearly labelled **SIMULATED** pilot explorer with a buffer slider, and the scenario table. Light and dark themes, hover tooltips and a table view under each chart.

If you edit `data/results.json` or the pilot CSV, regenerate `data/data.js` (it is just `window.PW = {...}` built from those files).

### Original dashboard plan (for reference)

1. **Header tiles:** TAM, LTV:CAC, payback, funding ask, portions saved.
2. **Scenario switcher:** Pessimistic / Base / Optimistic toggles the J-curve, ARR, funding need.
3. **J-curve:** monthly net and cumulative cash, with the trough and first cash-positive month marked.
4. **Market funnel:** contract catering -> food -> waste -> reducible -> TAM -> SAM -> SOM.
5. **Unit economics:** client saving vs PlateWise revenue, cost and profit per site; sliders for price, CAC and churn that recompute LTV:CAC (formula in `model/model.py`, `unit()` and `ltvcac()`).
6. **Tornado:** sensitivity of LTV:CAC.
7. **Pilot explorer (labelled SIMULATED):** demand vs fixed norm vs forecast by day; MAPE comparison; waste vs shortage trade-off.
8. **Risk and assumptions table:** low-confidence assumptions highlighted.

Colour system used in the report: blue `#2A78D6`, teal `#118A60`, violet `#6250D6`, orange `#D9541E`, navy `#14284B`. Orange = adverse / risk, blue = base, teal = favourable.

## Regenerate the data

```bash
pip install -r requirements.txt
python model/model.py      # rewrites data/results.json, the CSVs and charts/*.png
```
Random seeds are fixed, so results are reproducible.

## Publish on GitHub Pages

1. Create a new GitHub repo and upload this folder's contents (or `git init`, `git add .`, `git commit`, `git push`).
2. Keep `index.html` at the repo root, next to the `data/` folder.
3. In the repo: Settings -> Pages -> Deploy from branch -> `main` / root. Your dashboard appears at `https://<your-username>.github.io/<repo-name>/`.

## Licence and attribution

Student coursework. AI-assisted (Claude, Anthropic); AI use is disclosed in the report, Section 11.

## Simulated survey (synthetic data)

`simulated_survey/` holds the synthetic student-survey rehearsal described in Section 2.6 and Appendix D of the report: `simulate_survey.py` (fixed seed), `SIMULATED_survey_responses.csv` and `sim_survey_results.json`. **All responses are synthetic, drawn from assumed distributions; none is real student data.** The model and dashboard do not read these files.
