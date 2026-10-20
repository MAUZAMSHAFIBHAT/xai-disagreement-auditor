# XAI Disagreement Auditor

**When explanations disagree, which one do you trust?**

SHAP, LIME, Integrated Gradients and other post-hoc explainers often give *different answers for the same
prediction*. Researchers call this the "disagreement problem" (Krishna et al., 2022). This project turns it
into a working tool. It:

1. runs **five attribution methods** on the same prediction, with the same model, background data and baseline;
2. measures how much they agree using **six agreement metrics**;
3. combines those into a per-prediction **Explanation Trust Score (0–100)** and flags **contested features**;
4. checks the methods with **faithfulness tests** (PGI, PGU, deletion curves) and against **ground truth** on
   synthetic data where the real drivers are known;
5. runs a **controlled experiment** showing that feature correlation erodes agreement, and that
   **agreement predicts correctness (r ≈ +0.61)**.

It includes an interactive dashboard, a REST API, a CLI, 33 unit tests, Docker and CI.

AIPI 590: Emerging Trends in Explainable AI · Duke University · Fall 2026 · Project 1

---

## Quick start

**Requirements:** Python 3.10 or newer. Check with `python3 --version`; if you need it, download it from
[python.org](https://www.python.org/downloads/). No GPU or internet access is needed once the dependencies are
installed.

### macOS / Linux
```bash
cd xai-disagreement-auditor
./run.sh
```
On a Mac you can also double-click **`Start Dashboard.command`**. If macOS blocks it, right-click it, choose
**Open**, then **Open** again.

### Windows
Double-click **`run.bat`**, or run it from a terminal.

The first run creates a private virtual environment in `.venv/` and installs everything, which takes 1–2 minutes.
After that it starts in a few seconds. The dashboard opens at **http://localhost:8501**.

| Command | What it does |
|---|---|
| `./run.sh` | Dashboard at http://localhost:8501 |
| `./run.sh api` | REST API at http://localhost:8000 (interactive docs at `/docs`) |
| `./run.sh test` | Run the 33 unit and integration tests |
| `./run.sh bench` | Re-run the ground-truth benchmark and write `results/` |
| `docker compose up --build` | Dashboard on :8501 and API on :8000, in containers |

---

## Dashboard tour

| Page | What you see |
|---|---|
| **Overview** | The question, how the auditor works, the current dataset and model, and the five methods |
| **Instance Audit** | One prediction: its Trust Score, auto-generated findings, an **explanation fingerprint** heatmap (share and rank per method), side-by-side bars, agreement matrix, pairwise metrics, consensus explanation, faithfulness, ground-truth check, a **what-if mode** with sliders, and JSON/CSV export |
| **Global Audit** | Distribution of trust across many predictions, trust vs confidence and errors, the mean agreement matrix, which features cause disagreement, the most contested predictions (one click opens them), and a **model-family comparison** |
| **Who's Right?** | PGI/PGU bars against a random baseline, deletion curves, the ground-truth check, an overall scorecard, and caveats |
| **Ground-Truth Lab** | The controlled experiment: correlation ρ from 0 to 0.95 × 4 models × 5 methods, and whether disagreement signals error |
| **Methods & Metrics** | Definitions, formulas, limitations and references |

Use the sidebar to switch between **3 built-in datasets** (Credit Risk (synthetic, ground truth known), Breast Cancer,
Diabetes), **4 models** (Logistic Regression, Random Forest, Gradient Boosting, MLP), top-k, and more. You can
also **upload your own CSV**. Try `sample_data/wine_example.csv`.

---

## Headline results

From `results/benchmark_summary.md`, which you can regenerate with `./run.sh bench`:

- **Agreement predicts correctness.** Across 400 predictions, the Trust Score correlates with true-driver precision
  at **r = +0.61**. When methods disagree, explanations really are wrong more often.
- **Correlation erodes agreement for non-linear models.** From ρ = 0 to ρ = 0.95, mean trust falls by about
  **10 points** for Random Forest, Gradient Boosting and the MLP, while Logistic Regression stays at about **94**.
- **Correctness falls too.** Mean true-driver precision drops by about **17 points** as correlation rises.
- **Method ranking (true-driver precision).** LIME and Shapley recover the true drivers most reliably. The local
  gradient method (SmoothGrad × Input) is the weakest, especially on tree models, whose outputs are piecewise constant.
- **Faithfulness vs truth.** Occlusion wins removal-based faithfulness tests, but those tests share its removal
  operation. On ground truth it ranks mid-table. The dashboard points this out.

---

## Project structure

```
xai-disagreement-auditor/
├── xaudit/                  core library (no UI code)
│   ├── explainers.py        5 attribution methods, vectorised NumPy, from scratch
│   ├── agreement.py         6 agreement metrics, Trust Score, consensus & contested features
│   ├── faithfulness.py      PGI, PGU, deletion curves (single batched model call)
│   ├── audit.py             instance and global audits, ground-truth scoring, plain-English summaries
│   ├── benchmark.py         ground-truth correlation experiment
│   ├── data.py              built-in datasets, synthetic generators, CSV upload cleaning
│   ├── models.py            model zoo and training
│   ├── config.py            constants and tunable budgets (env-var overridable)
│   └── cli.py               `python -m xaudit.cli benchmark | audit`
├── app/                     Streamlit dashboard
│   ├── streamlit_app.py     entry point, sidebar, navigation
│   ├── views/               one module per page
│   ├── charts.py, ui.py     Plotly figures, design tokens, CSS
│   └── state.py             caching and session config
├── api/main.py              FastAPI service
├── tests/                   pytest suite (explainer maths, metrics, API, end-to-end)
├── results/                 precomputed benchmark (loaded by the Ground-Truth Lab)
├── docs/                    METHODOLOGY.md, RESPONSIBLE_USE.md, PRESENTATION_GUIDE.md, poster/
├── sample_data/             example CSV for the upload feature
├── run.sh / run.bat / Start Dashboard.command
├── Dockerfile, docker-compose.yml, .github/workflows/ci.yml
└── requirements.txt, requirements-dev.txt, pyproject.toml
```

## API examples

```bash
./run.sh api   # then, in another terminal:

curl localhost:8000/health
curl -X POST localhost:8000/audit/instance -H 'content-type: application/json' \
     -d '{"dataset":"breast_cancer","model":"random_forest","index":7,"k":5}'
curl -X POST localhost:8000/audit/custom -H 'content-type: application/json' \
     -d '{"dataset":"credit_risk","model":"gradient_boosting","features":{"debt_to_income":0.55,"num_late_payments":5}}'
curl -X POST localhost:8000/audit/global -H 'content-type: application/json' \
     -d '{"dataset":"diabetes","model":"mlp","n":30}'
```

## Configuration

Explainer budgets can be changed with environment variables, for example
`XAUDIT_SHAPLEY_PERMUTATIONS=512 ./run.sh`. The variables are `XAUDIT_SHAPLEY_PERMUTATIONS` (256),
`XAUDIT_LIME_SAMPLES` (1500), `XAUDIT_IG_STEPS` (48), `XAUDIT_SMOOTHGRAD_SAMPLES` (24),
`XAUDIT_BACKGROUND_SIZE` (100), `XAUDIT_MAX_UPLOAD_ROWS` (20000) and `XAUDIT_MAX_FEATURES` (60).

## Troubleshooting

| Problem | Fix |
|---|---|
| `Python 3.10+ is required` | Install Python from python.org, then run `./run.sh` again |
| `permission denied: ./run.sh` | `chmod +x run.sh "Start Dashboard.command"` |
| Port 8501 already in use | `PORT=8502 ./run.sh` |
| Browser didn't open | Go to http://localhost:8501 yourself |
| Want a clean reinstall | Delete the `.venv` folder and run again |

## Responsible use

Who this is for, how it can be misused, and what has to be true before anyone relies on it in production:
[`docs/RESPONSIBLE_USE.md`](docs/RESPONSIBLE_USE.md). The short version: a high Trust Score means the methods
agree, not that the model is accurate, fair or lawful. Agreement is a warning signal when it is absent, not a
certificate when it is present.

## Poster

Two print-ready PDFs in `docs/poster/`: `XAI_Disagreement_Auditor_poster.pdf` (portrait, 36in x 48in)
and `XAI_Disagreement_Auditor_poster_landscape.pdf` (48in x 36in), both with a QR code to this repository.
`XAI_Disagreement_Auditor_report.pdf` is a two-page companion summary.
Regenerate with:

```bash
python3 scripts/make_poster_figures.py        # figures, drawn from the live code and results
python3 scripts/build_poster.py both https://github.com/<you>/xai-disagreement-auditor
# "portrait" | "landscape" | "both"; the URL is optional and adds a QR code
```

## Citations and AI use

All five explainers and all metrics are implemented in this repository. The methods follow the papers listed in
`docs/METHODOLOGY.md` and on the Methods page. Built-in datasets come from scikit-learn (UCI Breast Cancer
Wisconsin; Efron et al. diabetes) or are generated synthetically. Development was assisted by Claude (Anthropic),
in line with the course policy on AI use.
