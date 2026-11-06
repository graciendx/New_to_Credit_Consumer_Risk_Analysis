# Quick start

## Read before running

1. Open the [research notebook](../notebooks/NTC_Consumer_Credit.ipynb) for the complete narrative and visible outputs.
2. Open the [result summary](../reports/summary.md) for the numerical comparison and four figures.
3. Review the [data dictionary](../data/data_dictionary.md) and [method](methodology_review.md) for analytical choices.

## Reproduce locally

Use Python 3.12. Open the repository folder in a terminal. Create `.venv`, activate it, install `requirements.txt` and run the command in the main README. To avoid activation entirely, use the environment's executable directly:

| System | Python executable after creating `.venv` |
| --- | --- |
| macOS / Linux | `.venv/bin/python` |
| Windows | `.venv\Scripts\python.exe` |

For example, on Windows:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe scripts/reproduce.py
```

This avoids PowerShell execution-policy changes. The reproduction helper downloads the pinned source once and verifies its checksum. Core analysis uses CPU estimators with one worker and requires no credentials or network access after dependencies and the source file are downloaded.

## Notebook environment

Install `requirements-notebook.txt` in the same environment. In VS Code, select `.venv` with the notebook kernel picker. In JupyterLab, select a kernel using that environment. Run `python scripts/fetch_data.py` once before opening the notebook. The notebook locates the repository root from its working directory and uses relative file paths.

For automated execution, run `python scripts/execute_notebook.py`. It uses the invoking Python environment for a fresh kernel. If the host explicitly prohibits local kernel sockets, use `--in-process` to execute cells sequentially with IPython output capture. The notebook records which mode produced its saved outputs; errors fail the command.

## Expected outputs

`reports/` contains `evaluation.json`, validation comparison and threshold CSVs, test summary, data profile, schema profile, FICO coverage, four SVG figures, PNG notebook previews and `summary.md`. `data/tableau_input.csv` is a generated local reporting file and is ignored by Git.

## Troubleshooting

- **Missing module:** install requirements using the same Python executable used to run the project.
- **Missing source file:** run `python scripts/fetch_data.py`, or pass `--data` with a verified local CSV path.
- **Source download unavailable:** download the file from the pinned source link in `data/README.md`, save it to `data/source/v_credit_data_NTC_v51825.csv`, then run the fetch helper to verify its checksum without downloading again.
- **Schema error:** keep the original header and outcome encoding. The validator reports the failed field instead of silently recoding it.
- **Different results:** use the pinned requirements and seed 42; inspect dependency versions and the source fingerprint in `evaluation.json`.
- **XGBoost installation:** use a 64-bit Python 3.12 environment on a supported platform. `--without-xgboost` is an explicit baseline-only alternative; its selected model can differ from the full published run.
- **Tableau connection:** regenerate the local input and follow the [Tableau guide](tableau.md).

## Verify a fresh run

```sh
python -m ntc_risk --output outputs/reproduced
python scripts/verify_reproduction.py --actual outputs/reproduced
```

The check compares source fingerprints, split membership counts, the frozen selection and held-out metrics to the published result, with a small numerical tolerance.
