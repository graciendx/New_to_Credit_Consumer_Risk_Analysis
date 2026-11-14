# Reproduction and verification

## Verified environment

Python 3.12; NumPy 2.3.5; pandas 2.2.3; scikit-learn 1.8.0; XGBoost 3.0.5; Matplotlib 3.10.8. Direct requirements are pinned. The evaluation report records the actual runtime versions.

## Checks completed

A fresh HTTPS download of the pinned public source was verified against the recorded SHA-256 before running the analysis.

- Thirteen behavioral tests pass, covering feature exclusions, stratified disjoint splits, training-only imputation, frozen thresholds, reporting boundaries, source integrity, reproduction guards and Tableau band ordering.
- The complete original CSV runs through all four model pipelines and reproduces the published metrics.
- An independent fresh run matches source fingerprints, split counts, frozen selection and held-out metrics within the declared tolerance.
- The notebook contains seven executed code cells and four embedded PNG figure previews. Its format validates against the notebook schema and contains no error outputs.
- Saved notebook outputs were captured sequentially with IPython because the publication host prohibits kernel sockets. The checked-in execution helper also supports a fresh Jupyter kernel, which is the default mode used in GitHub Actions.
- All four vector figures and their PNG previews were rendered and visually inspected for labels, clipping and legibility.
- The local Tableau CSV export completes. Workbook XML parses, and referenced source columns are checked against the generated input. Manual-sort values match the current exported bands, including no-score and unknown groups. Captions identify the annual-income ratio and student loan balance. Native Tableau Desktop rendering is not claimed.

## Automated checks

GitHub Actions installs the pinned analysis and notebook dependencies, fetches the pinned source into the runner, runs all tests, regenerates the original-data report, compares it with the published evaluation, runs the separate software demonstration and executes the notebook in a fresh kernel.

## Reproduce the numerical comparison

```sh
python -m ntc_risk --output outputs/reproduced
python scripts/verify_reproduction.py --actual outputs/reproduced
```

This verifies the reviewed retrospective analysis. Feature timing and prospective lending validity are separate research questions described in the methods.
