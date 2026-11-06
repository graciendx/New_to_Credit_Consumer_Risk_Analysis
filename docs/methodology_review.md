# Methodological review

## Evidence basis

The original joint-project repository now supplies the complete CSV and original executable notebook. The local CSV is fetched unchanged and pinned to upstream commit `8b55c183e045e80b87ffb9e5efd0a1268443b7e4`. The reviewed workflow has been rerun on all 2,000 original profiles. Source hashes and the upstream map are recorded in `source_manifest.json` and `original_project.md`.

## Feature eligibility and target definition

The original code excludes `Ever_ChargeOff`, `FICO` and `No_Hit` when selecting model columns. It does not exclude `ChargeOff_Balance` or `Unpaid_CO_84M`. A balance associated with the outcome is not an eligible predictor without a separately established historical observation window. The corrected allowlist excludes both fields. It also excludes age, IDs, unknown columns and report-derived bands.

Other delinquency and collection fields still require a timing review before prospective use. The historical `Ever_ChargeOff` label does not define a future default horizon. Neither the original scores nor the refactor should be presented as validated underwriting probabilities.

## Preprocessing and sample composition

The original IQR routine repeatedly calculates cut-offs and drops rows on the entire dataset before splitting. It also iterates over numeric outcome-related fields. This can change class composition and allows holdout information to influence the analysis. The corrected implementation retains rare profiles and estimates imputation statistics only from training records.

The report calls the splits stratified; the saved code has no `stratify` argument. Original split counts are 1,172 training, 251 validation and 252 test rows after filtering. Saved counts contain 238 positive labels across those splits, or approximately 14.21%, while 14.15% refers to the unfiltered 283 positives among 2,000 records.

Missing FICO is represented by an indicator and train-fitted median imputation. A sentinel of 999 is no longer used. Duplicate feature profiles fail validation rather than leaking identical profiles across random splits. Real repeated customers would require an entity-aware split, which cannot be implemented without identifiers and a data dictionary.

## Evaluation reconciliation

Both original models use a 0.70 classification threshold on validation. Random Forest calls `.predict()` on test, while XGBoost applies 0.70 explicitly. The original test precision figures therefore cannot support a controlled threshold comparison. The saved XGBoost test output is 58.8% precision and 0.935 AUC, which differs from the slide figures of 45.9% and 0.939.

The refactor adds a no-skill baseline and logistic regression, uses identical split definitions and applies a single threshold function. Model selection uses validation average precision. Threshold choice maximizes validation precision subject to disclosed recall and flag-count constraints. The selected model and threshold are frozen before test scoring. No post-selection refit changes the scoring distribution.

The reproduced original-data run selects XGBoost by validation average precision (0.633). Its frozen threshold is 0.35; held-out average precision is 0.563 and ROC AUC is 0.917. Test precision is 45.3% and recall 81.0%. These figures come from the reviewed pipeline, not the legacy notebook. The separate toy-data demonstration remains a software check.

## Tableau and financial terminology

The original export assigns `Ever_ChargeOff` directly to `Predicted_Chargeoff_Score`. That is an observed label, not a prediction. The reviewed export uses `Observed_Chargeoff`. The workbook calculates observed shares and labels them accordingly. Stale embedded thumbnails were removed so old labels are not presented as a new render.

Numeric FICO calculations replace string comparisons. Missing scores receive a no-score category, and the minimum and maximum valid scores are retained. Zero income produces an unknown debt-to-income ratio, not infinity. The ratio is unsecured debt stock divided by annual income and must not be represented as monthly debt-service DTI. An observed charge-off share is not net credit loss, which requires loss amounts and an exposure denominator. The workbook's 20% comparison is explicitly illustrative.

## Scope of interpretation

- The original CSV is fetched locally; field meanings follow source names and project materials. A prospective study would need a validated data dictionary and availability time for each feature.
- Observation dates, an outcome horizon and entity identifiers for temporal or grouped evaluation.
- Calibration, stability and appropriate subgroup evaluation on suitable data.
- Exposure, loss severity, recovery and acquisition-value assumptions for any economic policy study.
- A native Tableau Desktop check after reconnecting the original data.

The publication reproduces the reviewed implementation on the original CSV and publishes its aggregate evidence. That retrospective reproduction does not establish prospective underwriting validity.
