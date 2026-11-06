# Analytical brief

## Question

How can limited-history credit profiles be ranked for analytical review while making the consequences of a flag visible?

## Data and model selection

The original project CSV contains 2,000 profiles, 283 recorded charge-offs and 830 missing FICO values. The reviewed implementation retains all rows, explicitly represents missing FICO and excludes outcome-related fields. The fixed split is 1,400 / 300 / 300.

Validation average precision selects XGBoost (0.633), compared with Random Forest (0.614), logistic regression (0.508) and the no-skill baseline (0.143). This is a ranking-first research objective. Random Forest has higher validation precision at its own selected threshold; the selection rule is not a claim that XGBoost is best for every business cost structure.

## Held-out result

At the frozen 0.35 threshold, XGBoost identifies 34 of 42 positive test outcomes. It flags 75 profiles, 41 of which have no recorded charge-off. Recall is 81.0%, precision 45.3%, ROC AUC 0.917 and average precision 0.563. The no-skill average precision is the test positive share, 0.140.

## Decision implication

The output exposes review capacity and missed-case trade-offs. A team with different review costs or recall requirements would need to specify those objectives before model and threshold selection. The project does not infer financial value from classification metrics alone.

## Inspect the evidence

- [Generated report](../reports/summary.md)
- [Validation comparison](../reports/validation_comparison.csv)
- [Threshold table](../reports/validation_thresholds.csv)
- [Held-out metrics](../reports/test_summary.csv)
- [Executed notebook](../notebooks/NTC_Consumer_Credit.ipynb)

The original project uses a historical-status outcome. The current result is a retrospective reproduction on the pinned educational dataset.
