# Tableau reporting guide

The reviewed workbook contains nine worksheets and one dashboard. It is an exploratory view of recorded outcomes, separate from the machine-learning evaluation.

## Connect

1. Run `python -m ntc_risk --output reports --tableau-export data/tableau_input.csv` from the repository root.
2. Open `tableau/NTC.twb` in Tableau Desktop.
3. If Tableau requests a data location, select the generated `data/tableau_input.csv` file and refresh the connection.

The workbook uses a relative `../data` connection. The original Google Drive and user-specific paths have been removed.

## Read the fields

- `Observed_Chargeoff` is the recorded `Ever_ChargeOff` indicator.
- Group averages of that indicator are observed charge-off shares, not predicted default probabilities.
- The legacy `DTI` fields use unsecured debt divided by annual income. They are not monthly debt-service DTI.
- Missing FICO remains a separate category; zero-income ratios use an unknown category.
- The 20% comparison in the original dashboard is an illustrative reference, not an estimated loss threshold.

The workbook XML, referenced fields and the generated CSV are checked programmatically. Native Tableau Desktop rendering remains a separate application check. The Python figures in `reports/` provide immediately inspectable, reproduced visual output without Tableau.
