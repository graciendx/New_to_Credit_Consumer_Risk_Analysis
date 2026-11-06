# Project data

The original input contains **2,000 rows and 27 columns** and remains in the shared project repository by Chloe Pham and Grace Ngo. `python scripts/fetch_data.py` downloads a checksum-verified local copy to `data/source/v_credit_data_NTC_v51825.csv`. Raw records are not republished in this repository.

- [Pinned upstream source](https://github.com/chloephammi/New-To-Credit-Consumer/blob/8b55c183e045e80b87ffb9e5efd0a1268443b7e4/v_credit_data_NTC_v51825.csv)
- Upstream commit: `8b55c183e045e80b87ffb9e5efd0a1268443b7e4`
- Git blob: `f615dd215a5ade27450279a8220598b126cc6e36`
- SHA-256: `8807ab09fdf399d825e92a56633515f86a33f14725c543f6304c32bbbd5ef33b`
- Outcome: `Ever_ChargeOff`; 283 positive labels (14.15%).
- Missing FICO: 830 profiles (41.5%).

The original project report describes synthetic educational profiles. The upstream README describes anonymized credit profiles; no generator specification or original field dictionary is supplied there. This publication retains the educational-project setting and does not characterize the file as a representative lender population. No additional data license is asserted. Original authorship and upstream provenance remain visible.

## Reproduce

From the repository root, run `python scripts/reproduce.py`. It fetches the pinned original input over HTTPS, verifies its SHA-256 and regenerates the report and local Tableau CSV. Once downloaded, `python -m ntc_risk --output reports` can run offline. No API key or Google Drive mount is needed.

[`data_dictionary.md`](data_dictionary.md) explains field roles and the reviewed feature policy. The generated [`schema_profile.csv`](../reports/schema_profile.csv) records types, missingness and observed cardinality.

## Tableau input

Run:

```sh
python -m ntc_risk --output reports --tableau-export data/tableau_input.csv
```

Then open [`tableau/NTC.twb`](../tableau/NTC.twb) in Tableau Desktop. Its relative connection points to `data/tableau_input.csv`. This export contains observed labels and reporting bands, not model predictions. Generated row-level exports are ignored by Git. The command runs locally and sends no records to an external service.
