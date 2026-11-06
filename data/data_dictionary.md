# Data dictionary and feature policy

Descriptions follow source column names and the original project context. They are documented interpretations, not a newly recovered lender data dictionary. Monetary currency, observation dates and a prospective outcome window are not defined in the upstream files.

| Source field | Representation | Interpretation | Role |
| --- | --- | --- | --- |
| `No_Hit` | Binary | Source no-hit indicator | Input |
| `Age` | Years | Recorded age | Excluded |
| `Ever_Bankruptcy` | Binary | Recorded bankruptcy history | Input |
| `Months_Since_Bankruptcy` | Months | Elapsed time since recorded bankruptcy; source zero values retained | Input |
| `Ever_Collection` | Binary | Recorded collection history | Input |
| `Unpaid_Collection_Trades` | Count | Unpaid collection trade count | Input |
| `Unpaid_Collection_Balance` | Source monetary units | Unpaid collection balance | Input |
| `Counseling_Last_12M` | Binary | Counseling indicator over named 12-month window | Input |
| `Unpaid_CO_84M` | Source numeric indicator | Prior unpaid charge-off field; timing relative to outcome is ambiguous | Excluded |
| `Delinq_90D_12M` | Binary | Named 90-day delinquency indicator within 12 months | Input |
| `Delinq_60D_12M` | Binary | Named 60-day delinquency indicator within 12 months | Input |
| `Delinq_60D_Curr` | Binary | Named current 60-day delinquency indicator | Input |
| `Delinq_30D_3M` | Binary | Named 30-day delinquency indicator within 3 months | Input |
| `Delinq_30D_Curr` | Binary | Named current 30-day delinquency indicator | Input |
| `Checking_Asset` | Source monetary units | Checking-account assets | Input |
| `Card_Inquiry_12M` | Count | Card inquiries over named 12-month window | Input |
| `Card_Open_24M` | Count | Cards opened over named 24-month window | Input |
| `Personal_Loan` | Binary | Personal-loan indicator | Input |
| `Unsecured_Debt` | Source monetary units | Unsecured debt balance | Input |
| `Revolving_Util` | Ratio | Revolving utilization; source values are fractional, not percentage points | Input |
| `Income` | Source monetary units | Income field; annual interpretation follows the original project | Input |
| `Auth_User_Only` | Binary | Authorized-user-only indicator | Input |
| `Student` | Binary | Student indicator | Input |
| `Student_Loan_Amt` | Source monetary units | Student-loan amount | Input |
| `Ever_ChargeOff` | Binary | Recorded historical charge-off status; positive class is 1 | Outcome |
| `ChargeOff_Balance` | Source monetary units | Outcome-associated balance | Excluded |
| `FICO_V` | Score / missing | Recorded FICO score; missingness is retained as information | Input |

## Derived model inputs

- `FICO_Missing`: 1 where `FICO_V` is missing.
- `Unsecured_Debt_to_Annual_Income`: unsecured debt divided by positive income; zero denominators become missing. This is a debt-stock ratio, not monthly debt-service DTI.

The 23 allowed source inputs and two derived columns form 25 declared model inputs. Median imputation and missing-value indicators are fitted within each training pipeline; logistic regression also uses training-fitted scaling. Unknown extra columns and Tableau bands are never added automatically.

## Outcome boundary

`ChargeOff_Balance` and `Unpaid_CO_84M` are excluded to avoid outcome-related information without a separately established observation window. Age is excluded from model inputs. The remaining variables support retrospective study under the project’s declared feature policy; the source cannot establish that every measurement predates a future credit event.

## Tableau compatibility

The workbook retains legacy `DTI` field names for compatibility. Their documented meaning is unsecured debt / annual income. `Observed_Chargeoff` is copied from the recorded label and is never described as a model score.
