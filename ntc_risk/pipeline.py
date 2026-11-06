"""Retrospective classification workflow. This module makes no lending decisions."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET = 'Ever_ChargeOff'
# Explicit inclusion prevents outcome fields, IDs and later reporting bands entering a model.
FEATURES = (
    'No_Hit', 'Ever_Bankruptcy', 'Months_Since_Bankruptcy', 'Ever_Collection',
    'Unpaid_Collection_Trades', 'Unpaid_Collection_Balance', 'Counseling_Last_12M',
    'Delinq_90D_12M', 'Delinq_60D_12M', 'Delinq_60D_Curr', 'Delinq_30D_3M',
    'Delinq_30D_Curr', 'Checking_Asset', 'Card_Inquiry_12M', 'Card_Open_24M',
    'Personal_Loan', 'Unsecured_Debt', 'Revolving_Util', 'Income', 'Auth_User_Only',
    'Student', 'Student_Loan_Amt', 'FICO_V',
)
EXCLUDED = ('ChargeOff_Balance', 'Unpaid_CO_84M', 'Age')
BINARY = (
    'No_Hit', 'Ever_Bankruptcy', 'Ever_Collection', 'Counseling_Last_12M',
    'Delinq_90D_12M', 'Delinq_60D_12M', 'Delinq_60D_Curr', 'Delinq_30D_3M',
    'Delinq_30D_Curr', 'Personal_Loan', 'Auth_User_Only', 'Student',
)


@dataclass(frozen=True)
class EvaluationConfig:
    seed: int = 42
    minimum_recall: float = 0.50
    minimum_flags: int = 10
    thresholds: tuple[float, ...] = tuple(round(i / 100, 2) for i in range(5, 100, 5))

    def __post_init__(self):
        if not 0 <= self.minimum_recall <= 1:
            raise ValueError('minimum_recall must be between 0 and 1.')
        if not isinstance(self.minimum_flags, int) or self.minimum_flags < 1:
            raise ValueError('minimum_flags must be a positive integer.')
        if not self.thresholds or any(not np.isfinite(t) or not 0 <= t <= 1 for t in self.thresholds):
            raise ValueError('Thresholds must be finite values between 0 and 1.')


def prepare_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Validate the declared schema; keep missing numeric values for train-only imputation."""
    if frame.columns.duplicated().any():
        raise ValueError('Duplicate column names are not supported.')
    missing = sorted(set(FEATURES + (TARGET,)) - set(frame.columns))
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    selected = frame.loc[:, [*FEATURES, TARGET]].copy()
    for column in selected:
        try:
            selected[column] = pd.to_numeric(selected[column], errors='raise')
        except (ValueError, TypeError) as exc:
            raise ValueError(f'{column} must contain numeric values.') from exc
    if not selected[TARGET].isin([0, 1]).all():
        raise ValueError('Ever_ChargeOff must contain only 0 and 1, without missing labels.')
    if set(selected[TARGET].unique()) != {0, 1}:
        raise ValueError('Both outcome classes are required.')
    for column in BINARY:
        if not selected[column].dropna().isin([0, 1]).all():
            raise ValueError(f'{column} must be binary when observed.')
    values = selected.drop(columns=TARGET)
    if np.isinf(values.to_numpy(dtype=float)).any():
        raise ValueError('Infinite input values are not valid; inspect the source data.')
    if (values.drop(columns='FICO_V') < 0).any().any():
        raise ValueError('Declared count, amount and ratio fields cannot be negative.')
    fico = values['FICO_V'].dropna()
    if not fico.between(300, 850).all():
        raise ValueError('Observed FICO_V must be between 300 and 850; use missing values for no score.')
    if values.isna().all().any():
        raise ValueError('A required feature has no observed values.')
    if values.duplicated().any():
        raise ValueError('Duplicate feature profiles detected; resolve records or use a group-aware split.')
    y = selected[TARGET].astype(int)
    if len(selected) < 100 or y.value_counts().min() < 20:
        raise ValueError('The three-way evaluation requires at least 100 rows and 20 per outcome class.')
    # Explicitly preserve FICO missingness instead of encoding an impossible score of 999.
    values['FICO_Missing'] = values['FICO_V'].isna().astype(int)
    # This is a debt-stock / annual-income ratio, not a monthly debt-service DTI measure.
    values['Unsecured_Debt_to_Annual_Income'] = values['Unsecured_Debt'].div(
        values['Income'].where(values['Income'] > 0)
    ).replace([np.inf, -np.inf], np.nan)
    return values, y


def split_data(X, y, seed=42):
    """Split before estimating any preprocessing statistics; preserve class proportions."""
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_holdout, y_holdout, test_size=0.50, random_state=seed, stratify=y_holdout,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate_scores(y, scores, threshold):
    """Use one explicit threshold for every classifier and evaluation split."""
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError('Scores must be finite values between 0 and 1.')
    if not 0 <= threshold <= 1:
        raise ValueError('threshold must be between 0 and 1.')
    y = np.asarray(y, dtype=int)
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    return {
        'threshold': float(threshold), 'n': len(y), 'prevalence': float(np.mean(y)),
        'roc_auc': float(roc_auc_score(y, scores)),
        'average_precision': float(average_precision_score(y, scores)),
        'brier_score': float(brier_score_loss(y, scores)),
        'accuracy': float(accuracy_score(y, predictions)),
        'precision': float(precision_score(y, predictions, zero_division=0)),
        'recall': float(recall_score(y, predictions, zero_division=0)),
        'flag_rate': float(np.mean(predictions)),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }


def choose_threshold(y_validation, validation_scores, config):
    """Illustrative validation-only policy; it is not an estimated lending profit rule."""
    table = pd.DataFrame([
        evaluate_scores(y_validation, validation_scores, threshold)
        for threshold in config.thresholds
    ])
    eligible = table[
        (table['recall'] >= config.minimum_recall)
        & ((table['tp'] + table['fp']) >= config.minimum_flags)
    ]
    if eligible.empty:
        raise ValueError('No validation threshold meets the declared recall and flag-count constraints.')
    best = eligible.sort_values(
        ['precision', 'recall', 'threshold'], ascending=[False, False, True], kind='stable',
    ).iloc[0]
    return float(best['threshold']), table


def build_models(y_train, seed=42, include_xgboost=True):
    estimators = {
        'dummy_prior': DummyClassifier(strategy='prior'),
        'logistic_regression': LogisticRegression(class_weight='balanced', max_iter=2000, random_state=seed),
        'random_forest': RandomForestClassifier(
            n_estimators=100, max_depth=6, class_weight='balanced', random_state=seed, n_jobs=1,
        ),
    }
    if include_xgboost:
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError('Install the project dependencies, or explicitly use --without-xgboost.') from exc
        positives = int(np.sum(y_train))
        estimators['xgboost'] = XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            scale_pos_weight=(len(y_train) - positives) / positives,
            random_state=seed, eval_metric='logloss', tree_method='hist', n_jobs=1,
        )
    models = {}
    for name, estimator in estimators.items():
        steps = [('imputer', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True))]
        if name == 'logistic_regression':
            steps.append(('scaler', StandardScaler()))
        models[name] = Pipeline([*steps, ('classifier', estimator)])
    return models


def run_analysis(frame, output_dir, *, dataset_kind, config=None, include_xgboost=True,
                 source_path=None, make_figures=True):
    if dataset_kind not in {'demonstration', 'coursework'}:
        raise ValueError('dataset_kind must identify demonstration or coursework input.')
    config = config or EvaluationConfig()
    X, y = prepare_data(frame)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, config.seed)
    models = build_models(y_train, config.seed, include_xgboost)
    validation = []
    threshold_tables = []
    selected_thresholds = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        scores = model.predict_proba(X_val)[:, 1]
        threshold, table = choose_threshold(y_val, scores, config)
        selected_thresholds[name] = threshold
        validation.append({'model': name, **evaluate_scores(y_val, scores, threshold)})
        threshold_tables.append(table.assign(model=name))
    validation_table = pd.DataFrame(validation).sort_values(
        ['average_precision', 'model'], ascending=[False, True], kind='stable',
    )
    # Freeze model choice and threshold before reading test scores. Do not refit afterward.
    selected_model = validation_table.iloc[0]['model']
    selected_threshold = selected_thresholds[selected_model]
    test_scores = models[selected_model].predict_proba(X_test)[:, 1]
    test_metrics = evaluate_scores(y_test, test_scores, selected_threshold)
    baseline_test = evaluate_scores(
        y_test, models['dummy_prior'].predict_proba(X_test)[:, 1], 0.50,
    )
    report = {
        'dataset_kind': dataset_kind,
        'status': 'Software demonstration on generated toy data.' if dataset_kind == 'demonstration'
                  else 'Reproduced retrospective evaluation on the original project dataset.',
        'config': asdict(config), 'n_rows': len(frame),
        'splits': {'train': len(y_train), 'validation': len(y_val), 'test': len(y_test)},
        'features': list(X.columns), 'excluded_outcome_or_review_fields': list(EXCLUDED),
        'selected_model': selected_model, 'selection_metric': 'validation average precision',
        'selected_threshold': selected_threshold,
        'selected_model_test': test_metrics, 'dummy_prior_test_at_0_50': baseline_test,
        'data_fingerprint_sha256': hashlib.sha256(
            pd.util.hash_pandas_object(frame, index=True).values.tobytes()
        ).hexdigest(),
        'environment': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'matplotlib']},
        'limitations': [
            'Ever_ChargeOff is a historical-status label; no prospective performance horizon is established.',
            'Feature timestamps and a data-generation specification are unavailable for the original project.',
            'Scores are uncalibrated. Brier loss is a diagnostic, not proof of calibration.',
            'No approval policy, expected loss, fairness certification or production suitability is inferred.',
        ],
    }
    if include_xgboost:
        report['environment']['xgboost'] = importlib.metadata.version('xgboost')
    if source_path is not None:
        report['source_file_sha256'] = hashlib.sha256(Path(source_path).read_bytes()).hexdigest()
    report['split_class_counts'] = {
        name: {'n': len(labels), 'positive': int(labels.sum()), 'negative': int(len(labels)-labels.sum())}
        for name, labels in [('train', y_train), ('validation', y_val), ('test', y_test)]
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_table.to_csv(output_dir / 'validation_comparison.csv', index=False)
    thresholds = pd.concat(threshold_tables, ignore_index=True)
    thresholds.to_csv(output_dir / 'validation_thresholds.csv', index=False)
    (output_dir / 'evaluation.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    from .reporting import write_report
    write_report(frame, report, validation_table, thresholds, y_test, test_scores,
                 output_dir, make_figures=make_figures)
    return report, validation_table


def make_demo_data(n=600, seed=42):
    """Create new toy inputs for software verification, never a reconstruction of coursework data."""
    if n < 200:
        raise ValueError('Use at least 200 demonstration rows.')
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({name: rng.binomial(1, 0.2, n) for name in BINARY})
    frame['Months_Since_Bankruptcy'] = rng.integers(0, 100, n)
    frame['Unpaid_Collection_Trades'] = rng.poisson(.3, n)
    frame['Unpaid_Collection_Balance'] = rng.exponential(150, n)
    frame['Checking_Asset'] = rng.exponential(1600, n)
    frame['Card_Inquiry_12M'] = rng.poisson(2, n)
    frame['Card_Open_24M'] = rng.poisson(1, n)
    frame['Unsecured_Debt'] = rng.exponential(3000, n)
    frame['Revolving_Util'] = rng.beta(2, 4, n)
    frame['Income'] = rng.uniform(18000, 80000, n)
    frame['Student_Loan_Amt'] = rng.exponential(12000, n)
    frame['FICO_V'] = rng.integers(300, 851, n).astype(float)
    frame.loc[rng.random(n) < .4, 'FICO_V'] = np.nan
    frame['Age'] = rng.integers(18, 26, n)
    logit = -2.7 + 2.5 * frame['Delinq_90D_12M'] + 1.4 * frame['Delinq_60D_Curr'] + frame['Revolving_Util']
    frame[TARGET] = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    # Deliberate leakage traps make it possible to test that the allowlist protects the model.
    frame['ChargeOff_Balance'] = np.where(frame[TARGET] == 1, 1000, np.nan)
    frame['Unpaid_CO_84M'] = frame[TARGET]
    return frame


def tableau_export(frame):
    """Create reporting bands while retaining no-score and zero-denominator cases."""
    prepare_data(frame)
    result = frame.copy()
    result['DTI'] = result['Unsecured_Debt'].div(result['Income'].where(result['Income'] > 0))
    result['Accounts'] = 1
    result['Observed_Chargeoff'] = result[TARGET]
    result['FICO'] = pd.cut(result['FICO_V'], [300, 580, 670, 740, 800, 851], right=False,
                            labels=['Poor', 'Fair', 'Good', 'Very Good', 'Exceptional']).astype(object).fillna('No score')
    starts = np.floor((result['FICO_V'] - 300) / 10) * 10 + 300
    result['FICO_Band'] = starts.map(lambda x: 'No score' if pd.isna(x) else f'{int(x)}-{min(int(x)+9,850)}')
    definitions = {
        'Utilization_Band': ('Revolving_Util', [0,.1,.3,.6,1,np.inf], ['<10%','10–<30%','30–<60%','60–<100%','≥100%']),
        'DTI_Band': ('DTI', [0,.36,.41,.43,.5,np.inf], ['<36%','36–<41%','41–<43%','43–<50%','≥50%']),
        'DTI_Band_10pct': ('DTI', [0,.01,.03,.05,.07,.1,np.inf], ['<1%','1–<3%','3–<5%','5–<7%','7–<10%','≥10%']),
        'LoanAmount_Band': ('Student_Loan_Amt', [0,1,1000,5000,10000,15000,np.inf], ['<1','1–<1K','1K–<5K','5K–<10K','10K–<15K','≥15K']),
        'Asset_Band': ('Checking_Asset', [0,500,1000,1500,2000,2500,np.inf], ['<500','500–<1000','1000–<1500','1500–<2000','2000–<2500','≥2500']),
    }
    for label, (column, bins, labels) in definitions.items():
        result[label] = pd.cut(result[column], bins, labels=labels, right=False).astype(object).fillna('Unknown')
    # Legacy names below preserve compatibility with the supplied Tableau workbook.
    result['DTI_Band_10pct'] = result['DTI_Band_10pct'].astype(str)
    return result


def load_csv(path):
    """Reject duplicate CSV headers before pandas can rename them silently."""
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        header = next(csv.reader(stream), [])
    if not header or len(header) != len(set(header)):
        raise ValueError('CSV headers must be nonempty and unique.')
    return pd.read_csv(path, encoding='utf-8-sig')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--data', type=Path, help='CSV input; defaults to the locally downloaded original project dataset.')
    source.add_argument('--demo', action='store_true', help='Run on newly generated verification data.')
    parser.add_argument('--output', type=Path, default=Path('outputs/latest'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--minimum-recall', type=float, default=.50)
    parser.add_argument('--minimum-flags', type=int, default=10)
    parser.add_argument('--without-xgboost', action='store_true')
    parser.add_argument('--tableau-export', type=Path, help='Optional local row-level CSV export; do not commit it.')
    args = parser.parse_args()
    if not args.demo and args.data is None:
        args.data = Path(__file__).resolve().parents[1] / 'data/source/v_credit_data_NTC_v51825.csv'
    if args.data and args.tableau_export and args.data.resolve() == args.tableau_export.resolve():
        parser.error('The Tableau export must not overwrite the input CSV.')
    if args.data and not args.data.is_file():
        parser.error(f'Data file not found: {args.data}. Run python scripts/fetch_data.py first, or specify --data.')
    try:
        frame = make_demo_data(seed=args.seed) if args.demo else load_csv(args.data)
        report, comparison = run_analysis(
            frame, args.output, dataset_kind='demonstration' if args.demo else 'coursework',
            config=EvaluationConfig(args.seed, args.minimum_recall, args.minimum_flags),
            include_xgboost=not args.without_xgboost,
            source_path=args.data,
        )
        if args.tableau_export:
            args.tableau_export.parent.mkdir(parents=True, exist_ok=True)
            tableau_export(frame).to_csv(args.tableau_export, index=False)
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(report['status'])
    print(comparison[['model', 'average_precision', 'threshold', 'precision', 'recall']].to_string(index=False))
    print(f"Validation-selected model: {report['selected_model']}; threshold: {report['selected_threshold']:.2f}")
    print(f'Results written to {args.output}. No applicant decisions or row-level predictions were exported.')


if __name__ == '__main__':
    main()
