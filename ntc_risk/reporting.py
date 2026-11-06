"""Aggregate research reports and static figures; no applicant-level score exports."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

LABELS = {
    'dummy_prior': 'No-skill baseline', 'logistic_regression': 'Logistic regression',
    'random_forest': 'Random Forest', 'xgboost': 'XGBoost',
}
INK, BLUE, TEAL, GRAY = '#243447', '#315b86', '#318b83', '#a5afb9'


def profile_data(frame):
    """Describe source coverage without imputing, filtering or exposing records."""
    missing = frame['FICO_V'].isna()
    profile = {
        'n_rows': len(frame), 'n_columns': len(frame.columns),
        'positive_outcomes': int(frame['Ever_ChargeOff'].sum()),
        'observed_chargeoff_share': float(frame['Ever_ChargeOff'].mean()),
        'missing_fico_count': int(missing.sum()), 'missing_fico_share': float(missing.mean()),
        'duplicate_full_rows': int(frame.duplicated().sum()),
    }
    schema = pd.DataFrame({
        'column': frame.columns, 'dtype': [str(t) for t in frame.dtypes],
        'missing_count': frame.isna().sum().to_numpy(),
        'missing_share': frame.isna().mean().to_numpy(),
        'distinct_observed_values': frame.nunique(dropna=True).to_numpy(),
    })
    groups = frame.assign(fico_group=np.where(missing, 'No FICO score', 'FICO score observed'))
    coverage = groups.groupby('fico_group', sort=True)['Ever_ChargeOff'].agg(
        n='size', positive='sum', observed_share='mean',
    ).reset_index()
    return profile, schema, coverage


def _save(fig, path):
    fig.savefig(path, format='svg', bbox_inches='tight', metadata={'Date': None})
    # Raster previews preserve typography in notebook hosts that sanitize inline SVG styles.
    fig.savefig(path.with_suffix('.png'), dpi=160, bbox_inches='tight')
    plt.close(fig)


def _figures(profile, coverage, report, comparison, thresholds, y_test, scores, directory):
    directory.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({
        'font.family': 'DejaVu Sans', 'font.size': 10, 'text.color': INK,
        'axes.labelcolor': INK, 'axes.spines.top': False, 'axes.spines.right': False,
        'svg.fonttype': 'none', 'svg.hashsalt': 'ntc-credit-risk', 'figure.facecolor': 'white',
    }):
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout='constrained')
        counts = [profile['n_rows']-profile['missing_fico_count'], profile['missing_fico_count']]
        axes[0].bar(['FICO observed', 'No FICO score'], counts, color=[BLUE, TEAL], width=.55)
        for i, n in enumerate(counts):
            axes[0].text(i, n+20, f'{n:,} ({n/profile["n_rows"]:.1%})', ha='center')
        axes[0].set(title='Credit-score coverage', ylabel='Profiles', ylim=(0, max(counts)*1.2))
        axes[1].bar(coverage['fico_group'], coverage['observed_share'], color=[BLUE, TEAL], width=.55)
        for i, row in enumerate(coverage.itertuples()):
            axes[1].text(i, row.observed_share+.005,
                         f'{row.observed_share:.1%}\n{row.positive}/{row.n} profiles', ha='center')
        axes[1].set(title='Recorded outcome by score coverage', ylabel='Observed charge-off share',
                    ylim=(0, max(coverage['observed_share'])*1.4))
        axes[1].yaxis.set_major_formatter(PercentFormatter(1))
        fig.suptitle('Original project dataset | descriptive coverage', fontweight='bold')
        _save(fig, directory/'data_profile.svg')

        ordered = comparison.sort_values('average_precision')
        fig, ax = plt.subplots(figsize=(8.5, 3.6), layout='constrained')
        colors = [TEAL if n == report['selected_model'] else GRAY if n == 'dummy_prior' else BLUE
                  for n in ordered['model']]
        bars = ax.barh([LABELS[n] for n in ordered['model']], ordered['average_precision'], color=colors)
        ax.bar_label(bars, labels=[f'{x:.3f}' for x in ordered['average_precision']], padding=7)
        ax.set(xlim=(0, min(1, ordered['average_precision'].max()+.15)),
               xlabel='Validation average precision', title='Model selection | common validation sample')
        ax.text(0, -.25, 'Teal marks the model selected before test evaluation.', transform=ax.transAxes,
                fontsize=9, color=INK)
        _save(fig, directory/'validation_comparison.svg')

        table = thresholds[thresholds['model'] == report['selected_model']]
        fig, ax = plt.subplots(figsize=(8.5, 4.4), layout='constrained')
        for metric, label, color in [('precision','Precision',BLUE),('recall','Recall',TEAL),
                                     ('flag_rate','Flagged share',GRAY)]:
            ax.plot(table['threshold'], table[metric], marker='o', markersize=3, label=label, color=color)
        ax.axvline(report['selected_threshold'], color=INK, linestyle='--', linewidth=1,
                   label=f'Frozen threshold: {report["selected_threshold"]:.2f}')
        ax.set(xlabel='Risk-score threshold', ylabel='Share', ylim=(0, 1.05),
               title=f'{LABELS[report["selected_model"]]} | validation threshold trade-offs')
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(.5, -.18), ncol=2, fontsize=9)
        _save(fig, directory/'validation_thresholds.svg')

        metrics = report['selected_model_test']
        fpr, tpr, _ = roc_curve(y_test, scores)
        precision, recall, _ = precision_recall_curve(y_test, scores)
        fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8), layout='constrained')
        axes[0].plot(fpr, tpr, color=BLUE, label=f'ROC AUC {metrics["roc_auc"]:.3f}')
        axes[0].plot([0,1], [0,1], '--', color=GRAY, label='No-skill reference')
        axes[0].set(xlabel='False-positive rate', ylabel='True-positive rate', title='Ranking | ROC curve')
        axes[0].legend(frameon=False, loc='lower right', fontsize=8)
        axes[1].step(recall, precision, where='post', color=TEAL,
                     label=f'Average precision {metrics["average_precision"]:.3f}')
        axes[1].axhline(metrics['prevalence'], linestyle='--', color=GRAY,
                       label=f'Positive share {metrics["prevalence"]:.1%}')
        axes[1].set(xlabel='Recall', ylabel='Precision', ylim=(0,1.05), title='Ranking | precision–recall')
        axes[1].legend(frameon=False, loc='upper right', fontsize=8)
        matrix = np.array([[metrics['tn'],metrics['fp']],[metrics['fn'],metrics['tp']]])
        axes[2].imshow(matrix, cmap='Blues', vmin=0, vmax=max(matrix.max(),1))
        for i in range(2):
            for j in range(2):
                axes[2].text(j, i, str(matrix[i,j]), ha='center', va='center', fontsize=16,
                             color='white' if matrix[i,j] > matrix.max()*.6 else INK)
        axes[2].set(xticks=[0,1], xticklabels=['Not flagged','Flagged'], yticks=[0,1],
                    yticklabels=['No charge-off','Charge-off'], ylabel='Recorded outcome',
                    title=f'Frozen threshold {report["selected_threshold"]:.2f}')
        fig.suptitle(f'{LABELS[report["selected_model"]]} | held-out test sample (n={metrics["n"]:,})',
                     fontweight='bold')
        _save(fig, directory/'test_diagnostics.svg')


def write_report(frame, report, comparison, thresholds, y_test, scores, output_dir, *, make_figures=True):
    """Persist aggregate evidence after model and threshold selection are frozen."""
    output_dir = Path(output_dir)
    profile, schema, coverage = profile_data(frame)
    (output_dir/'data_profile.json').write_text(json.dumps(profile, indent=2, allow_nan=False)+'\n')
    schema.to_csv(output_dir/'schema_profile.csv', index=False)
    coverage.to_csv(output_dir/'fico_coverage.csv', index=False)
    summary = pd.DataFrame([
        {'model':report['selected_model'], **report['selected_model_test']},
        {'model':'dummy_prior', **report['dummy_prior_test_at_0_50']},
    ])
    summary.to_csv(output_dir/'test_summary.csv', index=False)
    if make_figures:
        _figures(profile, coverage, report, comparison, thresholds, y_test, scores, output_dir/'figures')
    m = report['selected_model_test']
    lines = [
        '# Reproduced analytical results', '',
        f'Dataset: {report["dataset_kind"]}. {report["status"]}', '',
        '## Data and design', '',
        f'- {profile["n_rows"]:,} profiles; {profile["positive_outcomes"]:,} recorded positive outcomes '
        f'({profile["observed_chargeoff_share"]:.2%}).',
        f'- Missing FICO: {profile["missing_fico_count"]:,} ({profile["missing_fico_share"]:.1%}).',
        f'- Split sizes: {report["splits"]["train"]:,} train / {report["splits"]["validation"]:,} validation '
        f'/ {report["splits"]["test"]:,} test.',
        '- Select by validation average precision; choose each model’s threshold by validation precision '
        'subject to the declared recall and flag-count constraints.', '',
        '## Validation comparison', '',
        '| Model | Average precision | ROC AUC | Threshold | Precision | Recall | Flagged share |',
        '| --- | --- | --- | --- | --- | --- | --- |',
    ]
    for row in comparison.itertuples():
        lines.append(f'| {LABELS[row.model]} | {row.average_precision:.3f} | {row.roc_auc:.3f} | '
                     f'{row.threshold:.2f} | {row.precision:.1%} | {row.recall:.1%} | {row.flag_rate:.1%} |')
    lines += ['', '## Held-out test evaluation', '',
        f'Frozen choice: **{LABELS[report["selected_model"]]}**, threshold **{report["selected_threshold"]:.2f}**.', '',
        f'- Average precision: **{m["average_precision"]:.3f}** '
        f'(no-skill baseline: {report["dummy_prior_test_at_0_50"]["average_precision"]:.3f}).',
        f'- ROC AUC: **{m["roc_auc"]:.3f}**.',
        f'- Flagged **{m["tp"]+m["fp"]} of {m["n"]}** profiles ({m["flag_rate"]:.1%}).',
        f'- Identified **{m["tp"]} of {m["tp"]+m["fn"]}** positive outcomes ({m["recall"]:.1%} recall).',
        f'- **{m["precision"]:.1%} precision**; {m["fp"]} false-positive flags and {m["fn"]} missed positives.', '',
        'The test sample evaluates only the validation-selected model and the no-skill baseline. '
        'It is not used to rank or retune the remaining candidates.', '',
        '## Interpretation', '',
        'These are retrospective classification results on the project dataset. The positive label is a '
        'recorded charge-off; a model flag is an analytical classification, not a credit decision. '
        'See the repository methods for feature eligibility and timing assumptions.', '',
        '## Figures', '',
    ]
    if make_figures:
        for title, name in [('Data coverage','data_profile'),('Validation model comparison','validation_comparison'),
                            ('Validation threshold trade-offs','validation_thresholds'),
                            ('Held-out test diagnostics','test_diagnostics')]:
            lines += [f'![{title}](figures/{name}.svg)', '']
    (output_dir/'summary.md').write_text('\n'.join(lines)+'\n')
