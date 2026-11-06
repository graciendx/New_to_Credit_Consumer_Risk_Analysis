"""Behavioral checks for leakage boundaries, threshold policy and reproducibility."""
import json
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from ntc_risk.pipeline import (
    EvaluationConfig, build_models, choose_threshold, evaluate_scores,
    make_demo_data, prepare_data, run_analysis, split_data, tableau_export,
    load_csv,
)
from ntc_risk.reporting import profile_data
from scripts.verify_reproduction import verify


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.frame = make_demo_data(400)

    def test_outcome_proxies_and_age_never_enter_features(self):
        self.frame['unreviewed_target_copy'] = self.frame['Ever_ChargeOff']
        X, _ = prepare_data(self.frame)
        self.assertFalse({'Ever_ChargeOff', 'ChargeOff_Balance', 'Unpaid_CO_84M', 'Age',
                          'unreviewed_target_copy'} & set(X.columns))

    def test_disjoint_stratified_splits_retain_all_rows(self):
        X, y = prepare_data(self.frame)
        a, b, c, ya, yb, yc = split_data(X, y)
        self.assertEqual((len(a), len(b), len(c)), (280, 60, 60))
        self.assertFalse(set(a.index) & set(b.index) | set(a.index) & set(c.index) | set(b.index) & set(c.index))
        self.assertEqual(set(a.index) | set(b.index) | set(c.index), set(X.index))
        for target in [ya, yb, yc]:
            self.assertEqual(set(target.unique()), {0, 1})
            self.assertLess(abs(target.mean() - y.mean()), .03)

    def test_threshold_changes_error_counts_consistently(self):
        y = [0, 0, 1, 1]
        scores = [.2, .6, .65, .8]
        high = evaluate_scores(y, scores, .7)
        low = evaluate_scores(y, scores, .5)
        self.assertEqual([high[k] for k in ['tn','fp','fn','tp']], [2,0,1,1])
        self.assertEqual([low[k] for k in ['tn','fp','fn','tp']], [1,1,0,2])
        self.assertEqual(high['roc_auc'], low['roc_auc'])

    def test_no_feasible_threshold_is_an_explicit_error(self):
        config = EvaluationConfig(minimum_recall=.8, minimum_flags=3, thresholds=(.9,))
        with self.assertRaisesRegex(ValueError, 'No validation threshold'):
            choose_threshold([0,0,1,1], [.1,.2,.3,.4], config)

    def test_imputer_statistics_only_use_training_input(self):
        X, y = prepare_data(self.frame)
        X_train, X_val, _, y_train, _, _ = split_data(X, y)
        model = build_models(y_train, include_xgboost=False)['random_forest']
        X_train = X_train.copy()
        X_train['FICO_V'] = 500.
        X_train.iloc[0, X_train.columns.get_loc('FICO_V')] = np.nan
        X_val = X_val.copy()
        X_val['FICO_V'] = 850.
        model.fit(X_train, y_train)
        model.predict_proba(X_val)
        index = X_train.columns.get_loc('FICO_V')
        self.assertEqual(model.named_steps['imputer'].statistics_[index], 500.)

    def test_reporting_boundaries_and_zero_income(self):
        self.frame.loc[0, 'FICO_V'] = 300.
        self.frame.loc[1, 'FICO_V'] = 850.
        self.frame.loc[2, 'FICO_V'] = np.nan
        self.frame.loc[0, 'Income'] = 0.
        output = tableau_export(self.frame)
        self.assertEqual(output.loc[0, 'FICO'], 'Poor')
        self.assertEqual(output.loc[1, 'FICO'], 'Exceptional')
        self.assertEqual(output.loc[1, 'FICO_Band'], '850-850')
        self.assertEqual(output.loc[2, 'FICO'], 'No score')
        self.assertTrue(np.isnan(output.loc[0, 'DTI']))
        self.assertNotIn('Predicted_Chargeoff_Score', output.columns)
        self.assertTrue(output['Observed_Chargeoff'].equals(output['Ever_ChargeOff']))

    def test_invalid_inputs_fail_instead_of_silent_recoding(self):
        for field, value in [('Ever_ChargeOff',2),('FICO_V',999),('No_Hit',3),('Income',-1)]:
            frame = self.frame.copy()
            frame.loc[0, field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                prepare_data(frame)
        with self.assertRaisesRegex(ValueError,'Duplicate feature'):
            prepare_data(__import__('pandas').concat([self.frame,self.frame.iloc[[0]]], ignore_index=True))

    def test_model_selection_does_not_read_test_labels(self):
        X, y = prepare_data(self.frame)
        parts = split_data(X, y)
        changed = (*parts[:5], 1-parts[5])
        with tempfile.TemporaryDirectory() as directory:
            with patch('ntc_risk.pipeline.split_data',return_value=parts):
                first,_ = run_analysis(self.frame,Path(directory)/'a',dataset_kind='demonstration',include_xgboost=False)
            with patch('ntc_risk.pipeline.split_data',return_value=changed):
                second,_ = run_analysis(self.frame,Path(directory)/'b',dataset_kind='demonstration',include_xgboost=False)
        self.assertEqual(first['selected_model'],second['selected_model'])
        self.assertEqual(first['selected_threshold'],second['selected_threshold'])
        self.assertNotEqual(first['selected_model_test']['roc_auc'],second['selected_model_test']['roc_auc'])

    def test_all_models_run_and_test_uses_frozen_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            report,table=run_analysis(self.frame,directory,dataset_kind='demonstration')
            self.assertEqual(set(table['model']),{'dummy_prior','logistic_regression','random_forest','xgboost'})
            self.assertEqual(report['selected_threshold'],report['selected_model_test']['threshold'])
            self.assertTrue((Path(directory)/'evaluation.json').is_file())
            self.assertEqual(report['splits'],{'train':280,'validation':60,'test':60})

    def test_historical_metrics_are_recomputed_from_counts(self):
        records=json.loads((Path(__file__).parents[1]/'docs/historical_results.json').read_text())['records']
        for row in records:
            self.assertAlmostEqual(row['precision'],row['tp']/(row['tp']+row['fp']))
            self.assertAlmostEqual(row['recall'],row['tp']/(row['tp']+row['fn']))
            self.assertEqual(row['n'],sum(row[k] for k in ['tn','fp','fn','tp']))

    def test_original_source_integrity_and_coverage(self):
        root = Path(__file__).parents[1]
        source = root/'data/source/v_credit_data_NTC_v51825.csv'
        manifest = json.loads((root/'docs/source_manifest.json').read_text())['original_dataset']
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), manifest['sha256'])
        frame = load_csv(source)
        profile, schema, groups = profile_data(frame)
        self.assertEqual(frame.shape, (2000, 27))
        self.assertEqual(profile['positive_outcomes'], 283)
        self.assertEqual(profile['missing_fico_count'], 830)
        self.assertEqual(int(groups['n'].sum()), len(frame))
        self.assertEqual(int(groups['positive'].sum()), 283)
        self.assertEqual(set(schema['column']), set(frame.columns))

    def test_reproduction_guard_rejects_changed_selection_and_counts(self):
        root = Path(__file__).parents[1]
        expected = json.loads((root/'reports/evaluation.json').read_text())
        verify(expected, copy.deepcopy(expected))
        for field in ['selected_threshold', 'selected_model_test']:
            changed = copy.deepcopy(expected)
            if field == 'selected_threshold':
                changed[field] = .99
            else:
                changed[field]['tp'] += 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                verify(expected, changed)


if __name__=='__main__':
    unittest.main()
