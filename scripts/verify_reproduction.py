"""Compare a fresh original-data run with the published numerical result."""
import argparse
import json
import math
from pathlib import Path


def verify(expected, actual):
    for key in ['source_file_sha256', 'data_fingerprint_sha256', 'selected_model', 'selected_threshold',
                'splits', 'split_class_counts', 'features', 'config']:
        if expected[key] != actual[key]:
            raise ValueError(f'Reproduction differs in {key}.')
    for section in ['selected_model_test', 'dummy_prior_test_at_0_50']:
        for key, value in expected[section].items():
            if not math.isclose(value, actual[section][key], rel_tol=1e-8, abs_tol=1e-10):
                raise ValueError(f'Reproduction differs in {section}.{key}.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--actual', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    expected = json.loads((root/'reports/evaluation.json').read_text())
    actual = json.loads((args.actual/'evaluation.json').read_text())
    verify(expected, actual)
    print('PASS: source fingerprint, frozen selection, splits and held-out metrics reproduce.')


if __name__ == '__main__':
    main()
