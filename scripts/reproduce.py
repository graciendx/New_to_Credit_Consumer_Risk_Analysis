"""Fetch the pinned source locally, then regenerate the complete research report."""
from pathlib import Path
import subprocess
import sys

from fetch_data import fetch_source


def main():
    root = Path(__file__).resolve().parents[1]
    source = fetch_source()
    subprocess.run([
        sys.executable, '-m', 'ntc_risk', '--data', str(source),
        '--output', str(root/'reports'), '--tableau-export', str(root/'data/tableau_input.csv'),
    ], cwd=root, check=True)


if __name__ == '__main__':
    main()
