"""Download the pinned public project input for local analysis; never upload records."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = ('https://raw.githubusercontent.com/chloephammi/New-To-Credit-Consumer/'
              '8b55c183e045e80b87ffb9e5efd0a1268443b7e4/v_credit_data_NTC_v51825.csv')


def fetch_source(destination=None):
    manifest = json.loads((ROOT/'docs/source_manifest.json').read_text())['original_dataset']
    destination = Path(destination) if destination is not None else ROOT/manifest['path']
    expected = manifest['sha256']
    if destination.is_file():
        if hashlib.sha256(destination.read_bytes()).hexdigest() != expected:
            raise ValueError('Existing input does not match the pinned source; it was not overwritten.')
        return destination
    with urlopen(SOURCE_URL, timeout=30) as response:
        content = response.read(1_000_001)
    if len(content) > 1_000_000 or hashlib.sha256(content).hexdigest() != expected:
        raise ValueError('Downloaded input failed the pinned size/checksum check; nothing was installed.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        path = fetch_source(args.output)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f'Original project input verified locally: {path.name}')


if __name__ == '__main__':
    main()
