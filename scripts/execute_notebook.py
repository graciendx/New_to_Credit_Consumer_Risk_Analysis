"""Execute the research notebook and retain its outputs; fresh kernel is the default."""
import argparse
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--in-process', action='store_true',
                        help='Use sequential IPython capture when the host forbids kernel sockets.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = root/'notebooks/NTC_Consumer_Credit.ipynb'
    notebook = nbformat.read(path, as_version=4)
    if args.in_process:
        from IPython.core.interactiveshell import InteractiveShell
        from IPython.utils.capture import capture_output
        InteractiveShell.instance()
        namespace = {'__name__': '__main__'}
        count = 0
        for cell in notebook.cells:
            if cell.cell_type != 'code':
                continue
            count += 1
            with capture_output() as captured:
                exec(compile(cell.source, f'notebook-cell-{count}', 'exec'), namespace)
            cell.execution_count = count
            cell.outputs = []
            for name, value in [('stdout', captured.stdout), ('stderr', captured.stderr)]:
                if value:
                    cell.outputs.append(nbformat.v4.new_output('stream', name=name, text=value))
            for output in captured.outputs:
                cell.outputs.append(nbformat.v4.new_output('display_data', data=output.data,
                                                           metadata=output.metadata))
        mode = 'sequential IPython capture (no kernel sockets)'
    else:
        manager = KernelManager(kernel_name='python3')
        # Use this process's Python environment, even when a global kernel is installed.
        manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
        client = NotebookClient(notebook, km=manager, timeout=180,
                                resources={'metadata': {'path': str(root)}})
        client.execute()
        mode = 'fresh Jupyter kernel'
    notebook.metadata['verification'] = {'execution_mode': mode, 'dataset': 'pinned original project CSV'}
    nbformat.validate(notebook)
    output = args.output or path
    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output)
    count = sum(cell.cell_type == 'code' for cell in notebook.cells)
    print(f'PASS: {count} notebook code cells executed with {mode}; saved {output.name}.')


if __name__ == '__main__':
    main()
