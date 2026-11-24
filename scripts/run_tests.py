"""Run synthetic regression suites without datasets or trained checkpoints."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    """Run models in separate interpreters to isolate same-name modules."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('all', 'mnist', 'yolov2_tiny'), default='all')
    args = parser.parse_args()
    checked = subprocess.run([sys.executable, 'scripts/check_repository.py'], cwd=ROOT)
    if checked.returncode:
        raise SystemExit(checked.returncode)
    models = ('mnist', 'yolov2_tiny') if args.model == 'all' else (args.model,)
    for model in models:
        print(f'Running {model} regression tests', flush=True)
        result = subprocess.run([sys.executable, '-m', 'unittest', 'discover',
                                 '-s', f'tests/{model}', '-p', 'test_*.py', '-v'], cwd=ROOT)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
