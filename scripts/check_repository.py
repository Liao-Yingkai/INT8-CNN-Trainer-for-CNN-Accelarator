"""Check source syntax, relative documentation links and portable path literals."""
import ast
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SKIP = {'.git', 'data', 'generated', 'checkpoints', '__pycache__', '.venv', 'venv',
        'env', '.vscode', '.idea', 'build', 'dist', '.pytest_cache', '.mypy_cache',
        '.ruff_cache', 'htmlcov'}


def source_files(directory):
    """Walk source only, never traversing datasets, trained models or build output."""
    for path in sorted(directory.iterdir()):
        if path.name in SKIP or path.is_symlink():
            continue
        if path.is_dir():
            yield from source_files(path)
        elif path.suffix in {'.py', '.md', '.yaml', '.txt'}:
            yield path


def main():
    """Fail without modifying files when syntax, links or fixed machine paths leak."""
    errors = []
    address = re.compile(r'/(?:home|Users|opt|mnt|tmp|path/to)/|[A-Za-z]:\\')
    for path in source_files(ROOT):
        text = path.read_text(encoding='utf-8')
        relative = path.relative_to(ROOT)
        if address.search(text):
            errors.append(f'{relative}: hard-coded machine path')
        if path.suffix == '.py':
            try:
                ast.parse(text, filename=str(relative))
            except SyntaxError as error:
                errors.append(f'{relative}: {error}')
        if path.suffix == '.md':
            for link in re.findall(r'\]\(([^)]+)\)', text):
                target = link.split('#', 1)[0]
                if not target or re.match(r'[a-zA-Z][a-zA-Z0-9+.-]*:', target):
                    continue
                if target.startswith('/') or not (path.parent/target).exists():
                    errors.append(f'{relative}: invalid relative link {target}')
    if errors:
        raise SystemExit('\n'.join(errors))
    print('Repository checks passed: source syntax, relative links and path literals.')


if __name__ == '__main__':
    main()
