"""Check tracked Markdown containment and local links without compiling code."""
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote

files=subprocess.check_output(['git','ls-files'],text=True).splitlines()
# Include newly moved/untracked docs during pre-commit migration checks.
markdown=set(Path(f) for f in files if f.endswith('.md') and Path(f).exists())
markdown.update(Path('docs').rglob('*.md'))
errors=[]
for p in sorted(markdown):
    if 'docs' != p.parts[0]:errors.append(f'Markdown outside docs/: {p}')
    text=p.read_text()
    for target in re.findall(r'\]\(([^)]+)\)',text):
        target=target.strip().split(' "')[0].strip('<>')
        if not target or target.startswith(('#','http:','https:','mailto:')):continue
        path=unquote(target.split('#')[0])
        if path and not (p.parent/path).exists():errors.append(f'{p}: broken link {target}')
if not Path('doxygen_docs/Doxyfile').exists():errors.append('Missing relocated Doxygen config')
if list(Path('.').glob('*.md')):errors.append('Root Markdown still present')
if errors:raise SystemExit('\n'.join(errors))
print(f'PASS docs containment and local links ({len(markdown)} Markdown documents)')
