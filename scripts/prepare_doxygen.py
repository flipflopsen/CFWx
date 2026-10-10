"""Prepare source annotations and manual pages. Doxygen only; no GPU execution."""
from pathlib import Path
import re
import runpy
# Refinements run only on the pre-refinement annotator. They must not append
# repeated data dictionaries when a documentation workflow reruns.
p=Path('scripts/document_repository.py')
s=p.read_text()
if "'clip': 'Positive finite global gradient-clipping threshold.'" not in s:
    runpy.run_path('scripts/refine_doxygen.py')
    runpy.run_path('scripts/finalize_doxygen.py')
    runpy.run_path('scripts/polish_doxygen.py')

# Regenerate only tool-authored API blocks if an earlier workflow produced them.
# Keep manually authored file/contracts/type comments and all executable code.
for root in ('include','src','test'):
    for path in Path(root).rglob('*'):
        if path.suffix not in ('.h','.c','.cu'):continue
        text=path.read_text()
        text=re.sub(r'/\*\*\n \* @anchor api_[\s\S]*?\*/\n', '', text)
        path.write_text(text)
runpy.run_path('scripts/document_repository.py')
# Use anonymous @file to disambiguate legacy same-basename include/*.c files.
for path in Path('include/maths').glob('*.c'):
    text=path.read_text().replace('@file '+str(path),'@file')
    path.write_text(text)
# Type-aware, operation-specific details already appear in API header blocks;
# preserve a manual catalog page alongside generic/internal source attribution.
p=Path('Doxyfile');text=p.read_text()
text=text.replace('EXAMPLE_PATH           = test src/experiments','EXAMPLE_PATH           = . test src/experiments docs docs/gpu-runtime')
text=text.replace('EXAMPLE_PATTERNS       = *.c\n','EXAMPLE_PATTERNS       = *.c *.md\n')
p.write_text(text)
