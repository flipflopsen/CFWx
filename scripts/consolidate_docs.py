"""Consolidate Markdown under docs/ and Doxygen assets under doxygen_docs/.
Source/document-only operation. Never configures or executes CUDA.
"""
from pathlib import Path
import re
import shutil

root_markdown = sorted(Path('.').glob('*.md'))
for path in root_markdown:
    target = Path('docs') / path.name
    if target.exists():
        raise SystemExit(f'Conflicting documentation destination: {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    path.rename(target)

Path('doxygen_docs').mkdir(exist_ok=True)
for path in Path('docs/doxygen').glob('*.dox'):
    target=Path('doxygen_docs')/path.name
    if target.exists():raise SystemExit(f'Conflicting Doxygen page: {target}')
    path.rename(target)
if Path('Doxyfile').exists():
    Path('Doxyfile').rename('doxygen_docs/Doxyfile')

names=['README.md','ARCHITECTURE.md','TRAINING.md','RUNTIME.md','CUDA.md','TRANSFORMER_BLOCKS.md','CODE_STYLEGUIDE.md']
# All links in moved root guides remain siblings; root Doxyfile command/output
# paths and Doxygen include directives need explicit new locations.
paths=[]
for directory in ('docs','doxygen_docs','scripts','cmake','.github/workflows'):
    paths.extend(p for p in Path(directory).rglob('*') if p.is_file() and 'generated' not in p.parts and p.suffix in ('.md','.dox','.py','.cmake','.yml','.yaml'))
paths.append(Path('CMakeLists.txt'))
for path in paths:
    if path.name=='consolidate_docs.py':continue
    text=path.read_text()
    text=text.replace('build-docs/','doxygen_docs/generated/').replace('/build-docs','/doxygen_docs/generated')
    text=text.replace('doxygen Doxyfile','doxygen doxygen_docs/Doxyfile')
    text=text.replace('${CMAKE_SOURCE_DIR}/Doxyfile','${CMAKE_SOURCE_DIR}/doxygen_docs/Doxyfile')
    text=text.replace("Path('docs/doxygen/guides.dox')","Path('doxygen_docs/guides.dox')")
    text=text.replace('docs/doxygen/**','doxygen_docs/**, docs/**')
    # File IO in maintenance tools must not regenerate root Markdown.
    if path.suffix=='.py':
        for name in names:
            text=text.replace("Path('"+name+"')","Path('docs/"+name+"')")
            text=text.replace('Path("'+name+'")','Path("docs/'+name+'")')
        text=text.replace("('README.md','TRAINING.md','ARCHITECTURE.md')","('docs/README.md','docs/TRAINING.md','docs/ARCHITECTURE.md')")
        text=text.replace("Path('Doxyfile')","Path('doxygen_docs/Doxyfile')")
        text=text.replace("Path('build-docs/xml')","Path('doxygen_docs/generated/xml')")
    if path.suffix=='.dox':
        for name in names:
            text=re.sub(r'(@include(?:\{doc\})?\s+)'+re.escape(name)+r'\b',r'\1docs/'+name,text)
    if path.suffix in ('.yml','.yaml'):
        # Stage canonical doc trees and removals, not deleted legacy root paths.
        for name in names:text=re.sub(r'(?<![/\w])'+re.escape(name)+r'(?![\w])','docs/'+name,text)
        text=text.replace(' docs/doxygen ', ' docs/ doxygen_docs/ ')
        text=text.replace(' Doxyfile ', ' doxygen_docs/Doxyfile ')
    if path==Path('docs/README.md'):
        text=text.replace('[TRAINING.md](TRAINING.md)','[TRAINING.md](TRAINING.md)')
        text=text.replace('docs/doxygen/README.md','doxygen/README.md')
    path.write_text(text)

p=Path('doxygen_docs/Doxyfile');text=p.read_text()
text=re.sub(r'^OUTPUT_DIRECTORY\s*=.*$', 'OUTPUT_DIRECTORY = doxygen_docs/generated',text,flags=re.M)
text=re.sub(r'^INPUT\s*=.*$', 'INPUT = include src test scripts docs doxygen_docs',text,flags=re.M)
text=re.sub(r'^WARN_LOGFILE\s*=.*$', 'WARN_LOGFILE = doxygen_docs/generated/doxygen-warnings.log',text,flags=re.M)
text=re.sub(r'^EXCLUDE_PATTERNS\s*=.*$', 'EXCLUDE_PATTERNS = */generated/* */build/* */build-*/* */bin/* */log/* */.git/* */.cache/*',text,flags=re.M)
text=text.replace('EXAMPLE_PATH           = . test src/experiments docs docs/gpu-runtime','EXAMPLE_PATH = . test src/experiments docs docs/gpu-runtime doxygen_docs')
p.write_text(text)

# Convert old configuration filename text in page/workflow trigger only.
for path in Path('.github/workflows').glob('*.yml'):
    text=path.read_text().replace('paths: [Doxyfile,','paths: [doxygen_docs/Doxyfile,')
    path.write_text(text)
p=Path('.gitignore');text=p.read_text()
if 'doxygen_docs/generated/' not in text:text+='\ndoxygen_docs/generated/\n'
if '.venv/' not in text:text+='\.venv/\n'.replace('\\','')
p.write_text(text)

p=Path('docs/README.md');text=p.read_text()
if '## Developer documentation index' not in text:
    text+='''
## Developer documentation index
- [Environment and complete setup](developer/setup.md)
- [Developer contribution workflow](developer/contributing.md)
- [Documentation layout](developer/documentation-layout.md)
- [Architecture](ARCHITECTURE.md)
- [Training and datasets](TRAINING.md)
- [Runtime, checkpoints and generation](RUNTIME.md)
- [Transformer blocks](TRANSFORMER_BLOCKS.md)
- [Code style](CODE_STYLEGUIDE.md)
- [CUDA manual build/validation](CUDA.md)
- [GPU implementation roadmap](gpu-runtime/overview.md)
- [Doxygen manual instructions](doxygen/README.md)

All Markdown lives under docs/. Doxygen configuration/pages live under
[doxygen_docs](../doxygen_docs/Doxyfile); generated HTML/XML are ignored under
`doxygen_docs/generated/`. Run `doxygen doxygen_docs/Doxyfile` from repository root.
'''
p.write_text(text)
print('Consolidated Markdown and relocated Doxygen assets; updated canonical references.')
