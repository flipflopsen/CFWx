## @file scripts/check_doxygen.py
# @brief check doxygen source/document maintenance tool.
# @ingroup maintenance
# @details Textual maintenance only; does not itself validate numerical correctness.

"""Check generated XML for named public API documentation and guide hierarchy.

Runs no compiler, hardware probe or GPU workload. XML generation must precede it.
"""
from pathlib import Path
import re
import xml.etree.ElementTree as ET

root=Path('build-docs/xml')
if not (root/'index.xml').exists():raise SystemExit('Run doxygen Doxyfile first')
functions={}
for file in root.glob('*.xml'):
    if file.name=='index.xml':continue
    tree=ET.parse(file)
    for member in tree.findall('.//memberdef[@kind="function"]'):
        name=member.findtext('name','')
        if not name.startswith(('cfwx_','experiment_logger_','numerical_runner','transformer_')):continue
        brief=''.join(member.find('briefdescription').itertext()).strip() if member.find('briefdescription') is not None else ''
        functions.setdefault(name,[]).append(bool(brief))
missing=[]
# Public declaration names, including CUDA ABI. Definitions remain source-linked.
for path in Path('include').rglob('*.h'):
    text=path.read_text()
    for name in re.findall(r'\b((?:cfwx_|experiment_logger_|numerical_runner|transformer_)\w*)\s*\([^;{}]*\)\s*;',re.sub(r'/\*.*?\*/','',text,flags=re.S)):
        if name not in functions or not any(functions[name]):missing.append(str(path)+': '+name)
index=ET.parse(root/'index.xml')
groups=index.findall('.//compound[@kind="group"]')
pages=index.findall('.//compound[@kind="page"]')
print(f'Documented named APIs: {len(functions)}; groups: {len(groups)}; pages: {len(pages)}')
if len(groups)<15 or len(pages)<8:raise SystemExit('Incomplete guide/group hierarchy')
if missing:raise SystemExit('Missing documented public APIs:\n'+'\n'.join(sorted(set(missing))))
print('PASS named public API brief and module/page coverage')
