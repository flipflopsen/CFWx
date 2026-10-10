"""Idempotent source-only Doxygen parser/annotation refinement. No CUDA execution."""
from pathlib import Path
p=Path('scripts/document_repository.py');s=p.read_text()
s=s.replace("else:brief='Perform the '","else:brief='Perform the '")
# Ret tokens must end at a lexical boundary: integer() must not be parsed as int eger().
s=s.replace("std::vector<int>))\\s*(?P<name>", "std::vector<int>))(?=\\s|\\*)(?:\\s+|(?<=\\*)\\s*)(?P<name>")
# Match only declaration/definition-like argument types, not std::vector constructors.
s=s.replace("            name=m['name']\n", "            name=m['name']\n            if any(re.fullmatch(r'[A-Za-z_]\\w*', a.strip()) and a.strip() != 'void' for a in m['args'].split(',')):\n                continue\n")
s=s.replace("[' * @copydetails '+name,' * @see '+str(public[name])]", "[' * @details Implementation of the public declaration; its canonical contract', ' * remains in the corresponding header and is merged by Doxygen.', ' * @see '+name+' '+str(public[name])]")
s=s.replace("elif re.search(r'\\bvoid\\b',ret):block+=[' * @return No value; inspect the documented mutation/lifetime effects.']", "elif re.search(r'\\bvoid\\b',ret):block+=[' * @note This operation returns no value; inspect its documented mutation/lifetime effects.']")
# Stable markers and generated typedef comments must be idempotent.
s=s.replace("if not (output and '@brief Record' in output[-1]):", "if not (output and '@brief Record' in output[-1]):")
# Do not promote ordinary in-body comments to API blocks; retain detailed source commentary.
s=s.replace("text = re.sub(r'/\\*(?![*!])', '/**', text)", "text = re.sub(r'(?m)^/\\*(?![*!])', '/**', text)")
# Type/member pass previously considered long comment paragraphs to be fields.
s=s.replace("if in_record and stripped and not stripped.startswith", "if in_record and re.match(r'^(?:const\\s+)?(?:size_t|uint64_t|double|bool|int|unsigned|char|cfwx_\\w+|LogLevel|logger_sink_fn|node|cudaStream_t)\\b', stripped) and not stripped.startswith")
p.write_text(s)
p=Path('Doxyfile');s=p.read_text()
s=s.replace('EXAMPLE_PATH           = test src/experiments','EXAMPLE_PATH           = . test src/experiments docs docs/gpu-runtime')
s=s.replace('EXAMPLE_PATTERNS       = *.c','EXAMPLE_PATTERNS       = *.c *.md')
# Config metadata is explained in the tooling page, not parsed as fake C++.
s=s.replace(' CMakeLists.txt .github/workflows .vscode .gitattributes .gitignore','')
s=s.replace(' *.cmake *.txt *.yml *.yaml *.json .gitattributes .gitignore','')
s=s.replace(' scripts cmake docs',' scripts docs')
p.write_text(s)
p=Path('docs/doxygen/guides.dox');s=p.read_text().replace('Use @file, @brief, @details, @ingroup, @param, @return, @pre, @post, @warning,','Use file, brief, details, ingroup, param, return, pre, post, warning commands,').replace('@note and @see where appropriate.','note and see commands where appropriate.')
p.write_text(s)
