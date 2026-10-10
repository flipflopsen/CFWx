# Building and maintaining the complete HTML manual

## Local generation
Install Doxygen (recommended 1.9.8 or newer) and optionally Graphviz. From the repository root:

```sh
doxygen doxygen_docs/Doxyfile
# Open doxygen_docs/generated/html/index.html
```

Doxyfile controls comprehensive C/CUDA, Python, guide and source-file extraction. CUDA is parsed as C++ text; no `nvcc`, device discovery, training or GPU tests occur. CMake integration is optional and additive:

```sh
cmake -S . -B build -DCFWX_USE_VCPKG=OFF -DCFWX_CUDA=OFF
cmake --build build --target docs
cmake --build build --target docs-clean
```

Original vcpkg/cmocka discovery, custom targets and optional CUDA build are preserved. The older attached CMake template is not copied over the current repository. Generated HTML/XML/warning logs are under ignored `doxygen_docs/generated/`.

## Reference contents
- Dedicated main page and reading paths
- Hierarchical module groups for contracts, containers, mathematics, models, analytic training, runtime, checkpoint, GPU, logging, experiments, tests and tooling
- Public function briefs, parameter descriptions, adjacent shape/ownership contracts and declaration/definition cross-links
- Internal static helpers, source browsing, inline bodies and referenced-by relations
- Owning records, members, enums and macros
- Architecture, numerical contracts, training/data, runtime/cache/persistence, validation, tooling and documentation-maintenance guide pages
- All existing Markdown guides and GPU design proposals as browsable pages

Graphviz provides include, collaboration, group and call/caller graphs. Set HAVE_DOT=NO if dot is not installed; the API/manual still generates. Graph size/depth limits prevent very large reverse-mode graphs from overwhelming navigation.

## Maintenance and validation
Public headers are canonical. Document exact dimensions, borrowed/owned lifetimes, aliasing restrictions, input/output roles, reduction and failed-output mutation. Implementations link the declaration contract and retain algorithm explanations. Internal helper comments are a reference, not a promise of public ABI stability.

```sh
python scripts/document_repository.py
doxygen doxygen_docs/Doxyfile
python scripts/check_doxygen.py
```

The annotation tool is conservative, not a full C++ parser. It preserves executable code and existing contracts, annotates named declarations/definitions, and adds file/module attribution. Review ambiguous internal parameters and any new API manually. Coverage checks generated XML for named public API briefs and group/page hierarchy. EXTRACT_ALL exposes otherwise undocumented symbols; it does not prove exhaustive semantic documentation. Read `doxygen_docs/generated/doxygen-warnings.log` and inspect HTML after each update.

Doxygen does not natively model YAML/JSON/CMake as scientific APIs. Such files are retained for source browsing and explained in the tooling guide. CUDA parser limitations may produce source-only launch/kernel references. Do not hide those limitations by claiming compiler-level coverage or GPU validation.

Documentation generation should be version-pinned for published artifacts. Source annotations and design pages belong in Git; generated outputs do not. CPU-only documentation workflows must not invoke CUDA configure/build/test steps.
