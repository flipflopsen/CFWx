# Documentation layout and migration policy

## Canonical locations
- `docs/README.md`: repository documentation index. No Markdown files remain in the project root.
- `docs/developer/`: complete setup, contribution and documentation-layout guides.
- `docs/ARCHITECTURE.md`, `docs/TRAINING.md`, `docs/RUNTIME.md`, `docs/CUDA.md`, `docs/TRANSFORMER_BLOCKS.md`, `docs/CODE_STYLEGUIDE.md`: relocated repository guides.
- `docs/gpu-runtime/`: optimization and interoperability design proposals.
- `docs/doxygen/`: Markdown usage/validation documentation about Doxygen.
- `doxygen_docs/Doxyfile`: source-only Doxygen configuration.
- `doxygen_docs/*.dox`: Doxygen main page, groups and guide definitions.
- `doxygen_docs/generated/`: ignored generated HTML/XML and warnings.

## Completed root scan
The root contained seven Markdown files: README.md, ARCHITECTURE.md, TRAINING.md, RUNTIME.md, CUDA.md, TRANSFORMER_BLOCKS.md and CODE_STYLEGUIDE.md. All seven were moved to docs/. Existing Markdown under docs/ was retained. Five `.dox` files moved from docs/doxygen/ to doxygen_docs/, along with the former root Doxyfile.

The repository currently contains 21 Markdown documentation files, all under docs/. Git records relocations; source prose was retained and path references repaired. The layout checker checks local Markdown file links, not remote URLs or rendered anchors.

## Local generation

```sh
python scripts/check_docs_layout.py
doxygen doxygen_docs/Doxyfile
python scripts/check_doxygen.py
```

Run from repository root. Open `doxygen_docs/generated/html/index.html`. CMake's `docs` target uses the relocated config and root working directory; `docs-clean` removes generated output only. CUDA is parsed as text, never configured/probed/executed by documentation generation.

## Tooling policy
Maintenance scripts open `docs/README.md` and sibling guides rather than recreating deleted root Markdown. Documentation CI is read-only and uploads HTML; it does not regenerate comments or race collaborator pushes. Historical source-migration scripts remain available for intentional reviewed maintenance, not automatic formatting. Use `prepare_doxygen.py` only when regenerating tool-authored comments deliberately and review the diff.

Doxygen warnings remain visible; API/group/page coverage is not a claim of exhaustive semantic documentation. Config/YAML/JSON are represented as source examples and tooling explanations rather than fake callable C APIs.
