# Documentation layout and migration policy

## Canonical locations
- `docs/README.md`: repository documentation landing page; no Markdown landing file remains in the root.
- `docs/developer/`: environment, contribution and documentation-maintenance instructions.
- `docs/ARCHITECTURE.md`, `docs/TRAINING.md`, `docs/RUNTIME.md`, `docs/CUDA.md`, `docs/TRANSFORMER_BLOCKS.md`, `docs/CODE_STYLEGUIDE.md`: existing guides moved intact with repaired links.
- `docs/gpu-runtime/`: optimization/interoperability proposals and precise implementation boundaries.
- `docs/doxygen/`: Markdown instructions/validation notes about Doxygen; these remain Markdown prose under docs/.
- `doxygen_docs/Doxyfile`: Doxygen configuration.
- `doxygen_docs/*.dox`: Doxygen-specific main page, groups and guide sources.
- `doxygen_docs/generated/`: ignored HTML/XML/diagnostics, regenerated locally.

The root scan found README.md, ARCHITECTURE.md, TRAINING.md, RUNTIME.md, CUDA.md, TRANSFORMER_BLOCKS.md and CODE_STYLEGUIDE.md. All are relocated into docs/. Existing Markdown already under docs/ remains there. Existing `.dox` files move out of docs/doxygen/ to doxygen_docs/.

## Local commands

```sh
doxygen doxygen_docs/Doxyfile
python scripts/check_doxygen.py
python scripts/check_docs_layout.py
```

Run Doxygen from the repository root because INPUT/include paths are root-relative. The CMake docs target uses that working directory and configuration path. docs-clean removes only doxygen_docs/generated/, not source comments/configuration or Markdown guides.

## Links and tooling
Markdown links are relative to the current document, not the repository root. Developer guides use `../TRAINING.md`; Doxygen instruction Markdown links to `../../doxygen_docs/Doxyfile`. Doxygen include directives use repository-root-relative paths such as `docs/ARCHITECTURE.md`. Source-maintenance scripts must open docs/README.md and not regenerate root Markdown. Workflows stage docs/ and doxygen_docs/ rather than deleted root documentation.

The layout checker rejects tracked Markdown outside docs/ and broken local Markdown links. It skips external URLs and anchor-only links; it does not validate remote servers or rendered Doxygen anchors. Generated documentation is intentionally excluded from repository layout checks. Historical warning counts are snapshots, not promises that current generation is warning-free.

No executable GPU behavior is changed by this reorganization. Documentation parsing never probes or runs the GPU.
