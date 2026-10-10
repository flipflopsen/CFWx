# Documentation integration validation

The integrated reference generated HTML/XML successfully and its named-API coverage check passed: **103 named APIs, 20 groups and 28 guide pages** at the validation snapshot. The configuration-reference page is an additional guide added subsequently. Counts are not guarantees of semantic completeness.

The documented CPU sources were built with CUDA disabled and all nine available dependency-free CPU suites passed. cmocka was not installed in that local validation environment. Documentation workflow generation and coverage also passed. No CUDA configure/compiler/GPU execution or hardware probes occurred.

## Remaining warning diagnostics
The local Doxygen run retained approximately 312 warning-log lines (not necessarily 312 independent warnings). They remain visible rather than suppressed. Main categories include declaration/definition parameter-name differences in legacy APIs, parser/extraction edge cases and incompletely documented implementation members. The generated public header contracts are canonical; source definitions often use abbreviated names such as `m`, `g`, `eps` instead of the header's `model`, `gradient`, `epsilon`, which can trigger merged parameter warnings.

This is a substantial HTML reference integration, **not a warning-free exhaustive semantic audit**. Read `build-docs/doxygen-warnings.log` and review the generated pages. Internal helpers may have general type-aware descriptions and should receive more specific algorithm invariants as those interfaces evolve. Coverage tooling checks named public briefs and group/page presence; it does not assert all member/parameter descriptions are scientifically adequate.

## Updating annotations
The prepared annotation script is checked into source alongside the generated comments. Normal documentation generation needs only:

```sh
doxygen Doxyfile
python scripts/check_doxygen.py
```

To regenerate tool-authored annotations intentionally:

```sh
python scripts/prepare_doxygen.py
doxygen Doxyfile
python scripts/check_doxygen.py
```

Do not rerun historical refinement scripts independently after the annotator has already been prepared. Review the diff: no executable semantics should change. Use `prepare_doxygen.py` rather than the earlier low-level annotation command when adding APIs. Documentation output stays under ignored `build-docs/`.
