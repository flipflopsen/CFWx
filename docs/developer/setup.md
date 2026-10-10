# Developer environment and setup

This guide describes the current checkout, not the original C project template. Run commands from the repository root unless specified. Start with [the documentation index](../README.md), [architecture](../ARCHITECTURE.md), and [training](../TRAINING.md).

## Supported environment and prerequisites
The application targets Linux/POSIX; Windows development is through WSL2 or a Linux environment. Original `getopt` and `ptrace` startup code is not a native Windows port. Numerical kernels use C17 and double precision. Dependencies:

| Dependency | Required for |
|---|---|
| Git | Checkout and contribution |
| GCC or Clang with C17 support | CPU build |
| CMake and a build tool (Make/Ninja) | Target configuration/build |
| Python 3 | Maintenance and documentation coverage scripts |
| cmocka development package | Additional unit-test target; core regressions work without it |
| Doxygen, optionally Graphviz | API HTML/XML documentation |
| clang-format 22.1.0 | Reproducible repository formatting |
| Valgrind | Optional memory-check target |
| vcpkg | Original default package integration, optional in system mode |
| CUDA Toolkit and supported host compiler | Explicit optional CUDA build only |

CPU CMake retains the original low minimum; **use CMake >=3.24** for a unified developer setup and the optional CUDA build. User-reported GPU driver versions do not establish Toolkit installation. No GPU environmental validation is performed by this guide or documentation generation.

## 1. Checkout

```sh
git clone --branch feature/implement_basics https://github.com/flipflopsen/CFWx.git
cd CFWx
git status --short
```

Use a personal feature branch for changes. Do not reset/rewrite collaborator commits or use the original attached template to overwrite the current CMakeLists.txt.

## 2. Install Linux tools
Example Debian/Ubuntu setup (adapt package names to your distribution):

```sh
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build git python3 python3-venv \
  pkg-config libcmocka-dev doxygen graphviz valgrind
python3 -m venv .venv
. .venv/bin/activate
python -m pip install clang-format==22.1.0
```

Use an isolated virtual environment rather than modifying distribution-managed Python. Add `.venv/` to local/global excludes if needed. Pin compiler/CMake/Doxygen versions for reproducible published artifacts; dependency availability alone is not a guarantee of identical floating-point output.

## 3. Build with system packages (recommended first run)

```sh
cmake -S . -B build-debug -G Ninja \
  -DCFWX_USE_VCPKG=OFF -DCFWX_CUDA=OFF -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug --parallel
ctest --test-dir build-debug --output-on-failure
./build-debug/Debug/cfwx
```

Main initializes the legacy backend and runs its numerical example. Logs use the configured project root. Tests cover numerical operations, derivatives, Transformer composition, logging, training, persistence and cache options. cmocka tests are registered when discovery succeeds; inspect configuration output rather than assuming they ran.

## 4. Original vcpkg integration
The default root build references `/opt/vcpkg/scripts/buildsystems/vcpkg.cmake` and a cmocka configuration under `/opt/vcpkg/installed/x64-linux/share/cmocka`. If that setup is absent, use system mode above or provide your own toolchain on the **first configure of a fresh tree**:

```sh
cmake -S . -B build-vcpkg \
  -DCMAKE_TOOLCHAIN_FILE=/your/vcpkg/scripts/buildsystems/vcpkg.cmake \
  -Dcmocka_DIR=/your/vcpkg/installed/x64-linux/share/cmocka \
  -DCFWX_CUDA=OFF -DCMAKE_BUILD_TYPE=Debug
cmake --build build-vcpkg --parallel
ctest --test-dir build-vcpkg --output-on-failure
```

Install the cmocka package with the matching vcpkg triplet beforehand. Do not mix incompatible architecture/triplet/compiler binaries. This repository does not claim a tested automatic vcpkg bootstrap. Toolchain paths and triplet selection remain developer choices.

## 5. Release and CPU memory checks

```sh
cmake -S . -B build-release -DCFWX_USE_VCPKG=OFF -DCFWX_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --parallel
ctest --test-dir build-release --output-on-failure

cmake -S . -B build-asan -DCFWX_USE_VCPKG=OFF -DCFWX_CUDA=OFF -DCFWX_SANITIZERS=ON
cmake --build build-asan --parallel
ASAN_OPTIONS=detect_leaks=1 ctest --test-dir build-asan --output-on-failure
```

Test assertions remain active under NDEBUG. Optional `valgrind` checks the application, not automatically every test. Run specific test binaries under Valgrind if needed. Some sanitizers/debugger detection have process/platform interactions; preserve diagnostics and do not label a skipped suite as passed.

## 6. Dataset training and inference

```sh
mkdir -p data
printf 'the cat sits. the dog runs. the cat runs. the dog sits.\n%.0s' $(seq 1 100) > data/tiny.txt
./build-release/Release/cfwx_run train data/tiny.txt --checkpoint-name sample 300 cpu
./build-release/Release/cfwx_run resume data/tiny.txt --checkpoint-name sample 100 cpu
./build-release/Release/cfwx_run generate sample.ckpt 'the cat ' 40 sliding 32
```

Checkpoints resolve relative to the process working directory. Resume rejects a changed dataset fingerprint. Cache mode is fixed or sliding; positional indices remain absolute during eviction. POSIX publication fsyncs the file and parent directory; a post-rename sync failure may leave new data visible without a durability guarantee. See [runtime](../RUNTIME.md) and [durability](../gpu-runtime/durability.md).

The original `cfwx_train` remains a small deterministic demonstration. Repetitive data can yield trivial validation; it is not a benchmark. Check tokenizer/vocabulary, valid-token reductions, document splits and model architecture before realistic experiments.

## 7. Documentation
All prose Markdown is under `docs/`. Doxygen-specific configuration and `.dox` source pages are under `doxygen_docs/`. Generated output is separate and ignored:

```sh
doxygen doxygen_docs/Doxyfile
python scripts/check_doxygen.py
# Open doxygen_docs/generated/html/index.html
cmake --build build-debug --target docs
```

Graphviz supplies optional diagrams. Set HAVE_DOT=NO in a local config if dot is unavailable. HTML extraction is not a complete semantic audit: inspect warnings under `doxygen_docs/generated/doxygen-warnings.log`. Existing abbreviated implementation parameter names can yield merge warnings. Do not erase diagnostics by claiming warning-free coverage.

## 8. CUDA: manual opt-in only
Read [CUDA.md](../CUDA.md) and [GPU roadmap](../gpu-runtime/overview.md) first. A CUDA Toolkit supporting SM89 and a supported host compiler are separate prerequisites. CUDA is OFF in ordinary builds. Source-level integration does not prove numerical correctness or speedup.

```sh
cmake -S . -B build-cuda -DCFWX_USE_VCPKG=OFF -DCFWX_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release -DCFWX_CUDA_ARCHITECTURES=89
cmake --build build-cuda --parallel 12
# Run these manually on your own GPU only:
./build-cuda/Release/cfwx_cuda_manual --test
CFWX_CUDA_FUSED_BATCH=1 ./build-cuda/Release/cfwx_cuda_manual data/tiny.txt 200
```

Do not combine CPU ASan and CUDA builds. Manual Compute Sanitizer tests are documented separately. Launch-fused sequence interpretation is not cuBLAS/Tensor Core/FlashAttention optimization. CPU/CUDA checkpoint interchange and GPU KV caches remain design proposals, not automatic features.

## 9. IDE and custom targets
VSCode configurations are historically user-specific; replace hardcoded paths and executable names with the active build tree. Use `compile_commands.json` from that tree for clangd. Configure/debug the actual `build-debug/Debug/...` binary rather than guessing `src/bin` paths from the original template.

`welp` prints target help; `start` copies the template to an external PROJECT_PATH, rejecting self/descendant copies; `clear` deletes generated bin/log output but not the active build tree; `docs-clean` removes generated Doxygen output only. `PROJECT_PATH` also influences legacy logger output—do not set it to an unintended location.

## Troubleshooting
- Missing `/opt/vcpkg`: configure a fresh tree with CFWX_USE_VCPKG=OFF or supply your actual toolchain.
- cmocka not found: install development package or fix cmocka_DIR/pkg-config; review which tests were registered.
- Doxygen not found: install it, then reconfigure CMake to expose docs target.
- Markdown link failure after moves: run `python scripts/check_docs_layout.py`.
- Wrong binary path: inspect generator/config-specific output and CMake build logs.
- Failed backward/update: do not reuse partial gradients or save a poisoned CUDA model.
- CUDA compile errors: capture Toolkit/compiler/version and source diagnostics locally; no assistant-side GPU validation is implied.
