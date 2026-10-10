# VSCode build, run and debug setup

## Which configuration is used?
**VSCode CMake Tools uses `CMakePresets.json`**, not Visual Studio's `CMakeSettings.json`. Both are supplied; do not assume changing one changes the other. The requested CMakeSettings file provides Debug/Release compatibility configurations in separate `build/vs-linux-*` directories for compatible Visual Studio Linux environments. It does not configure remote SSH/WSL transport and does not make the POSIX application a native Windows/MSVC build.

For VSCode on Windows, open this repository **inside WSL** (`WSL: Reopen Folder in WSL`) and install extensions/tools in WSL. Linux/WSL dependencies: CMake >=3.24, Ninja, GCC/G++, GDB, Python 3, optional cmocka, Doxygen and Graphviz. Setup details: [setup.md](setup.md).

## Extensions
Install workspace recommendations: CMake Tools, C/C++, and optionally C/C++ Runner. CMake Tools is the IntelliSense configuration provider; it supplies real compile definitions, include directories and flags. The previous incorrect self-referencing provider has been replaced. The GCC fallback path is Linux-specific; Clang users should select a suitable CMake compiler/kit or user preset.

## Configure and build
1. Open the repository root, not docs/ or src/.
2. Run **CMake: Select Configure Preset** -> `cpu-debug`.
3. Run **CMake: Configure**, then **CMake: Build**.
4. Select a launch target with CMake Tools and use **CMake: Debug**, or choose a named profile in Run and Debug.
5. `Ctrl+Shift+B` runs the default **CFWx: Build Debug** task, configuring first.

Presets: `cpu-debug`, `cpu-release`, `cpu-asan`, `cuda-release`. Build roots are `build/<preset>/`; normal executable paths are `build/cpu-debug/Debug/<target>` and `build/cpu-release/Release/<target>`. Current per-config runtime output settings also place the numerical/cmocka runners in these directories. Existing `bin` properties do not override these per-config settings. CMake-selected target debugging resolves the active target path dynamically and is preferable if output rules change.

The presets default to system dependencies with vcpkg OFF and tests ON. Use `CMakeUserPresets.json` (local, do not commit machine-specific paths) to inherit a preset and supply an actual vcpkg toolchain/cmocka_DIR. Use a fresh build tree when changing compiler/toolchain. Ninja must be on PATH. These settings do not bootstrap compilers/packages.

## Run tasks
Use **Terminal: Run Task**:
- Configure/Build Debug, Release, ASan; Test Debug, Release, ASan
- Run Application, Transformer Examples, Built-in Training
- Train Dataset, Resume Dataset, Generate
- Run CPU Test Runner (pick executable; cmocka runner requires discovered cmocka)
- Clean Debug, Documentation, Documentation Checks, Markdown Checks
- Configure/Build CUDA, CUDA Gradient Test, CUDA Fused Train: explicitly marked **MANUAL**

Runner tasks use `process` execution, with each argument separate, so dataset paths/prompts with spaces are not shell-interpolated. All execute with workspace root as working directory: `sample.ckpt` and relative dataset paths resolve there. Prompt inputs request dataset path, steps, checkpoint identifier/path and cache options. Prepare the dataset first; default `data/tiny.txt` is not automatically created. Fixed cache capacity must cover prompt plus generated tokens.

## Debugging
F5 profiles build Debug first (except the CMake-selected target profile, where CMake Tools controls the selected target/build). Available profiles: application, Transformer examples, built-in training, dataset train/resume, cached generation and picked CPU test. Breakpoints work in kernels/runners as normal C source; inspect owning structs without copying/freeing them in expressions.

GDB is selected through `cppdbg`, with integrated console output. Linux ptrace restrictions can affect debugger attachment. The legacy application uses PTRACE_TRACEME to detect a debugger; other runners avoid that startup behavior. Sanitizers are primarily validated via dedicated test tasks rather than combining a default GDB profile with ASan.

C/C++ Runner remains configured for C17/C++17 and GDB, **but its standalone compile-active-file button is not the project build system**. These executables link multiple modules and generated CMake definitions such as PROJECT_ROOT. Use CMake tasks and the supplied C/C++ debugger launch profiles, not a broad source glob/all-main compilation. Do not assume the runner extension can independently compile an experiment into the correct project target.

## CUDA policy
CUDA preset is opt-in, Release, SM89, sanitizers OFF and automatic GPU CTest registration OFF. It is not selected or run by default. CUDA tasks are for your manual execution only; no assistant-side Toolkit probing, CUDA compilation, GPU test or training validation is implied. Host GDB profiles do not provide CUDA kernel debugging; configure NVIDIA Nsight/cuda-gdb separately if desired. See [CUDA.md](../CUDA.md).

## Documentation
**CFWx: Documentation** invokes `doxygen doxygen_docs/Doxyfile`; generated output stays in `doxygen_docs/generated/`. Documentation Checks generates first, checks Markdown layout, then validates XML API coverage. CMake's docs target remains available. Installing Doxygen after configuring may require reconfiguration to expose that target.

## Troubleshooting
- Missing compiler/Ninja/GDB: install Linux/WSL tools and ensure PATH, then configure again.
- VSCode reads CMakeSettings instead of presets: it should not; use CMake Tools and select the configure preset explicitly.
- Missing test executable: inspect BUILD_TESTING and cmocka discovery in configure output.
- Relative dataset/checkpoint missing: tasks launch from workspace root; enter correct path.
- Wrong preset debug executable: named profiles deliberately use cpu-debug; dynamic selected-target profile uses the active preset instead.
- Compiling every source with C/C++ Runner causes multiple mains: use CMake's separate targets.
- Native Windows compilation fails on getopt/ptrace: use WSL; CMakeSettings does not remove POSIX dependencies.
