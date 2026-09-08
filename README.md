# CFWx
CFWx is basically a Playground for experimenting with C.
Ran it only on Linux for now through WSL + VSCode, thus I will include the .vscode folder for runner configs.


## Configure correctly

### VSCode
You need to replace the following paths in two configuration files in `.vscode` if you want to use it with VSCode:
**tasks.json**:
	- tasks.options.cwd = `/full/path/to/CFWx`

**launch.json**:
	- configurations.cwd = `/full/path/to/CFWx/src`
	- configurations.program = `/full/path/to/CFWx/src/bin/cfwx`

### CMakeLists
Furthermore you have to edit the `CMakeLists.txt` file and set the correct project path, same as above.

### Dependencies
- vcpkg as a package manager for simplicity
- clangd
- cmake
- cmocka
- valgrind 