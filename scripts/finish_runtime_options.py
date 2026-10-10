## @file scripts/finish_runtime_options.py
# @brief finish runtime options source/document maintenance tool.
# @ingroup maintenance
# @details Textual maintenance only; does not itself validate numerical correctness.

"""Finalize source integration only; CUDA compilation/execution remains manual."""
from pathlib import Path
import re
p=Path('test/runtime_driver.c');s=p.read_text()
s=re.sub(r'(static int generate\([^{}]*?size_t count)\)',r'\1, size_t window, cfwx_cache_mode mode)',s,count=1)
p.write_text(s)
p=Path('test/cuda_manual.c');s=p.read_text()
if 'CFWX_CUDA_FUSED_BATCH' not in s:
 # Both manual test and dataset train read an explicit environment opt-in.
 s=s.replace('    size_t count = cfwx_cuda_parameter_count(m);', '''    const char *fused = getenv("CFWX_CUDA_FUSED_BATCH");
    if (fused && strcmp(fused, "1") == 0) cfwx_cuda_set_fused_batching(m, 1);
    size_t count = cfwx_cuda_parameter_count(m);''')
 s=s.replace('    size_t ids[32], targets[32];', '''    const char *fused = getenv("CFWX_CUDA_FUSED_BATCH");
    if (fused && strcmp(fused, "1") == 0) cfwx_cuda_set_fused_batching(m, 1);
    size_t ids[32], targets[32];''')
 p.write_text(s)
p=Path('CMakeLists.txt');s=p.read_text()
if 'NAME cache_options' not in s:
 s+='''
if(BUILD_TESTING)
    add_executable(cfwx_cache_options test/cache_options.c)
    target_link_libraries(cfwx_cache_options PRIVATE cfwx_core)
    add_test(NAME cache_options COMMAND cfwx_cache_options)
    if(CFWX_SANITIZERS AND CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(cfwx_cache_options PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
        set_property(TARGET cfwx_cache_options APPEND_STRING PROPERTY LINK_FLAGS " -fsanitize=address,undefined")
    endif()
endif()
'''
 p.write_text(s)
# Account for segment metadata in device budget, including allocation after model create.
p=Path('src/gpu/cuda_training.cu');s=p.read_text()
if 'm->config.batch * sizeof(Segment)' not in s:
 s=s.replace('tokens * sizeof(size_t) + sizeof(int)', 'tokens * sizeof(size_t) + sizeof(int) + m->config.batch * sizeof(Segment)')
p.write_text(s)
import subprocess
subprocess.run(['clang-format','-i','test/runtime_driver.c','test/cache_options.c','test/cuda_manual.c','src/gpu/cuda_training.cu'],check=True)
