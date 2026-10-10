"""Idempotently register Transformer tests and examples without replacing the original build."""
from pathlib import Path
p=Path('CMakeLists.txt');s=p.read_text()
if 'NAME transformer_blocks' not in s:
 s+='''
# Modular Transformer regressions and named composition examples.
if(BUILD_TESTING)
    add_executable(cfwx_transformer_test_runner test/transformer_test.c)
    target_link_libraries(cfwx_transformer_test_runner PRIVATE cfwx_core)
    add_test(NAME transformer_blocks COMMAND cfwx_transformer_test_runner)
    add_executable(cfwx_transformer_examples test/transformer_examples.c ${SOURCES_EXPERIMENTS})
    target_link_libraries(cfwx_transformer_examples PRIVATE cfwx_core cfwx_logger)
    add_test(NAME transformer_examples COMMAND cfwx_transformer_examples)
    if(CFWX_SANITIZERS AND CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        foreach(target cfwx_transformer_test_runner cfwx_transformer_examples)
            target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
            set_property(TARGET ${target} APPEND_STRING PROPERTY LINK_FLAGS " -fsanitize=address,undefined")
        endforeach()
    endif()
endif()
'''
 p.write_text(s)
