cmake_minimum_required(VERSION 3.10...3.31)

# vcpkg integration: keep the original default, allow external toolchains.
option(CFWX_USE_VCPKG "Use the default vcpkg toolchain" ON)
if(CFWX_USE_VCPKG AND NOT CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "/opt/vcpkg/scripts/buildsystems/vcpkg.cmake"
        CACHE FILEPATH "vcpkg toolchain")
endif()

set(PROJECT_NAME "CFWx")
set(BINARY "cfwx")
set(PROJECT_PATH "${CMAKE_CURRENT_SOURCE_DIR}" CACHE PATH "Project root / template destination")

# Set project name
project(${PROJECT_NAME} C)

# Default build type
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# Set project C17 standard, with GNU extensions as in the original template.
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)

# Directory structure
set(BINDIR ${CMAKE_SOURCE_DIR}/bin)
set(SRCDIR ${CMAKE_SOURCE_DIR}/src)
set(LOGDIR ${CMAKE_SOURCE_DIR}/log)
set(LIBDIR ${CMAKE_SOURCE_DIR}/lib)
set(TESTDIR ${CMAKE_SOURCE_DIR}/test)
set(BUILDDIR ${CMAKE_BINARY_DIR})
file(MAKE_DIRECTORY ${BINDIR} ${LOGDIR} ${LIBDIR})

# Build flags
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g3 -O0 -DDEBUG=1")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options($<$<COMPILE_LANGUAGE:C>:-Wall> $<$<COMPILE_LANGUAGE:C>:-Wextra>
        $<$<COMPILE_LANGUAGE:C>:-pedantic> $<$<COMPILE_LANGUAGE:C>:-fstack-protector-all>
        $<$<COMPILE_LANGUAGE:C>:-Wstack-protector>)
endif()
add_compile_definitions(PROJECT_ROOT="${PROJECT_PATH}")

# Find source files, preserving the original module structure.
file(GLOB SOURCES_MAIN "${SRCDIR}/*.c")
file(GLOB SOURCES_UTIL "${SRCDIR}/util/*.c")
file(GLOB SOURCES_DS "${SRCDIR}/datastructures/*.c")
file(GLOB SOURCES_MATHS "${SRCDIR}/maths/*.c")
file(GLOB SOURCES_MODELS "${SRCDIR}/models/*.c")
file(GLOB SOURCES_OPTIMIZER "${SRCDIR}/optimizer/*.c")
file(GLOB SOURCES_THREADING "${SRCDIR}/threading/*.c")
file(GLOB SOURCES_EXPERIMENTS "${SRCDIR}/experiments/*.c")
set(SOURCES_CORE ${SOURCES_DS} ${SOURCES_MATHS} ${SOURCES_MODELS}
    ${SOURCES_OPTIMIZER} ${SOURCES_THREADING})
set(SOURCES ${SOURCES_MAIN} ${SOURCES_UTIL} ${SOURCES_EXPERIMENTS})
include_directories("${CMAKE_SOURCE_DIR}/include")

# Set output directories for binaries, per config.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/Debug)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/Release)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/Debug)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/Release)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/Debug)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/Release)

# Numerical library shared by the application and test executables.
add_library(cfwx_core STATIC ${SOURCES_CORE})
target_include_directories(cfwx_core PUBLIC "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(cfwx_core PUBLIC m)

# Logging stays separate from backend-independent numerical kernels.
add_library(cfwx_logger STATIC ${SRCDIR}/util/logger.c)
target_include_directories(cfwx_logger PUBLIC "${CMAKE_SOURCE_DIR}/include")
list(REMOVE_ITEM SOURCES "${SRCDIR}/util/logger.c")

# Main executable
add_executable(${BINARY} ${SOURCES})
target_link_libraries(${BINARY} PRIVATE cfwx_core cfwx_logger)
target_compile_definitions(${BINARY} PRIVATE CMAKE_EXPORT_COMPILE_COMMANDS=1)

# Libs / testing
include(CTest)
if(BUILD_TESTING)
    # Keep the existing full numerical regression suite independent of cmocka.
    add_executable(${BINARY}_numerical_test_runner ${TESTDIR}/main.c)
    target_link_libraries(${BINARY}_numerical_test_runner PRIVATE cfwx_core)
    set_target_properties(${BINARY}_numerical_test_runner PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${BINDIR})
    add_test(NAME numerical_foundations COMMAND ${BINARY}_numerical_test_runner)
endif()

## CMocka with VCPKG
if(CFWX_USE_VCPKG AND NOT cmocka_DIR)
    set(cmocka_DIR "/opt/vcpkg/installed/x64-linux/share/cmocka"
        CACHE PATH "cmocka package configuration directory")
endif()
find_package(cmocka CONFIG QUIET)
set(CFWX_CMOCKA_TARGET "")
if(cmocka_FOUND AND TARGET cmocka::cmocka)
    set(CFWX_CMOCKA_TARGET cmocka::cmocka)
elseif(cmocka_FOUND AND TARGET cmocka)
    # Some system cmocka packages export an unnamespaced target.
    set(CFWX_CMOCKA_TARGET cmocka)
else()
    # Fallback to system cmocka if the vcpkg configuration is unavailable.
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(CMOCKA QUIET IMPORTED_TARGET cmocka)
        if(CMOCKA_FOUND)
            set(CFWX_CMOCKA_TARGET PkgConfig::CMOCKA)
        endif()
    endif()
endif()
if(CFWX_CMOCKA_TARGET)
    target_link_libraries(${BINARY} PRIVATE ${CFWX_CMOCKA_TARGET})
    if(BUILD_TESTING)
        set(TEST_BINARY "${BINARY}_test_runner")
        add_executable(${TEST_BINARY} ${TESTDIR}/cmocka_main.c ${SOURCES_EXPERIMENTS})
        target_link_libraries(${TEST_BINARY} PRIVATE cfwx_core cfwx_logger ${CFWX_CMOCKA_TARGET})
        set_target_properties(${TEST_BINARY} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${BINDIR})
        add_test(NAME unit_tests COMMAND ${TEST_BINARY})
    endif()
else()
    message(WARNING "cmocka not found: cmocka unit_tests unavailable; numerical suite remains enabled.")
endif()

# Optional sanitizers. LINK_FLAGS keeps compatibility with older CMake.
option(CFWX_SANITIZERS "Enable ASan and UBSan" OFF)
if(CFWX_SANITIZERS AND CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    foreach(target cfwx_core cfwx_logger ${BINARY} ${BINARY}_numerical_test_runner ${BINARY}_test_runner)
        if(TARGET ${target})
            target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
            set_property(TARGET ${target} APPEND_STRING PROPERTY LINK_FLAGS " -fsanitize=address,undefined")
        endif()
    endforeach()
endif()

# Verify logger injection even without cmocka.
if(BUILD_TESTING)
    add_executable(${BINARY}_logger_test_runner ${TESTDIR}/logger_test.c ${SOURCES_EXPERIMENTS})
    target_link_libraries(${BINARY}_logger_test_runner PRIVATE cfwx_core cfwx_logger)
    add_test(NAME experiment_logging COMMAND ${BINARY}_logger_test_runner)
    if(CFWX_SANITIZERS AND CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${BINARY}_logger_test_runner PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
        set_property(TARGET ${BINARY}_logger_test_runner APPEND_STRING PROPERTY LINK_FLAGS " -fsanitize=address,undefined")
    endif()
endif()

# Custom targets for additional functionality
# Clear: remove generated outputs, not the active build system or source libraries.
# Deleting BUILDDIR while its build tool is executing is unsafe; use a fresh
# external shell to remove the entire build tree when a full reset is desired.
add_custom_target(clear
    COMMAND ${CMAKE_COMMAND} -E echo "Clearing generated project outputs..."
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${BINDIR}
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${LOGDIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${BINDIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${LIBDIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${LOGDIR}
    COMMAND ${CMAKE_COMMAND} -E echo "Use the clean target to remove build-tree binaries."
    VERBATIM)

add_custom_target(welp
    COMMAND ${CMAKE_COMMAND} -E echo "C Project Template"
    COMMAND ${CMAKE_COMMAND} -E echo "Target rules:"
    COMMAND ${CMAKE_COMMAND} -E echo "    all      - Compile binaries: cmake --build <build-dir>"
    COMMAND ${CMAKE_COMMAND} -E echo "    test     - Run tests: ctest --test-dir <build-dir>"
    COMMAND ${CMAKE_COMMAND} -E echo "    start    - Copy template to PROJECT_PATH (must differ from source)"
    COMMAND ${CMAKE_COMMAND} -E echo "    valgrind - Run application under Valgrind, when installed"
    COMMAND ${CMAKE_COMMAND} -E echo "    clean    - Remove build-tree binaries"
    COMMAND ${CMAKE_COMMAND} -E echo "    clear    - Remove bin and log outputs"
    COMMAND ${CMAKE_COMMAND} -E echo "    welp     - Print this help"
    VERBATIM)

# Prevent recursive copying into the source tree or copying onto itself.
add_custom_target(start
    COMMAND ${CMAKE_COMMAND}
        -DTEMPLATE_SOURCE=${CMAKE_SOURCE_DIR} -DTEMPLATE_DESTINATION=${PROJECT_PATH}
        -P ${CMAKE_SOURCE_DIR}/cmake/start_project.cmake
    VERBATIM)

find_program(VALGRIND_EXECUTABLE valgrind)
if(VALGRIND_EXECUTABLE)
    add_custom_target(valgrind
        COMMAND ${CMAKE_COMMAND} -E make_directory ${LOGDIR}
        COMMAND ${VALGRIND_EXECUTABLE}
            --track-origins=yes --leak-check=full --leak-resolution=high
            --error-exitcode=1 --log-file=${LOGDIR}/valgrind.log
            $<TARGET_FILE:${BINARY}>
        COMMAND ${CMAKE_COMMAND} -E echo "Check the log file: ${LOGDIR}/valgrind.log"
        DEPENDS ${BINARY} VERBATIM)
endif()

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

# Actual dataset trainer, available independently of BUILD_TESTING.
add_executable(cfwx_train test/train_transformer.c ${SOURCES_EXPERIMENTS})
target_link_libraries(cfwx_train PRIVATE cfwx_core cfwx_logger)
if(BUILD_TESTING)
    add_executable(cfwx_training_test_runner test/training_test.c)
    target_link_libraries(cfwx_training_test_runner PRIVATE cfwx_core)
    add_test(NAME transformer_gradients COMMAND cfwx_training_test_runner)
    add_test(NAME transformer_training COMMAND cfwx_train)
endif()
if(CFWX_SANITIZERS AND CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    foreach(target cfwx_train cfwx_training_test_runner)
        if(TARGET ${target})
            target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
            set_property(TARGET ${target} APPEND_STRING PROPERTY LINK_FLAGS " -fsanitize=address,undefined")
        endif()
    endforeach()
endif()

if(BUILD_TESTING)
    add_executable(cfwx_training_extra_test_runner test/training_extra_test.c)
    target_link_libraries(cfwx_training_extra_test_runner PRIVATE cfwx_core)
    add_test(NAME transformer_training_extra COMMAND cfwx_training_extra_test_runner)
    if(CFWX_SANITIZERS AND CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(cfwx_training_extra_test_runner PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
        set_property(TARGET cfwx_training_extra_test_runner APPEND_STRING PROPERTY LINK_FLAGS " -fsanitize=address,undefined")
    endif()
endif()

# Lifecycle regressions and extended training/generation driver.
add_executable(cfwx_run test/runtime_driver.c)
target_link_libraries(cfwx_run PRIVATE cfwx_core cfwx_logger)
if(BUILD_TESTING)
    add_executable(cfwx_runtime_test_runner test/runtime_test.c)
    target_link_libraries(cfwx_runtime_test_runner PRIVATE cfwx_core)
    add_test(NAME transformer_runtime COMMAND cfwx_runtime_test_runner)
endif()
if(CFWX_SANITIZERS AND CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    foreach(target cfwx_run cfwx_runtime_test_runner)
        if(TARGET ${target})
            target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
            set_property(TARGET ${target} APPEND_STRING PROPERTY LINK_FLAGS " -fsanitize=address,undefined")
        endif()
    endforeach()
endif()

# CUDA is explicitly opt-in; no CUDA configuration in ordinary CPU builds.
include(cmake/cuda_backend.cmake)

if(BUILD_TESTING)
    add_executable(cfwx_cache_options test/cache_options.c)
    target_link_libraries(cfwx_cache_options PRIVATE cfwx_core)
    add_test(NAME cache_options COMMAND cfwx_cache_options)
    if(CFWX_SANITIZERS AND CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(cfwx_cache_options PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
        set_property(TARGET cfwx_cache_options APPEND_STRING PROPERTY LINK_FLAGS " -fsanitize=address,undefined")
    endif()
endif()

# Documentation is additive and never requires configuring CUDA.
include(cmake/doxygen.cmake)
