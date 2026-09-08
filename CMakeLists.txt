cmake_minimum_required(VERSION 3.10...3.31)

# vcpkg integration
set(CMAKE_TOOLCHAIN_FILE "/opt/vcpkg/scripts/buildsystems/vcpkg.cmake")

set(PROJECT_NAME "CFWx")
set(BINARY "cfwx")
set(PROJECT_PATH "/full/path/to/CFWx")

# Set project name
project(${PROJECT_NAME} C)

# Default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# Set C standard
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)  # For GNU99

# Directory structure
set(BINDIR ${CMAKE_SOURCE_DIR}/bin)
set(SRCDIR ${CMAKE_SOURCE_DIR}/src)
set(LOGDIR ${CMAKE_SOURCE_DIR}/log)
set(LIBDIR ${CMAKE_SOURCE_DIR}/lib)
set(TESTDIR ${CMAKE_SOURCE_DIR}/test)

# Create directories
file(MAKE_DIRECTORY ${BINDIR} ${LOGDIR} ${LIBDIR})

# Build flags
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g3 -O0 -DDEBUG=1")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Warning flags
add_compile_options(
    -Wall
    -Wextra
    -pedantic
    -fstack-protector-all
    -Wstack-protector
)

# Macros
add_compile_definitions(PROJECT_ROOT="${PROJECT_PATH}")

# Find source files
file(GLOB SOURCES_MAIN "${SRCDIR}/*.c")
file(GLOB SOURCES_UTIL "${SRCDIR}/util/*.c")
file(GLOB SOURCES_DS "${SRCDIR}/datastructures/*.c")
file(GLOB SOURCES_MATHS "${SRCDIR}/maths/*.c")
file(GLOB SOURCES_MODELS "${SRCDIR}/models/*.c")
file(GLOB SOURCES_OPTIMIZER "${SRCDIR}/optimizer/*.c")
file(GLOB SOURCES_THREADING "${SRCDIR}/threading/*.c")
set(SOURCES
        ${SOURCES_MAIN}
        ${SOURCES_UTIL}
        ${SOURCES_DS}
        ${SOURCES_MATHS}
        ${SOURCES_MODELS}
        ${SOURCES_OPTIMIZER}
        ${SOURCES_THREADING})

# Header files
include_directories("${CMAKE_SOURCE_DIR}/include")



# Set output directory for binary
#set_target_properties(${BINARY} PROPERTIES
#    RUNTIME_OUTPUT_DIRECTORY ${BINDIR}
#)

# Set output directories for binaries, per config
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/Debug)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/Release)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/Debug)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/Release)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/Debug)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/Release)

# Main executable
add_executable(${BINARY} ${SOURCES})

target_compile_definitions(${BINARY} PRIVATE CMAKE_EXPORT_COMPILE_COMMANDS=1)
# Libs

## CMocka with VCPKG
set(cmocka_DIR "/opt/vcpkg/installed/x64-linux/share/cmocka")
find_package(cmocka CONFIG)
if(cmocka_FOUND)
    target_link_libraries(${BINARY} cmocka::cmocka)
    # Test executable
    set(TEST_BINARY "${BINARY}_test_runner")
    add_executable(${TEST_BINARY} ${TESTDIR}/main.c)
    target_link_libraries(${TEST_BINARY} cmocka::cmocka)

    set_target_properties(${TEST_BINARY} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${BINDIR}
    )

    # Enable testing
    enable_testing()
    add_test(NAME unit_tests COMMAND ${TEST_BINARY})
else()
    # Fallback to system cmocka if vcpkg version not found
    find_package(PkgConfig)
    if(PkgConfig_FOUND)
        pkg_check_modules(CMOCKA cmocka)
        if(CMOCKA_FOUND)
            set(TEST_BINARY "${BINARY}_test_runner")
            add_executable(${TEST_BINARY} ${TESTDIR}/main.c)
            target_link_libraries(${TEST_BINARY} ${CMOCKA_LIBRARIES})
            target_include_directories(${TEST_BINARY} PRIVATE ${CMOCKA_INCLUDE_DIRS})
            target_compile_options(${TEST_BINARY} PRIVATE ${CMOCKA_CFLAGS_OTHER})

            set_target_properties(${TEST_BINARY} PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY ${BINDIR}
            )

            enable_testing()
            add_test(NAME unit_tests COMMAND ${TEST_BINARY})
        endif()
    endif()
endif()


# Custom targets for additional functionality

# Clear target - removes everything from build, bin, lib, and log directories
add_custom_target(clear
    COMMAND ${CMAKE_COMMAND} -E echo "Clearing project directories..."
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${BUILDDIR}
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${BINDIR}
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${LIBDIR}
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${LOGDIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${BINDIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${LIBDIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${LOGDIR}
    COMMAND ${CMAKE_COMMAND} -E echo "Project directories cleared and recreated."
    VERBATIM
)

# Help target
add_custom_target(welp
    COMMAND ${CMAKE_COMMAND} -E echo "C Project Template"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Target rules:"
    COMMAND ${CMAKE_COMMAND} -E echo "    all      - Compiles and generates binary file (use 'make' or 'cmake --build .')"
    COMMAND ${CMAKE_COMMAND} -E echo "    tests    - Compiles with cmocka and run tests binary file (use 'make test')"
    COMMAND ${CMAKE_COMMAND} -E echo "    start    - Starts a new project using C project template"
    COMMAND ${CMAKE_COMMAND} -E echo "    valgrind - Runs binary file using valgrind tool"
    COMMAND ${CMAKE_COMMAND} -E echo "    clean    - Clean the project by removing binaries (use 'make clean')"
    COMMAND ${CMAKE_COMMAND} -E echo "    help     - Prints a help message with target rules"
)

# Start target
add_custom_target(start
    COMMAND ${CMAKE_COMMAND} -E echo "Creating project: ${PROJECT_NAME}"
    COMMAND ${CMAKE_COMMAND} -E make_directory ${PROJECT_PATH}
    COMMAND ${CMAKE_COMMAND} -E echo "Copying files from template to new directory:"
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR} ${PROJECT_PATH}
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Go to ${PROJECT_PATH} and compile your project: cmake --build ."
    COMMAND ${CMAKE_COMMAND} -E echo "Then execute it: bin/${BINARY} --help"
    COMMAND ${CMAKE_COMMAND} -E echo "Happy hacking o/"
)

# Valgrind target
find_program(VALGRIND_EXECUTABLE valgrind)
if(VALGRIND_EXECUTABLE)
    add_custom_target(valgrind
        COMMAND ${VALGRIND_EXECUTABLE}
            --track-origins=yes
            --leak-check=full
            --leak-resolution=high
            --log-file=${LOGDIR}/valgrind.log
            ${BINDIR}/${BINARY}
        COMMAND ${CMAKE_COMMAND} -E echo "Check the log file: ${LOGDIR}/valgrind.log"
        DEPENDS ${BINARY}
    )
endif()
