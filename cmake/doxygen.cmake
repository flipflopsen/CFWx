## @file cmake/doxygen.cmake
# @brief doxygen build helper.
# @ingroup maintenance

# Optional source-only documentation target; no CUDA language/toolkit is enabled.
find_program(CFWX_DOXYGEN_EXECUTABLE doxygen)
if(CFWX_DOXYGEN_EXECUTABLE)
    add_custom_target(docs
        COMMAND ${CFWX_DOXYGEN_EXECUTABLE} ${CMAKE_SOURCE_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Generating complete CFWx HTML/XML reference documentation"
        VERBATIM)
    add_custom_target(docs-clean
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_SOURCE_DIR}/build-docs
        COMMENT "Removing generated documentation only"
        VERBATIM)
else()
    message(STATUS "Doxygen not found: install it to enable the optional docs target")
endif()
