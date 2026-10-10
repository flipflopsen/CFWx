## @file cmake/start_project.cmake
# @brief start project build helper.
# @ingroup maintenance

get_filename_component(source "${TEMPLATE_SOURCE}" REALPATH)
get_filename_component(destination "${TEMPLATE_DESTINATION}" ABSOLUTE)
file(MAKE_DIRECTORY "${destination}")
get_filename_component(destination "${destination}" REALPATH)
file(RELATIVE_PATH relative "${source}" "${destination}")
if(relative STREQUAL "" OR relative STREQUAL "." OR
   (NOT relative MATCHES "^\\.\\.(/|$)" AND NOT IS_ABSOLUTE "${relative}"))
    message(FATAL_ERROR "PROJECT_PATH must be outside the template source tree.")
endif()
message(STATUS "Creating project: ${destination}")
file(COPY "${source}/" DESTINATION "${destination}"
    PATTERN ".git" EXCLUDE PATTERN "build*" EXCLUDE PATTERN "bin" EXCLUDE
    PATTERN "*.log" EXCLUDE PATTERN ".cache" EXCLUDE)
message(STATUS "Configure the copy: cmake -S ${destination} -B ${destination}/build")
message(STATUS "Then compile it: cmake --build ${destination}/build")
