cmake_minimum_required(VERSION 3.5...3.29)
option(DUCKDB_WASM_EXTENSION "Whether compiling for Wasm target" OFF)

# The emeans extension, built inside a DuckDB build (extension_config.cmake, `make release`), as the C++
# extension template builds. src/cpp/emeans_extension.cpp is the entry point; the C sources in
# src/functions register through DuckDB's C API (duckdb.h) inside the host. See docs/PLAN.md, "Moving
# to the C++ API".
if(NOT COMMAND build_loadable_extension)
    message(FATAL_ERROR "emeans builds as part of DuckDB: run `make release` (see DEVELOPMENT.md)")
endif()

set(EXTENSION_NAME emeans)
project(${EXTENSION_NAME} LANGUAGES C CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Explicit source list (no globbing), grouped by layer.
set(EXTENSION_SOURCES
        # Numerical core: no DuckDB dependency
        src/core/recipe.c
        src/core/rng.c
        src/core/assign.c
        src/core/parallel.c
        src/core/emeans.c
        src/core/geo.c
        # SQL surface: vector I/O and function registration
        src/functions/vec_io.c
        src/functions/emeans_recipe.c
        src/functions/emeans_cluster.c
        src/functions/emeans_assign.c
        src/functions/emeans_trace.c
        src/functions/emeans_agg.c
        src/functions/emeans_geo.c
        src/emeans_extension.c
        # C++: entry point, buffer-manager memory, the query, the window function
        src/cpp/emeans_agg.cpp
        src/cpp/emeans_cluster.cpp
        src/cpp/emeans_extension.cpp
        src/cpp/emeans_mem.cpp
        src/cpp/emeans_query.cpp
        src/cpp/emeans_window.cpp
)

build_static_extension(${EXTENSION_NAME} ${EXTENSION_SOURCES})
build_loadable_extension(${EXTENSION_NAME} " " ${EXTENSION_SOURCES})

foreach(_target ${EXTENSION_NAME}_extension ${EXTENSION_NAME}_loadable_extension)
    # DUCKDB_C_API empty: duckdb.h would declare the C API dllimport on Windows, but this extension links
    # DuckDB statically, and mingw's linker (unlike MSVC's) refuses __imp_ references to static symbols.
    target_compile_definitions(${_target} PRIVATE "DUCKDB_C_API=")
    target_include_directories(${_target} PRIVATE src/include)
    set_property(TARGET ${_target} PROPERTY C_STANDARD 11)
    # Threads for the batch assignment (src/core/parallel.c), and on macOS Accelerate for the kernel's dot
    # products (src/core/assign.c); the WASM build has neither. DuckDB's extension targets use the plain
    # target_link_libraries signature, and CMake does not let one target mix the two.
    if(NOT DUCKDB_WASM_EXTENSION)
        find_package(Threads REQUIRED)
        target_link_libraries(${_target} Threads::Threads)
    endif()
    if(APPLE AND NOT DUCKDB_WASM_EXTENSION)
        target_compile_definitions(${_target} PRIVATE EMEANS_USE_ACCELERATE)
        target_link_libraries(${_target} "-framework Accelerate")
    endif()
    # tools/crap/crap.py instruments only this extension's code, not DuckDB.
    if(EMEANS_COVERAGE)
        target_compile_options(${_target} PRIVATE -fprofile-instr-generate -fcoverage-mapping -O0 -g)
    endif()
endforeach()

install(TARGETS ${EXTENSION_NAME}_extension EXPORT "${DUCKDB_EXPORT_SET}"
        LIBRARY DESTINATION "${INSTALL_LIB_DIR}" ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
