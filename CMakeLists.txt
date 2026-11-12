cmake_minimum_required(VERSION 3.5...3.29)
option(DUCKDB_WASM_EXTENSION "Whether compiling for Wasm target" OFF)

###
# Configuration
###
# Two builds. Inside a DuckDB build (extension_config.cmake, `make release`) this is the C++ extension:
# src/cpp/emeans_extension.cpp is the entry point and the C sources register through DuckDB's C API on
# the host (docs/PLAN.md, "Moving to the C++ API"). Standalone (Makefile.capi, tools/crap) it is the
# C API extension it was before. EMEANS_CAPI=ON forces the second inside a DuckDB build.
option(EMEANS_CAPI "Build the stable C API extension even inside a DuckDB build" OFF)
if(NOT DEFINED EXTENSION_NAME)
    set(EXTENSION_NAME emeans)
endif()
if (DEFINED TARGET_DUCKDB_VERSION_MAJOR)
    add_definitions(-DDUCKDB_EXTENSION_API_VERSION_MAJOR=${TARGET_DUCKDB_VERSION_MAJOR})
endif()
if (DEFINED TARGET_DUCKDB_VERSION_MINOR)
    add_definitions(-DDUCKDB_EXTENSION_API_VERSION_MINOR=${TARGET_DUCKDB_VERSION_MINOR})
endif()
if (DEFINED TARGET_DUCKDB_VERSION_PATCH)
    add_definitions(-DDUCKDB_EXTENSION_API_VERSION_PATCH=${TARGET_DUCKDB_VERSION_PATCH})
endif()

add_definitions(-DDUCKDB_EXTENSION_NAME=${EXTENSION_NAME})

if (DEFINED DUCKDB_EXTENSION_API_VERSION_UNSTABLE)
    message(FATAL_ERROR "emeans targets the stable C API only; do not set USE_UNSTABLE_C_API")
endif()

###
# Build
###
project(${EXTENSION_NAME} LANGUAGES C CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

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
        src/functions/emeans_mem_capi.c
        # Entrypoint
        src/emeans_extension.c
)

# Threads for the batch assignment (src/core/parallel.c), and on macOS Accelerate for the kernel's dot
# products (src/core/assign.c). The WASM build has neither. DuckDB's extension targets use the plain
# target_link_libraries signature, and CMake does not let one target mix the two, so the C++ build clears
# EMEANS_LINK_SCOPE.
set(EMEANS_LINK_SCOPE PRIVATE)
function(emeans_link_platform target)
    if (NOT DUCKDB_WASM_EXTENSION)
        find_package(Threads REQUIRED)
        target_link_libraries(${target} ${EMEANS_LINK_SCOPE} Threads::Threads)
    endif()
    if (APPLE AND NOT DUCKDB_WASM_EXTENSION)
        target_compile_definitions(${target} PRIVATE EMEANS_USE_ACCELERATE)
        target_link_libraries(${target} ${EMEANS_LINK_SCOPE} "-framework Accelerate")
    endif()
endfunction()

# In-tree C++ build. The C sources see cpp_shim/duckdb_extension.h instead of duckdb_capi/'s, so their
# duckdb_* calls bind to the C API inside DuckDB rather than to an extension function table.
if(COMMAND build_loadable_extension AND NOT EMEANS_CAPI)
    set(EMEANS_CPP_SOURCES ${EXTENSION_SOURCES} src/cpp/emeans_extension.cpp src/cpp/emeans_mem.cpp)
    set(EMEANS_LINK_SCOPE "")
    build_static_extension(${EXTENSION_NAME} ${EMEANS_CPP_SOURCES})
    build_loadable_extension(${EXTENSION_NAME} " " ${EMEANS_CPP_SOURCES})
    foreach(_target ${EXTENSION_NAME}_extension ${EXTENSION_NAME}_loadable_extension)
        target_compile_definitions(${_target} PRIVATE EMEANS_CPP_BUILD)
        # BEFORE: DuckDB's own src/include has a duckdb_extension.h, which must not win over the shim.
        target_include_directories(${_target} BEFORE PRIVATE src/include/cpp_shim src/include)
        set_property(TARGET ${_target} PROPERTY C_STANDARD 11)
        emeans_link_platform(${_target})
        # tools/crap/crap.py instruments only this extension's code, not DuckDB.
        if(EMEANS_COVERAGE)
            target_compile_options(${_target} PRIVATE -fprofile-instr-generate -fcoverage-mapping -O0 -g)
        endif()
    endforeach()
    install(TARGETS ${EXTENSION_NAME}_extension EXPORT "${DUCKDB_EXPORT_SET}"
            LIBRARY DESTINATION "${INSTALL_LIB_DIR}" ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
    return()
endif()

# In-tree build as part of a DuckDB build (duckdb_extension_load(... SOURCE_DIR ...)).
if(COMMAND build_static_extension_capi)
    if(NOT DEFINED TARGET_DUCKDB_VERSION_MINOR)
        set(TARGET_DUCKDB_VERSION_MINOR 5)
    endif()
    if(NOT DEFINED TARGET_DUCKDB_VERSION_PATCH)
        set(TARGET_DUCKDB_VERSION_PATCH 6)
    endif()
    if(NOT DEFINED TARGET_DUCKDB_VERSION_MAJOR)
        set(TARGET_DUCKDB_VERSION_MAJOR 1)
    endif()
    build_loadable_extension_capi(${EXTENSION_NAME} ${TARGET_DUCKDB_VERSION_MAJOR} ${TARGET_DUCKDB_VERSION_MINOR} ${TARGET_DUCKDB_VERSION_PATCH} ${EXTENSION_SOURCES})
    build_static_extension_capi(${EXTENSION_NAME} ${TARGET_DUCKDB_VERSION_MAJOR} ${TARGET_DUCKDB_VERSION_MINOR} ${TARGET_DUCKDB_VERSION_PATCH} ${EXTENSION_SOURCES})
    foreach(_capi_target ${EXTENSION_NAME}_loadable_extension ${EXTENSION_NAME}_extension)
        target_include_directories(${_capi_target} PRIVATE src/include)
        emeans_link_platform(${_capi_target})
    endforeach()
    return()
endif()

if (DUCKDB_WASM_EXTENSION)
	add_library(${EXTENSION_NAME} STATIC ${EXTENSION_SOURCES})
else()
	add_library(${EXTENSION_NAME} SHARED ${EXTENSION_SOURCES})
endif()

# Hide symbols: only the entrypoint is exported (DUCKDB_CAPI_ENTRY_VISIBILITY)
set_target_properties(${EXTENSION_NAME} PROPERTIES C_VISIBILITY_PRESET hidden)
if (NOT APPLE AND NOT WIN32)
    # Strip release binaries (ld64 on macOS treats -s as obsolete)
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} -s")
endif()

if (CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${EXTENSION_NAME} PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

# Include own headers
target_include_directories(${EXTENSION_NAME} PRIVATE src/include)

# Include DuckDB C API headers
target_include_directories(${EXTENSION_NAME} PRIVATE duckdb_capi)

emeans_link_platform(${EXTENSION_NAME})

# libm for pow/log2/cos
if (NOT WIN32 AND NOT DUCKDB_WASM_EXTENSION)
    target_link_libraries(${EXTENSION_NAME} PRIVATE m)
endif()
