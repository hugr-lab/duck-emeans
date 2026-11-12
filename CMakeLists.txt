cmake_minimum_required(VERSION 3.5...3.29)
option(DUCKDB_WASM_EXTENSION "Whether compiling for Wasm target" OFF)

###
# Configuration
###
if(NOT DEFINED EXTENSION_NAME)
    message(FATAL_ERROR "DuckDB extension name is required")
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
project(${EXTENSION_NAME} LANGUAGES C)
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
        src/core/emeans.c
        # SQL surface: vector I/O and function registration
        src/functions/vec_io.c
        src/functions/emeans_recipe.c
        src/functions/emeans_cluster.c
        src/functions/emeans_assign.c
        # Entrypoint
        src/emeans_extension.c
)

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

# libm for pow/log2/cos
if (NOT WIN32 AND NOT DUCKDB_WASM_EXTENSION)
    target_link_libraries(${EXTENSION_NAME} PRIVATE m)
endif()
