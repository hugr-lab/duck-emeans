# Included by DuckDB's build system: the extensions to build with it.
duckdb_extension_load(emeans
    SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}
    LOAD_TESTS
)
