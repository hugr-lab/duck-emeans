# Assemble the site in dist/: both demos, the DuckDB-WASM bridge and the extension.
# WASM is the wasm_eh build from the distribution pipeline, e.g.
#   gh run download <run-id> -n emeans-v2.0.0-extension-wasm_eh -D ..
.PHONY: dist test serve clean

WASM ?= ../emeans.duckdb_extension.wasm

dist:
	./mill --ticker false app.fullLinkJS
	rm -rf dist && mkdir -p dist
	cp ../index.html ../demo.js ../demo-core.js ../duckdb.js ../playground.html ../playground-bridge.js dist/
	cp out/app/fullLinkJS.dest/main.js dist/playground.js
	cp $(WASM) dist/emeans.duckdb_extension.wasm

test:
	./mill --ticker false app.test

serve: dist
	python3 -m http.server -d dist 8099

clean:
	rm -rf dist out
