// DuckDB-WASM with the emeans extension loaded.
//
// The version is pinned: an extension only loads in the DuckDB release it was built for. This release is
// the DuckDB-WASM build of v2.0.0 (npm's "latest" tag was still 1.5.4 when this was written). The
// extension is not signed by DuckDB, so the database is opened with unsigned extensions allowed.
const VERSION = '1.33.1-dev114.0';

async function instantiate(duckdb) {
  const bundle = await duckdb.selectBundle(duckdb.getJsDelivrBundles());
  const workerUrl = URL.createObjectURL(
    new Blob([`importScripts("${bundle.mainWorker}");`], { type: 'text/javascript' }),
  );
  const db = new duckdb.AsyncDuckDB(new duckdb.VoidLogger(), new Worker(workerUrl));
  await db.instantiate(bundle.mainModule, bundle.pthreadWorker);
  URL.revokeObjectURL(workerUrl);
  await db.open({ allowUnsignedExtensions: true });
  return db;
}

export async function startDuckDB(extensionUrl) {
  const duckdb = await import(`https://cdn.jsdelivr.net/npm/@duckdb/duckdb-wasm@${VERSION}/+esm`);
  const db = await instantiate(duckdb);
  const conn = await db.connect();
  await conn.query(`LOAD '${new URL(extensionUrl, location.href).href}'`);
  return {
    query: async sql => (await conn.query(sql)).toArray(),
    // A text file DuckDB can read by name, such as a CSV for read_csv.
    registerText: (name, text) => db.registerFileText(name, text),
  };
}
