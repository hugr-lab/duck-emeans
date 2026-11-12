// The playground's link to DuckDB: the Scala.js app (playground/) calls window.EmeansDuck, through the
// facade in playground/app/src/Duck.scala. Points arrive as flat arrays x0 y0 x1 y1 ..., go to DuckDB as
// CSV, and become one-row tables train and test that emeans_trace reads.
import { startDuckDB } from './duckdb.js';
import { highDimSql, pointsCsv, readResult, readTrace, traceSql } from './demo-core.js';

const TABLES = new Set(['train', 'test']);

async function setPoints(db, name, xy) {
  if (!TABLES.has(name)) {
    throw new Error(`no table ${name}`);
  }
  await db.registerText(`${name}.csv`, pointsCsv(xy));
  await db.query(`CREATE OR REPLACE TABLE ${name} AS ` +
    `SELECT list([x, y]::FLOAT[] ORDER BY i) AS v FROM read_csv('${name}.csv')`);
}

async function timeHighDim(db, n, d, k) {
  const q = highDimSql({ n, d, groups: 16, k, nepochs: 2 });
  const t0 = performance.now();
  await db.query(q.create);
  const t1 = performance.now();
  const r = readResult((await db.query(q.cluster))[0].r);
  const t2 = performance.now();
  return { generateMs: t1 - t0, clusterMs: t2 - t1, batchSize: r.batchSize, passes: r.obj.length,
           objective: r.obj.at(-1), sql: `${q.create};\n${q.cluster};` };
}

window.EmeansDuck = {
  async start(extensionUrl) {
    const db = await startDuckDB(extensionUrl);
    const version = (await db.query('SELECT version() AS v'))[0].v;
    return {
      version,
      setPoints: (name, xy) => setPoints(db, name, xy),
      trace: async options => readTrace((await db.query(traceSql(options)))[0].t),
      traceSql: options => traceSql(options),
      timeHighDim: (n, d, k) => timeHighDim(db, n, d, k),
    };
  },
};
