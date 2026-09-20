/** Applies migrations/*.sql in order, once each (tracked in _migrations). Runs on every deploy before the server starts. */
import { readdirSync, readFileSync } from "node:fs";
import pg from "pg";

const url = process.env.DATABASE_URL;
if (!url) {
  console.log("[migrate] DATABASE_URL not set — nothing to do (the API will use its in-memory repo)");
  process.exit(0);
}

const dir = new URL("../migrations/", import.meta.url);
const client = new pg.Client({ connectionString: url });
await client.connect();
await client.query("CREATE TABLE IF NOT EXISTS _migrations (name text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())");
const done = new Set((await client.query<{ name: string }>("SELECT name FROM _migrations")).rows.map((r) => r.name));
for (const name of readdirSync(dir).filter((f) => f.endsWith(".sql")).sort()) {
  if (done.has(name)) continue;
  console.log(`[migrate] applying ${name}`);
  await client.query("BEGIN");
  try {
    await client.query(readFileSync(new URL(name, dir), "utf8"));
    await client.query("INSERT INTO _migrations (name) VALUES ($1)", [name]);
    await client.query("COMMIT");
  } catch (err) {
    await client.query("ROLLBACK");
    throw err;
  }
}
await client.end();
console.log("[migrate] up to date");
