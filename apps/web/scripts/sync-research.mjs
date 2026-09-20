// Copies docs/RESEARCH.md + NEURONS.md + figures into public/research so the static site can render them.
import { cpSync, existsSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const docs = resolve(here, "../../../docs");
const out = resolve(here, "../public/research");
mkdirSync(out, { recursive: true });
for (const name of ["RESEARCH.md", "NEURONS.md", "DECISIONS.md", "ARCHITECTURE.md"]) {
  if (existsSync(resolve(docs, name))) cpSync(resolve(docs, name), resolve(out, name));
}
if (existsSync(resolve(docs, "figures"))) cpSync(resolve(docs, "figures"), resolve(out, "figures"), { recursive: true });
console.log("[sync-research] docs copied to public/research");
