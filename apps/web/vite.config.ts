import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// Static build, deployable to any static host (Cloudflare Pages / GitHub Pages). `base` is relative so the same
// bundle works under a sub-path; routing is hash-based for the same reason.
export default defineConfig({
  base: "./",
  plugins: [react()],
  server: { port: 5173 },
  test: { include: ["test/**/*.test.ts"] },
});
