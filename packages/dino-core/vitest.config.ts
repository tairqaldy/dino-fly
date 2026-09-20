import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["test/**/*.test.ts"],
    coverage: {
      provider: "v8",
      include: ["src/**/*.ts"],
      exclude: ["src/index.ts", "src/types.ts"],
      reporter: ["text-summary", "json-summary"],
      thresholds: {
        // The engine's step() must be fully branch-covered (see test/engine.coverage.test.ts).
        branches: 95,
        lines: 95,
      },
    },
  },
});
