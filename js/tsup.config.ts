import { defineConfig } from "tsup";

export default defineConfig({
  entry: { index: "src/index.ts", "ai-sdk": "src/ai-sdk.ts" },
  format: ["esm", "cjs"],
  dts: true,
  clean: true,
  splitting: true,
  target: "node18",
  external: ["ai"],
});
