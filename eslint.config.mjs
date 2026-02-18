import js from "@eslint/js";

/** @type {import('eslint').Linter.Config[]} */
export default [
  { ignores: [".next", "node_modules", "*.config.*"] },
  js.configs.recommended,
];
