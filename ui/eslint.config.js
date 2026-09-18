import js from "@eslint/js";
import globals from "globals";
import hooks from "eslint-plugin-react-hooks";

export default [
    {
        ignores: [
            "dist/**",
            "node_modules/**",
            "coverage/**",
            "test-results/**",
            "playwright-report/**",
        ],
    },
    js.configs.recommended,
    {
        files: ["**/*.{js,jsx,mjs}"],
        languageOptions: {
            ecmaVersion: "latest",
            sourceType: "module",
            globals: {...globals.browser, ...globals.node},
            parserOptions: {ecmaFeatures: {jsx: true}},
        },
        plugins: {"react-hooks": hooks},
        rules: {
            ...hooks.configs.recommended.rules,
            "no-unused-vars": ["error", {argsIgnorePattern: "^_"}],
        },
    },
];
