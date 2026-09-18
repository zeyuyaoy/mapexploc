import {defineConfig} from "vite";

const apiTarget = process.env.MAPEXPLOC_API_PROXY || "http://127.0.0.1:8000";
export default defineConfig({
    oxc: {jsx: {runtime: "automatic"}},
    test: {
        include: ["src/**/*.test.{js,jsx}"],
        environment: "jsdom",
        setupFiles: "./src/test-setup.js",
    },
    server: {
        proxy: Object.fromEntries(
            ["/health", "/model", "/features", "/predict", "/explain"].map((path) => [
                path,
                apiTarget,
            ]),
        ),
    },
});
