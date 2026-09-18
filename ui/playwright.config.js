import {defineConfig} from "@playwright/test";

export default defineConfig({
    testDir: "./e2e",
    fullyParallel: false,
    use: {
        baseURL: process.env.MAPEXPLOC_UI_URL || "http://127.0.0.1:5173",
        headless: true,
        launchOptions: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE
            ? {executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE}
            : {},
    },
});
