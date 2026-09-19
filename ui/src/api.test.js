import { afterEach, expect, it, vi } from "vitest";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
  vi.resetModules();
});

it.each([
  [undefined, "/api"],
  ["/testing/", "/testing"],
  ["https://api.example.test/", "https://api.example.test"],
])("uses the API base %s consistently", async (override, base) => {
  vi.stubEnv("VITE_API_BASE_URL", override);
  const fetch = vi.fn(async () => ({
    ok: true,
    json: async () => ({ ok: true }),
  }));
  vi.stubGlobal("fetch", fetch);
  const { request } = await import("./api");
  for (const path of [
    "/health",
    "/model",
    "/features",
    "/predict",
    "/explain",
    "/v2/models",
    "/v3/models",
    "/v2/predict",
    "/v3/predict",
    "/v2/analyze",
    "/v3/analyze",
  ]) {
    await request(path);
    expect(fetch).toHaveBeenLastCalledWith(
      `${base}${path}`,
      expect.any(Object),
    );
  }
});
