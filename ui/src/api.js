const BASE = (import.meta.env.VITE_API_BASE_URL || "/api").replace(/\/$/, "");

export async function request(path, { body, signal } = {}) {
  let response;
  try {
    response = await fetch(`${BASE}${path}`, {
      signal,
      ...(body
        ? {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
          }
        : {}),
    });
  } catch (error) {
    if (error.name === "AbortError") throw error;
    throw new Error(
      "The analysis service is unreachable. Check the connection and try again.",
      { cause: error },
    );
  }
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    if (response.status === 503)
      throw new Error(
        "The model is unavailable. Ask the service operator to configure a compatible model, then retry.",
      );
    if (response.status === 501)
      throw new Error(
        "This model does not support SHAP explanations. Your predictions are still available.",
      );
    const detail = Array.isArray(payload?.detail)
      ? payload.detail.map((item) => item.msg).join(" ")
      : payload?.detail;
    throw new Error(
      typeof detail === "string"
        ? detail
        : "The analysis could not be completed. Please try again.",
    );
  }
  if (!payload)
    throw new Error(
      "The service returned an empty response. Please try again.",
    );
  return payload;
}
