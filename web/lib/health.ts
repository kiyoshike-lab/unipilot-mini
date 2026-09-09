export type ApiStatus = "Connecting" | "Online" | "Waking API" | "Unavailable";
export function measuredHealthStatus(responseOk: boolean, payload: unknown): ApiStatus {
  if (!responseOk || !payload || typeof payload !== "object") return "Unavailable";
  const value = payload as {status?: unknown; loaded?: unknown};
  return value.status === "ok" && value.loaded === true ? "Online" : "Unavailable";
}
