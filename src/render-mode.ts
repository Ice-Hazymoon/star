export type RenderMode = "server" | "client";

export const DEFAULT_RENDER_MODE: RenderMode = "server";

export function normalizeRenderMode(input: unknown, fallback: RenderMode = DEFAULT_RENDER_MODE): RenderMode {
  if (typeof input !== "string") {
    return fallback;
  }

  const normalized = input.trim().toLowerCase();
  if (normalized === "server" || normalized === "client") {
    return normalized;
  }

  return fallback;
}

export function shouldRenderOnServer(mode: RenderMode) {
  return mode === "server";
}
