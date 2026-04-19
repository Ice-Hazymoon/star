import { describe, expect, test } from "bun:test";
import { normalizeRenderMode, shouldRenderOnServer } from "./render-mode";

describe("render mode", () => {
  test("defaults to server", () => {
    expect(normalizeRenderMode(undefined)).toBe("server");
    expect(normalizeRenderMode("")).toBe("server");
  });

  test("accepts server and client values", () => {
    expect(normalizeRenderMode("server")).toBe("server");
    expect(normalizeRenderMode("client")).toBe("client");
    expect(normalizeRenderMode(" CLIENT ")).toBe("client");
  });

  test("falls back for unsupported values", () => {
    expect(normalizeRenderMode("both")).toBe("server");
    expect(normalizeRenderMode("none", "client")).toBe("client");
  });

  test("knows when server rendering is required", () => {
    expect(shouldRenderOnServer("server")).toBe(true);
    expect(shouldRenderOnServer("client")).toBe(false);
  });
});
