import { describe, expect, test } from "bun:test";
import { getRuntimeConfig } from "./runtime-config";

describe("runtime config", () => {
  test("uses sane defaults", () => {
    const config = getRuntimeConfig({});
    expect(config.port).toBe(3000);
    expect(config.maxRequestBodySizeBytes).toBeGreaterThan(config.maxUploadBytes);
    expect(config.corsAllowedOrigins).toBe("*");
  });

  test("parses booleans and bounded integers", () => {
    const config = getRuntimeConfig({
      PORT: "99999",
      MAX_UPLOAD_BYTES: "1048576",
      WORKER_JOB_TIMEOUT_MS: "6000",
      ALLOW_CLI_FALLBACK: "false",
    });

    expect(config.port).toBe(65_535);
    expect(config.maxUploadBytes).toBe(1_048_576);
    expect(config.workerJobTimeoutMs).toBe(6_000);
    expect(config.allowCliFallback).toBe(false);
  });

  test("parses explicit CORS origins", () => {
    const config = getRuntimeConfig({
      CORS_ALLOWED_ORIGINS: "http://localhost:5173, https://example.com, invalid-origin",
    });

    expect(config.corsAllowedOrigins).toEqual(["http://localhost:5173", "https://example.com"]);
  });
});
