import { describe, expect, test } from "bun:test";
import { HttpError, corsPreflightResponse, guessExtension, isCorsPreflightRequest, validateImageUpload, withCommonHeaders } from "./server-utils";

describe("server utils", () => {
  test("guessExtension falls back from mime type", () => {
    expect(guessExtension("mystery", "image/png")).toBe(".png");
    expect(guessExtension("mystery", "image/webp")).toBe(".webp");
  });

  test("validateImageUpload accepts supported images", () => {
    const file = new File(["demo"], "night-sky.jpeg", { type: "image/jpeg" });
    expect(validateImageUpload(file, 10).extension).toBe(".jpeg");
  });

  test("validateImageUpload rejects unsupported uploads", () => {
    const file = new File(["demo"], "notes.txt", { type: "text/plain" });
    expect(() => validateImageUpload(file, 10)).toThrow(HttpError);
  });

  test("validateImageUpload enforces upload limit", () => {
    const file = new File([new Uint8Array(32)], "night-sky.jpg", { type: "image/jpeg" });
    expect(() => validateImageUpload(file, 8)).toThrow(HttpError);
  });

  test("adds wildcard CORS headers for actual requests", () => {
    const request = new Request("http://localhost:3000/api/samples", {
      headers: {
        Origin: "http://localhost:5173",
      },
    });
    const response = withCommonHeaders(new Response("ok"), undefined, request, {
      allowedOrigins: "*",
    });

    expect(response.headers.get("Access-Control-Allow-Origin")).toBe("*");
    expect(response.headers.get("Access-Control-Expose-Headers")).toBe("X-Request-Id");
  });

  test("handles CORS preflight for configured origins", () => {
    const request = new Request("http://localhost:3000/api/analyze-sample", {
      method: "OPTIONS",
      headers: {
        Origin: "https://app.example.com",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "Content-Type, X-Demo",
      },
    });

    expect(isCorsPreflightRequest(request)).toBe(true);

    const response = corsPreflightResponse(request, {
      allowedOrigins: ["https://app.example.com"],
    });

    expect(response.status).toBe(204);
    expect(response.headers.get("Access-Control-Allow-Origin")).toBe("https://app.example.com");
    expect(response.headers.get("Access-Control-Allow-Methods")).toContain("POST");
    expect(response.headers.get("Access-Control-Allow-Headers")).toBe("Content-Type, X-Demo");
    expect(response.headers.get("Vary")).toContain("Origin");
  });

  test("rejects CORS preflight for disallowed origins", () => {
    const request = new Request("http://localhost:3000/api/analyze-sample", {
      method: "OPTIONS",
      headers: {
        Origin: "https://blocked.example.com",
        "Access-Control-Request-Method": "POST",
      },
    });

    const response = corsPreflightResponse(request, {
      allowedOrigins: ["https://app.example.com"],
    });

    expect(response.status).toBe(403);
    expect(response.headers.get("Access-Control-Allow-Origin")).toBeNull();
  });
});
