import path from "node:path";

const ALLOWED_IMAGE_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);
const DEFAULT_CORS_METHODS = ["GET", "POST", "OPTIONS"];
const DEFAULT_CORS_HEADERS = ["Accept", "Accept-Language", "Content-Language", "Content-Type", "Origin"];
const DEFAULT_CORS_EXPOSED_HEADERS = ["X-Request-Id"];

export type CorsConfig = {
  allowedOrigins: "*" | string[];
  allowedMethods?: string[];
  allowedHeaders?: string[];
  exposedHeaders?: string[];
  maxAgeSeconds?: number;
};

export class HttpError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

export function extractAllowedImageExtension(name: string) {
  const ext = path.extname(name).toLowerCase();
  if (ext === ".jpg" || ext === ".jpeg" || ext === ".png" || ext === ".webp") {
    return ext;
  }
  return null;
}

export function guessExtension(name: string, mimeType?: string) {
  const ext = extractAllowedImageExtension(name);
  if (ext) {
    return ext;
  }
  if (mimeType === "image/png") {
    return ".png";
  }
  if (mimeType === "image/webp") {
    return ".webp";
  }
  return ".jpg";
}

export function validateImageUpload(file: File, maxUploadBytes: number) {
  if (file.size <= 0) {
    throw new HttpError(400, "uploaded file is empty");
  }
  if (file.size > maxUploadBytes) {
    throw new HttpError(413, `image exceeds upload limit of ${Math.round(maxUploadBytes / (1024 * 1024))} MB`);
  }

  const mimeType = file.type.toLowerCase();
  const ext = extractAllowedImageExtension(file.name);
  if (!ext && !ALLOWED_IMAGE_TYPES.has(mimeType)) {
    throw new HttpError(415, "only JPG, PNG, and WebP images are supported");
  }

  return {
    extension: guessExtension(file.name, mimeType),
    mimeType,
  };
}

export function jsonResponse(body: unknown, init?: ResponseInit) {
  return Response.json(body, init);
}

function appendHeaderToken(headers: Headers, key: string, value: string) {
  const current = headers.get(key);
  if (!current) {
    headers.set(key, value);
    return;
  }

  const tokens = current
    .split(",")
    .map((token) => token.trim())
    .filter(Boolean);
  if (tokens.some((token) => token.toLowerCase() === value.toLowerCase())) {
    return;
  }

  tokens.push(value);
  headers.set(key, tokens.join(", "));
}

function normalizeOrigin(origin: string) {
  try {
    return new URL(origin).origin;
  } catch {
    return "";
  }
}

function resolveAllowedOrigin(requestOrigin: string | null, corsConfig: CorsConfig) {
  if (!requestOrigin) {
    return null;
  }

  const normalizedOrigin = normalizeOrigin(requestOrigin);
  if (!normalizedOrigin || normalizedOrigin === "null") {
    return null;
  }

  if (corsConfig.allowedOrigins === "*") {
    return "*";
  }

  return corsConfig.allowedOrigins.includes(normalizedOrigin) ? normalizedOrigin : null;
}

function applyCorsHeaders(headers: Headers, request: Request, corsConfig: CorsConfig, preflight = false) {
  const allowedOrigin = resolveAllowedOrigin(request.headers.get("origin"), corsConfig);
  if (!allowedOrigin) {
    return false;
  }

  headers.set("Access-Control-Allow-Origin", allowedOrigin);
  if (allowedOrigin !== "*") {
    appendHeaderToken(headers, "Vary", "Origin");
  }

  const exposedHeaders = corsConfig.exposedHeaders ?? DEFAULT_CORS_EXPOSED_HEADERS;
  if (exposedHeaders.length > 0) {
    headers.set("Access-Control-Expose-Headers", exposedHeaders.join(", "));
  }

  if (!preflight) {
    return true;
  }

  const allowedMethods = corsConfig.allowedMethods ?? DEFAULT_CORS_METHODS;
  headers.set("Access-Control-Allow-Methods", allowedMethods.join(", "));

  const requestHeaders = request.headers.get("access-control-request-headers")?.trim();
  headers.set("Access-Control-Allow-Headers", requestHeaders || (corsConfig.allowedHeaders ?? DEFAULT_CORS_HEADERS).join(", "));
  headers.set("Access-Control-Max-Age", String(corsConfig.maxAgeSeconds ?? 86_400));
  appendHeaderToken(headers, "Vary", "Access-Control-Request-Method");
  appendHeaderToken(headers, "Vary", "Access-Control-Request-Headers");
  return true;
}

export function isCorsPreflightRequest(request: Request) {
  return (
    request.method === "OPTIONS" &&
    Boolean(request.headers.get("origin")) &&
    Boolean(request.headers.get("access-control-request-method"))
  );
}

export function withCommonHeaders(
  response: Response,
  extraHeaders?: Headers | Record<string, string>,
  request?: Request,
  corsConfig?: CorsConfig,
) {
  const headers = new Headers(response.headers);
  headers.set("X-Content-Type-Options", "nosniff");
  headers.set("Referrer-Policy", "no-referrer");
  headers.set("X-Frame-Options", "DENY");
  headers.set("Content-Security-Policy", "default-src 'self'; img-src 'self' data: blob:; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'");

  if (extraHeaders) {
    if (extraHeaders instanceof Headers) {
      for (const [key, value] of extraHeaders.entries()) {
        headers.set(key, value);
      }
    } else {
      for (const [key, value] of Object.entries(extraHeaders)) {
        headers.set(key, value);
      }
    }
  }

  if (request && corsConfig) {
    applyCorsHeaders(headers, request, corsConfig, isCorsPreflightRequest(request));
  }

  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
}

export function corsPreflightResponse(request: Request, corsConfig: CorsConfig) {
  const headers = new Headers();
  const isAllowed = applyCorsHeaders(headers, request, corsConfig, true);

  return new Response(null, {
    status: isAllowed ? 204 : 403,
    headers,
  });
}
