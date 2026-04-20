type Env = Record<string, string | undefined>;

export type RuntimeConfig = {
  port: number;
  idleTimeoutSeconds: number;
  maxRequestBodySizeBytes: number;
  maxUploadBytes: number;
  maxConcurrentJobs: number;
  maxQueuedJobs: number;
  workerJobTimeoutMs: number;
  allowCliFallback: boolean;
  preloadWorkerOnStartup: boolean;
  logRequests: boolean;
  corsAllowedOrigins: "*" | string[];
};

function parseInteger(value: string | undefined, fallback: number, minimum: number, maximum: number) {
  const numeric = Number.parseInt(value ?? "", 10);
  if (!Number.isFinite(numeric)) {
    return fallback;
  }
  return Math.max(minimum, Math.min(maximum, numeric));
}

function parseBoolean(value: string | undefined, fallback: boolean) {
  if (value == null || value === "") {
    return fallback;
  }
  const normalized = value.trim().toLowerCase();
  if (["1", "true", "yes", "on"].includes(normalized)) {
    return true;
  }
  if (["0", "false", "no", "off"].includes(normalized)) {
    return false;
  }
  return fallback;
}

function parseOrigin(value: string) {
  try {
    return new URL(value).origin;
  } catch {
    return "";
  }
}

function parseCorsAllowedOrigins(value: string | undefined): "*" | string[] {
  if (value == null) {
    return "*";
  }

  const trimmed = value.trim();
  if (!trimmed) {
    return [];
  }
  if (trimmed === "*") {
    return "*";
  }

  const origins = trimmed
    .split(",")
    .map((entry) => parseOrigin(entry.trim()))
    .filter(Boolean);

  return [...new Set(origins)];
}

export function getRuntimeConfig(env: Env = process.env): RuntimeConfig {
  const maxUploadBytes = parseInteger(env.MAX_UPLOAD_BYTES, 25 * 1024 * 1024, 1_024 * 1_024, 100 * 1024 * 1024);
  const maxRequestBodySizeBytes = Math.max(
    maxUploadBytes + 1_024 * 1_024,
    parseInteger(env.MAX_REQUEST_BODY_BYTES, 30 * 1024 * 1024, maxUploadBytes, 128 * 1024 * 1024),
  );

  return {
    port: parseInteger(env.PORT, 3000, 1, 65_535),
    idleTimeoutSeconds: parseInteger(env.IDLE_TIMEOUT_SECONDS, 30, 5, 255),
    maxRequestBodySizeBytes,
    maxUploadBytes,
    maxConcurrentJobs: parseInteger(env.MAX_CONCURRENT_JOBS, 1, 1, 32),
    maxQueuedJobs: parseInteger(env.MAX_QUEUED_JOBS, 8, 0, 256),
    workerJobTimeoutMs: parseInteger(env.WORKER_JOB_TIMEOUT_MS, 120_000, 5_000, 15 * 60_000),
    allowCliFallback: parseBoolean(env.ALLOW_CLI_FALLBACK, true),
    preloadWorkerOnStartup: parseBoolean(env.PRELOAD_WORKER_ON_STARTUP, true),
    logRequests: parseBoolean(env.LOG_REQUESTS, true),
    corsAllowedOrigins: parseCorsAllowedOrigins(env.CORS_ALLOWED_ORIGINS),
  };
}
