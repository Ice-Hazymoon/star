import { existsSync, readdirSync, statSync } from "node:fs";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import {
  ASTROMETRY_DIR,
  CATALOG_DIR,
  PUBLIC_DIR,
  PYTHON_BIN_CANDIDATES,
  PYTHON_SCRIPT,
  PYTHON_WORKER_SCRIPT,
  REFERENCE_DIR,
  ROOT_DIR,
  REQUIRED_ASTROMETRY_INDEXES,
  SAMPLE_IMAGES,
  SAMPLES_DIR,
  STARDROID_CONSTELLATIONS_PATH,
  STARDROID_DSO_PATH,
  STARDROID_ENGLISH_LOCALIZATION_PATH,
  STARDROID_LOCALES_DIR,
  SUPPLEMENTAL_DSO_PATH,
} from "./app-config";
import type { AnnotationApiResponse, RawAnnotationResult } from "./api-types";
import {
  cloneOverlayOptions,
  DEFAULT_OVERLAY_OPTIONS,
  normalizeOverlayOptions,
  OVERLAY_PRESETS,
  type OverlayOptions,
} from "./overlay-options";
import {
  DEFAULT_RENDER_MODE,
  normalizeRenderMode,
  shouldRenderOnServer,
  type RenderMode,
} from "./render-mode";
import { getRuntimeConfig } from "./runtime-config";
import {
  corsPreflightResponse,
  HttpError,
  isCorsPreflightRequest,
  guessExtension,
  jsonResponse,
  validateImageUpload,
  withCommonHeaders,
} from "./server-utils";
import {
  createJobLimiter,
  JobQueueAbortedError,
  JobQueueFullError,
} from "./job-limiter";

type AnnotationWorkerRequest = {
  id: string;
  action: "annotate" | "ping" | "preload";
  input_path?: string;
  output_image_path?: string;
  index_dir?: string;
  catalog_path?: string;
  constellation_paths?: string[];
  star_names_path?: string;
  dso_paths?: string[];
  localization_paths?: string[];
  supplemental_dso_path?: string;
  locale?: string;
  overlay_options?: OverlayOptions;
};

type AnnotationWorkerResponse = {
  id: string | null;
  ok: boolean;
  result?: Record<string, unknown>;
  error?: string;
};

type PendingWorkerRequest = {
  worker: ReturnType<typeof Bun.spawn>;
  resolve: (value: Record<string, unknown>) => void;
  reject: (reason?: unknown) => void;
  timeout: ReturnType<typeof setTimeout>;
  cleanup: () => void;
};

const CONFIG = getRuntimeConfig();
const jobLimiter = createJobLimiter(CONFIG.maxConcurrentJobs, CONFIG.maxQueuedJobs);

const pendingWorkerRequests = new Map<string, PendingWorkerRequest>();
let workerProcess: ReturnType<typeof Bun.spawn> | null = null;
let workerReady = false;
let shuttingDown = false;
let workerWarmupPromise: Promise<void> | null = null;

function logInfo(message: string, metadata?: Record<string, unknown>) {
  if (!CONFIG.logRequests) {
    return;
  }
  console.log(`[star-server] ${message}${metadata ? ` ${JSON.stringify(metadata)}` : ""}`);
}

function logError(message: string, error?: unknown, metadata?: Record<string, unknown>) {
  const payload = {
    ...(metadata ?? {}),
    error: error instanceof Error ? error.message : error,
  };
  console.error(`[star-server] ${message}${Object.keys(payload).length ? ` ${JSON.stringify(payload)}` : ""}`);
}

// Pipeline failures that aren't going to be fixed by retrying via the CLI —
// the CLI runs the exact same Python pipeline on the same input, so e.g. a
// plate-solve timeout will recur and just double the client-visible latency.
const DETERMINISTIC_PIPELINE_ERROR_PATTERNS = [
  /plate solving aborted after/i,
];

function isDeterministicPipelineError(error: unknown): boolean {
  const message = error instanceof Error ? error.message : typeof error === "string" ? error : "";
  return DETERMINISTIC_PIPELINE_ERROR_PATTERNS.some((pattern) => pattern.test(message));
}

function parseOverlayOptionsFromFormData(formData: { get(name: string): string | File | null }) {
  const rawOptions = formData.get("options");
  if (rawOptions == null || rawOptions === "") {
    return cloneOverlayOptions();
  }
  if (typeof rawOptions !== "string") {
    throw new HttpError(400, "invalid overlay options payload");
  }
  try {
    return normalizeOverlayOptions(JSON.parse(rawOptions));
  } catch {
    throw new HttpError(400, "invalid overlay options JSON");
  }
}

function parseRenderModeFromFormData(formData: { get(name: string): string | File | null }) {
  const rawRenderMode = formData.get("render_mode");
  if (rawRenderMode == null || rawRenderMode === "") {
    return DEFAULT_RENDER_MODE;
  }
  if (typeof rawRenderMode !== "string") {
    throw new HttpError(400, "invalid render mode payload");
  }
  return normalizeRenderMode(rawRenderMode);
}

function normalizeLocaleTag(rawLocale: unknown) {
  if (typeof rawLocale !== "string") {
    return "";
  }
  const trimmed = rawLocale.replaceAll("_", "-").trim();
  if (!trimmed) {
    return "";
  }
  const parts = trimmed.split("-").filter(Boolean);
  if (parts.length === 0) {
    return "";
  }
  return parts
    .map((part, index) => {
      if (index === 0) {
        return part.toLowerCase();
      }
      if (part.length === 4 && /^[a-z]+$/i.test(part)) {
        return `${part[0].toUpperCase()}${part.slice(1).toLowerCase()}`;
      }
      if ((part.length === 2 || part.length === 3) && /^[a-z0-9]+$/i.test(part)) {
        return part.toUpperCase();
      }
      return part;
    })
    .join("-");
}

function parsePrimaryAcceptLanguage(headerValue: string | null) {
  if (!headerValue) {
    return "";
  }
  const firstToken = headerValue
    .split(",")[0]
    ?.split(";")[0]
    ?.trim();
  return normalizeLocaleTag(firstToken);
}

function parseLocaleFromFormData(
  formData: { get(name: string): string | File | null },
  acceptLanguageHeader: string | null,
) {
  const rawLocale = formData.get("locale");
  if (typeof rawLocale === "string" && rawLocale.trim()) {
    return normalizeLocaleTag(rawLocale) || "en";
  }
  return parsePrimaryAcceptLanguage(acceptLanguageHeader) || "en";
}

function listFilesRecursive(rootDir: string, targetFilename: string) {
  if (!existsSync(rootDir)) {
    return [];
  }

  const files: string[] = [];
  const queue = [rootDir];
  while (queue.length > 0) {
    const currentDir = queue.shift();
    if (!currentDir) {
      continue;
    }
    for (const entry of readdirSync(currentDir)) {
      const absolutePath = path.join(currentDir, entry);
      const stats = statSync(absolutePath);
      if (stats.isDirectory()) {
        queue.push(absolutePath);
        continue;
      }
      if (stats.isFile() && path.basename(absolutePath) === targetFilename) {
        files.push(absolutePath);
      }
    }
  }

  return files.sort();
}

function androidValuesDirectoryToLocale(valuesDirName: string) {
  if (valuesDirName === "values") {
    return "en";
  }
  if (valuesDirName.startsWith("values-b+")) {
    return normalizeLocaleTag(valuesDirName.slice("values-b+".length).replaceAll("+", "-")) || "en";
  }
  if (valuesDirName.startsWith("values-")) {
    return normalizeLocaleTag(valuesDirName.slice("values-".length)) || "en";
  }
  return "en";
}

function listLocalizationPaths() {
  return listFilesRecursive(STARDROID_LOCALES_DIR, "celestial_objects.xml");
}

function listAvailableLocales() {
  return [...new Set(
    listLocalizationPaths().map((filePath) => androidValuesDirectoryToLocale(path.basename(path.dirname(filePath)))),
  )].sort((left, right) => left.localeCompare(right));
}

function resolvePythonBinary() {
  for (const candidate of PYTHON_BIN_CANDIDATES) {
    if (candidate.includes(path.sep)) {
      if (existsSync(candidate)) {
        return candidate;
      }
      continue;
    }
    return candidate;
  }
  return "python3";
}

function resolvePublicImageUrl(imagePath: string) {
  if (imagePath.startsWith(`${SAMPLES_DIR}${path.sep}`)) {
    return `/samples/${path.basename(imagePath)}`;
  }
  return undefined;
}

function buildWorkerAssetPayload(
  overlayOptions: OverlayOptions,
  locale: string,
): Omit<AnnotationWorkerRequest, "id" | "action"> {
  return {
    index_dir: ASTROMETRY_DIR,
    catalog_path: path.join(CATALOG_DIR, "minimal_hipparcos.csv"),
    constellation_paths: [
      path.join(REFERENCE_DIR, "modern_st.json"),
      ...(existsSync(STARDROID_CONSTELLATIONS_PATH) ? [STARDROID_CONSTELLATIONS_PATH] : []),
    ],
    star_names_path: path.join(REFERENCE_DIR, "common_star_names.fab"),
    dso_paths: [
      path.join(REFERENCE_DIR, "NGC.csv"),
      ...(existsSync(STARDROID_DSO_PATH) ? [STARDROID_DSO_PATH] : []),
    ],
    localization_paths: listLocalizationPaths(),
    supplemental_dso_path: existsSync(SUPPLEMENTAL_DSO_PATH) ? SUPPLEMENTAL_DSO_PATH : undefined,
    locale,
    overlay_options: overlayOptions,
  };
}

function createRequestId() {
  return crypto.randomUUID();
}

function finalizeResponse(request: Request | undefined, response: Response, requestId?: string, cacheControl = "no-store") {
  return withCommonHeaders(response, {
    "Cache-Control": cacheControl,
    ...(requestId ? { "X-Request-Id": requestId } : {}),
  }, request, {
    allowedOrigins: CONFIG.corsAllowedOrigins,
  });
}

const PLATE_SOLVE_FAILURE_MARKERS = [
  "plate solving aborted",
  "plate solving failed",
];

function isPlateSolveFailure(message: string): boolean {
  const lower = message.toLowerCase();
  return PLATE_SOLVE_FAILURE_MARKERS.some((marker) => lower.includes(marker));
}

function errorResponse(error: unknown, requestId?: string, request?: Request) {
  if (error instanceof HttpError) {
    return finalizeResponse(request, jsonResponse({ error: error.message }, { status: error.status }), requestId);
  }
  const message = error instanceof Error ? error.message : String(error);
  if (isPlateSolveFailure(message)) {
    logInfo("plate-solve failed for request", requestId ? { requestId, message } : { message });
    return finalizeResponse(
      request,
      jsonResponse({ error: message, code: "plate_solve_failed" }, { status: 422 }),
      requestId,
    );
  }
  logError("request failed", error, requestId ? { requestId } : undefined);
  return finalizeResponse(request, jsonResponse({ error: "internal server error" }, { status: 500 }), requestId);
}

function createAbortError() {
  return new HttpError(499, "request aborted");
}

async function readWorkerLines(stream: ReadableStream<Uint8Array>, onLine: (line: string) => void) {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    let newlineIndex = buffer.indexOf("\n");
    while (newlineIndex >= 0) {
      const line = buffer.slice(0, newlineIndex).trim();
      buffer = buffer.slice(newlineIndex + 1);
      if (line) {
        onLine(line);
      }
      newlineIndex = buffer.indexOf("\n");
    }
  }
  const tail = buffer.trim();
  if (tail) {
    onLine(tail);
  }
}

function rejectPendingRequest(requestId: string, reason: unknown) {
  const pending = pendingWorkerRequests.get(requestId);
  if (!pending) {
    return;
  }
  pending.cleanup();
  pendingWorkerRequests.delete(requestId);
  pending.reject(reason);
}

function destroyWorker(reason: string) {
  if (!workerProcess) {
    workerReady = !CONFIG.preloadWorkerOnStartup;
    return;
  }

  logError("restarting annotation worker", reason);
  const processToStop = workerProcess;
  workerProcess = null;
  workerReady = !CONFIG.preloadWorkerOnStartup;

  try {
    processToStop.kill();
  } catch (error) {
    logError("failed to terminate worker", error);
  }
}

function ensureAnnotationWorker() {
  if (workerProcess) {
    return workerProcess;
  }

  const pythonBinary = resolvePythonBinary();
  const spawnedWorker = Bun.spawn({
    cmd: [pythonBinary, PYTHON_WORKER_SCRIPT],
    cwd: ROOT_DIR,
    stdin: "pipe",
    stdout: "pipe",
    stderr: "pipe",
  });
  workerProcess = spawnedWorker;

  const stdout = spawnedWorker.stdout;
  const stderr = spawnedWorker.stderr;
  if (!(stdout instanceof ReadableStream) || !(stderr instanceof ReadableStream)) {
    throw new Error("annotation worker stdio is not piped");
  }

  void readWorkerLines(stdout, (line) => {
    let payload: AnnotationWorkerResponse;
    try {
      payload = JSON.parse(line) as AnnotationWorkerResponse;
    } catch (error) {
      logError("worker stdout parse error", error);
      return;
    }

    if (!payload.id) {
      return;
    }

    const pending = pendingWorkerRequests.get(payload.id);
    if (!pending || pending.worker !== spawnedWorker) {
      return;
    }

    pending.cleanup();
    pendingWorkerRequests.delete(payload.id);

    if (payload.ok && payload.result) {
      workerReady = true;
      pending.resolve(payload.result);
      return;
    }

    pending.reject(new Error(payload.error || "annotation worker failed"));
  }).catch((error) => {
    logError("worker stdout reader failed", error);
  });

  void readWorkerLines(stderr, (line) => {
    logError("worker stderr", line);
  }).catch((error) => {
    logError("worker stderr reader failed", error);
  });

  void spawnedWorker.exited.then(() => {
    if (workerProcess === spawnedWorker) {
      workerProcess = null;
      workerReady = !CONFIG.preloadWorkerOnStartup;
    }
    const failure = new Error("annotation worker exited unexpectedly");
    for (const [requestId, pending] of pendingWorkerRequests.entries()) {
      if (pending.worker === spawnedWorker) {
        rejectPendingRequest(requestId, failure);
      }
    }
    queueWorkerWarmup();
  });

  return spawnedWorker;
}

async function sendWorkerRequest<T extends Record<string, unknown>>(
  payload: AnnotationWorkerRequest,
  timeoutMs = CONFIG.workerJobTimeoutMs,
  abortSignal?: AbortSignal,
) {
  const worker = ensureAnnotationWorker();
  let abortHandler: (() => void) | null = null;

  const resultPromise = new Promise<T>((resolve, reject) => {
    const timeout = setTimeout(() => {
      rejectPendingRequest(payload.id, new Error(`annotation worker timed out after ${timeoutMs} ms`));
      destroyWorker("timeout");
    }, timeoutMs);

    const cleanup = () => {
      clearTimeout(timeout);
      if (abortSignal && abortHandler) {
        abortSignal.removeEventListener("abort", abortHandler);
        abortHandler = null;
      }
    };

    pendingWorkerRequests.set(payload.id, {
      worker,
      resolve: resolve as (value: Record<string, unknown>) => void,
      reject,
      timeout,
      cleanup,
    });

    if (abortSignal) {
      abortHandler = () => {
        rejectPendingRequest(payload.id, createAbortError());
        destroyWorker("request aborted");
      };
      if (abortSignal.aborted) {
        abortHandler();
        return;
      }
      abortSignal.addEventListener("abort", abortHandler, { once: true });
    }
  });

  try {
    const stdin = worker.stdin;
    if (!stdin || typeof stdin === "number" || !("write" in stdin)) {
      throw new Error("annotation worker stdin is not writable");
    }
    stdin.write(`${JSON.stringify(payload)}\n`);
  } catch (error) {
    rejectPendingRequest(payload.id, error);
    destroyWorker("stdin write failed");
    throw error;
  }

  return resultPromise;
}

async function warmAnnotationWorker() {
  if (!CONFIG.preloadWorkerOnStartup) {
    workerReady = true;
    return;
  }

  const preloadRequest: AnnotationWorkerRequest = {
    id: createRequestId(),
    action: "preload",
    ...buildWorkerAssetPayload(DEFAULT_OVERLAY_OPTIONS, "en"),
  };

  await sendWorkerRequest<Record<string, unknown>>(preloadRequest, Math.max(CONFIG.workerJobTimeoutMs, 240_000));
  workerReady = true;
}

function queueWorkerWarmup() {
  if (shuttingDown || !CONFIG.preloadWorkerOnStartup || workerProcess || workerWarmupPromise) {
    return;
  }

  workerWarmupPromise = (async () => {
    try {
      await warmAnnotationWorker();
    } catch (error) {
      logError("background worker warmup failed", error);
    } finally {
      workerWarmupPromise = null;
    }
  })();
}

function runCommandCheck(command: string[], label: string) {
  const result = Bun.spawnSync({
    cmd: command,
    cwd: ROOT_DIR,
    stdout: "pipe",
    stderr: "pipe",
  });

  if (result.exitCode === 0) {
    return;
  }

  const stdout = new TextDecoder().decode(result.stdout).trim();
  const stderr = new TextDecoder().decode(result.stderr).trim();
  throw new Error(`${label} unavailable: ${stderr || stdout || `exit code ${result.exitCode}`}`);
}

function assertPathExists(targetPath: string, label: string) {
  if (!existsSync(targetPath)) {
    throw new Error(`${label} is missing: ${targetPath}`);
  }
}

async function validateRuntimePrerequisites() {
  assertPathExists(PYTHON_SCRIPT, "python annotator script");
  assertPathExists(PYTHON_WORKER_SCRIPT, "python worker script");
  assertPathExists(path.join(CATALOG_DIR, "minimal_hipparcos.csv"), "minimal Hipparcos catalog");
  assertPathExists(path.join(REFERENCE_DIR, "modern_st.json"), "constellation reference");
  assertPathExists(path.join(REFERENCE_DIR, "common_star_names.fab"), "star names reference");
  assertPathExists(path.join(REFERENCE_DIR, "NGC.csv"), "deep sky catalog");
  assertPathExists(STARDROID_ENGLISH_LOCALIZATION_PATH, "Stardroid English localization");
  assertPathExists(SUPPLEMENTAL_DSO_PATH, "supplemental deep sky objects reference");

  for (const index of REQUIRED_ASTROMETRY_INDEXES) {
    assertPathExists(path.join(ASTROMETRY_DIR, `index-${index}.fits`), `astrometry index ${index}`);
  }

  runCommandCheck([resolvePythonBinary(), "--version"], "python");
  runCommandCheck(["solve-field", "--help"], "solve-field");
}

async function withJobSlot<T>(callback: () => Promise<T>, abortSignal?: AbortSignal) {
  try {
    return await jobLimiter.run(callback, abortSignal);
  } catch (error) {
    if (error instanceof JobQueueFullError) {
      throw new HttpError(429, "server is busy, retry later");
    }
    if (error instanceof JobQueueAbortedError) {
      throw createAbortError();
    }
    throw error;
  }
}

async function withTemporaryDirectory<T>(prefix: string, callback: (directoryPath: string) => Promise<T>) {
  const directoryPath = await mkdtemp(path.join(tmpdir(), prefix));
  try {
    return await callback(directoryPath);
  } finally {
    await rm(directoryPath, { recursive: true, force: true });
  }
}

async function readFileAsBase64(filePath: string) {
  const buffer = await Bun.file(filePath).arrayBuffer();
  return Buffer.from(buffer).toString("base64");
}

function omitInternalPaths(result: RawAnnotationResult) {
  const { input_image: _inputImage, output_image: _outputImage, ...rest } = result;
  return rest;
}

async function runAnnotationViaWorker(
  inputImagePath: string,
  outputImagePath: string | undefined,
  overlayOptions: OverlayOptions,
  locale: string,
  abortSignal?: AbortSignal,
) {
  const request: AnnotationWorkerRequest = {
    id: createRequestId(),
    action: "annotate",
    input_path: inputImagePath,
    ...buildWorkerAssetPayload(overlayOptions, locale),
  };

  if (outputImagePath) {
    request.output_image_path = outputImagePath;
  }

  return sendWorkerRequest<RawAnnotationResult>(request, CONFIG.workerJobTimeoutMs, abortSignal);
}

async function runAnnotationViaCli(
  inputImagePath: string,
  outputImagePath: string | undefined,
  outputJsonPath: string,
  overlayOptions: OverlayOptions,
  locale: string,
  abortSignal?: AbortSignal,
) {
  const command = [
    resolvePythonBinary(),
    PYTHON_SCRIPT,
    "--input",
    inputImagePath,
    "--output-json",
    outputJsonPath,
    "--index-dir",
    ASTROMETRY_DIR,
    "--catalog",
    path.join(CATALOG_DIR, "minimal_hipparcos.csv"),
    "--constellations",
    path.join(REFERENCE_DIR, "modern_st.json"),
    "--star-names",
    path.join(REFERENCE_DIR, "common_star_names.fab"),
    "--dso-catalog",
    path.join(REFERENCE_DIR, "NGC.csv"),
    "--locale",
    locale,
  ];

  if (existsSync(STARDROID_CONSTELLATIONS_PATH)) {
    command.push("--constellations", STARDROID_CONSTELLATIONS_PATH);
  }

  if (existsSync(STARDROID_DSO_PATH)) {
    command.push("--dso-catalog", STARDROID_DSO_PATH);
  }

  for (const localizationPath of listLocalizationPaths()) {
    command.push("--localization", localizationPath);
  }

  if (existsSync(SUPPLEMENTAL_DSO_PATH)) {
    command.push("--supplemental-dso", SUPPLEMENTAL_DSO_PATH);
  }

  if (outputImagePath) {
    command.push("--output-image", outputImagePath);
  }

  command.push("--options-json", JSON.stringify(overlayOptions));

  const proc = Bun.spawn({
    cmd: command,
    cwd: ROOT_DIR,
    stdout: "pipe",
    stderr: "pipe",
  });

  const completionPromise = Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
    proc.exited,
  ]);

  const [stdout, stderr, exitCode] = await new Promise<[string, string, number]>((resolve, reject) => {
    let settled = false;
    const cleanup = () => {
      if (abortSignal && abortHandler) {
        abortSignal.removeEventListener("abort", abortHandler);
      }
    };
    const finishResolve = (value: [string, string, number]) => {
      if (settled) {
        return;
      }
      settled = true;
      cleanup();
      resolve(value);
    };
    const finishReject = (error: unknown) => {
      if (settled) {
        return;
      }
      settled = true;
      cleanup();
      reject(error);
    };
    const abortHandler = () => {
      try {
        proc.kill();
      } catch {
        // Ignore kill failures during request cancellation.
      }
      finishReject(createAbortError());
    };

    if (abortSignal) {
      if (abortSignal.aborted) {
        abortHandler();
        return;
      }
      abortSignal.addEventListener("abort", abortHandler, { once: true });
    }

    completionPromise.then(finishResolve).catch(finishReject);
  });

  if (exitCode !== 0) {
    throw new Error(stderr.trim() || stdout.trim() || "annotation failed");
  }

  return Bun.file(outputJsonPath).json() as Promise<RawAnnotationResult>;
}

type RunAnnotationOptions = {
  overlayOptions?: OverlayOptions;
  renderMode?: RenderMode;
  locale?: string;
  workspaceDir: string;
  abortSignal?: AbortSignal;
};

async function runAnnotation(
  inputImagePath: string,
  {
    overlayOptions = cloneOverlayOptions(),
    renderMode = DEFAULT_RENDER_MODE,
    locale = "en",
    workspaceDir,
    abortSignal,
  }: RunAnnotationOptions,
) {
  const runId = createRequestId();
  const outputImagePath = shouldRenderOnServer(renderMode)
    ? path.join(workspaceDir, `${runId}.png`)
    : undefined;
  const outputJsonPath = path.join(workspaceDir, `${runId}.json`);
  const startedAt = performance.now();

  try {
    const result = await runAnnotationViaWorker(inputImagePath, outputImagePath, overlayOptions, locale, abortSignal);
    const annotatedImageBase64 = outputImagePath ? await readFileAsBase64(outputImagePath) : undefined;
    const sanitizedResult = omitInternalPaths(result);
    return {
      ...sanitizedResult,
      render_options: overlayOptions,
      render_mode: renderMode,
      available_renders: {
        server: Boolean(annotatedImageBase64),
        client: true,
        default_view: annotatedImageBase64 ? "server" : "client",
      },
      inputImageUrl: resolvePublicImageUrl(inputImagePath) ?? null,
      annotatedImageBase64: annotatedImageBase64 ?? null,
      annotatedImageMimeType: annotatedImageBase64 ? "image/png" : null,
      processingMs: Math.round(performance.now() - startedAt),
    } satisfies AnnotationApiResponse;
  } catch (error) {
    if (!CONFIG.allowCliFallback || error instanceof HttpError || isDeterministicPipelineError(error)) {
      // Deterministic pipeline failures (e.g. plate-solve timeout) will reproduce
      // in the CLI fallback — retrying just doubles the client-visible latency.
      throw error;
    }
    logError("worker fallback to CLI", error);
    const result = await runAnnotationViaCli(inputImagePath, outputImagePath, outputJsonPath, overlayOptions, locale, abortSignal);
    const annotatedImageBase64 = outputImagePath ? await readFileAsBase64(outputImagePath) : undefined;
    const sanitizedResult = omitInternalPaths(result);
    return {
      ...sanitizedResult,
      render_options: overlayOptions,
      render_mode: renderMode,
      available_renders: {
        server: Boolean(annotatedImageBase64),
        client: true,
        default_view: annotatedImageBase64 ? "server" : "client",
      },
      inputImageUrl: resolvePublicImageUrl(inputImagePath) ?? null,
      annotatedImageBase64: annotatedImageBase64 ?? null,
      annotatedImageMimeType: annotatedImageBase64 ? "image/png" : null,
      processingMs: Math.round(performance.now() - startedAt),
    } satisfies AnnotationApiResponse;
  }
}

async function handleAnalyzeUpload(request: Request) {
  const requestId = createRequestId();

  try {
    const formData = await request.formData();
    const file = formData.get("image");
    const overlayOptions = parseOverlayOptionsFromFormData(formData);
    const renderMode = parseRenderModeFromFormData(formData);
    const locale = parseLocaleFromFormData(formData, request.headers.get("accept-language"));

    if (!(file instanceof File)) {
      return errorResponse(new HttpError(400, "missing file field 'image'"), requestId, request);
    }

    const { extension } = validateImageUpload(file, CONFIG.maxUploadBytes);
    const result = await withTemporaryDirectory("star-upload-", async (workspaceDir) => {
      const inputImagePath = path.join(workspaceDir, `${requestId}${guessExtension(file.name, file.type || undefined) || extension}`);
      await Bun.write(inputImagePath, file);

      logInfo("upload accepted", {
        requestId,
        filename: file.name,
        size: file.size,
        renderMode,
        locale,
      });

      return withJobSlot(() => runAnnotation(inputImagePath, {
        overlayOptions,
        renderMode,
        locale,
        workspaceDir,
        abortSignal: request.signal,
      }), request.signal);
    });
    logInfo("upload completed", {
      requestId,
      processingMs: result.processingMs,
    });
    return finalizeResponse(request, jsonResponse(result), requestId);
  } catch (error) {
    logError("upload failed", error, { requestId });
    return errorResponse(error, requestId, request);
  }
}

async function handleAnalyzeSample(request: Request) {
  const requestId = createRequestId();
  let body: { id?: string; options?: unknown; render_mode?: unknown; locale?: unknown } | null = null;

  try {
    body = await request.json() as { id?: string; options?: unknown; render_mode?: unknown; locale?: unknown };
  } catch {
    return errorResponse(new HttpError(400, "invalid JSON body"), requestId, request);
  }

  const sample = SAMPLE_IMAGES.find((entry) => entry.id === body?.id);
  if (!sample) {
    return errorResponse(new HttpError(400, "unknown sample id"), requestId, request);
  }

  try {
    const overlayOptions = normalizeOverlayOptions(body?.options ?? undefined);
    const renderMode = normalizeRenderMode(body?.render_mode);
    const locale = normalizeLocaleTag(body?.locale) || parsePrimaryAcceptLanguage(request.headers.get("accept-language")) || "en";
    logInfo("sample accepted", { requestId, sampleId: sample.id, renderMode, locale });
    const result = await withTemporaryDirectory(
      `star-sample-${sample.id}-`,
      (workspaceDir) => withJobSlot(() => runAnnotation(path.join(SAMPLES_DIR, sample.filename), {
        overlayOptions,
        renderMode,
        locale,
        workspaceDir,
        abortSignal: request.signal,
      }), request.signal),
    );
    logInfo("sample completed", { requestId, sampleId: sample.id, processingMs: result.processingMs });
    return finalizeResponse(request, jsonResponse(result), requestId);
  } catch (error) {
    logError("sample analyze failed", error, { requestId, sampleId: sample.id });
    return errorResponse(error, requestId, request);
  }
}

function serveStaticFile(request: Request, baseDir: string, relativePath: string, cacheControl = "private, max-age=3600") {
  const absoluteBaseDir = path.resolve(baseDir);
  const absolutePath = path.resolve(baseDir, relativePath);

  if (
    absolutePath !== absoluteBaseDir &&
    !absolutePath.startsWith(`${absoluteBaseDir}${path.sep}`)
  ) {
    return finalizeResponse(request, new Response("Not Found", { status: 404 }), undefined, "no-store");
  }

  if (!existsSync(absolutePath) || !statSync(absolutePath).isFile()) {
    return finalizeResponse(request, new Response("Not Found", { status: 404 }), undefined, "no-store");
  }

  return finalizeResponse(request, new Response(Bun.file(absolutePath)), undefined, cacheControl);
}

function healthPayload() {
  const limiterStats = jobLimiter.stats();
  return {
    ok: true,
    uptimeMs: Math.round(process.uptime() * 1000),
    activeJobs: limiterStats.activeJobs,
    queuedJobs: limiterStats.queuedJobs,
    workerReady,
    pendingWorkerRequests: pendingWorkerRequests.size,
    config: {
      maxUploadBytes: CONFIG.maxUploadBytes,
      maxConcurrentJobs: CONFIG.maxConcurrentJobs,
      maxQueuedJobs: CONFIG.maxQueuedJobs,
      allowCliFallback: CONFIG.allowCliFallback,
    },
  };
}

export async function startServer() {
  await validateRuntimePrerequisites();
  await warmAnnotationWorker();

  const server = Bun.serve({
    hostname: "0.0.0.0",
    port: CONFIG.port,
    idleTimeout: CONFIG.idleTimeoutSeconds,
    maxRequestBodySize: CONFIG.maxRequestBodySizeBytes,
    development: process.env.NODE_ENV !== "production",
    error(error) {
      logError("uncaught fetch handler error", error);
      return finalizeResponse(undefined, jsonResponse({ error: "internal server error" }, { status: 500 }));
    },
    async fetch(request) {
      try {
        const url = new URL(request.url);

        if (isCorsPreflightRequest(request)) {
          return finalizeResponse(request, corsPreflightResponse(request, {
            allowedOrigins: CONFIG.corsAllowedOrigins,
          }), undefined, "no-store");
        }

        if (request.method === "GET" && url.pathname === "/healthz") {
          return finalizeResponse(request, jsonResponse(healthPayload()));
        }

        if (request.method === "GET" && url.pathname === "/readyz") {
          if (!workerReady) {
            return finalizeResponse(request, jsonResponse({ ok: false, error: "worker not ready" }, { status: 503 }));
          }
          return finalizeResponse(request, jsonResponse(healthPayload()));
        }

        if (request.method === "GET" && url.pathname === "/") {
          return finalizeResponse(request, new Response(Bun.file(path.join(PUBLIC_DIR, "index.html"))));
        }

        if (request.method === "GET" && url.pathname === "/app.js") {
          return finalizeResponse(request, new Response(Bun.file(path.join(PUBLIC_DIR, "app.js"))));
        }

        if (request.method === "GET" && url.pathname === "/api/samples") {
          return finalizeResponse(request, jsonResponse(SAMPLE_IMAGES));
        }

        if (request.method === "GET" && url.pathname === "/api/overlay-options") {
          return finalizeResponse(request, jsonResponse({
            defaults: cloneOverlayOptions(),
            presets: OVERLAY_PRESETS,
            localization: {
              default_locale: "en",
              available_locales: listAvailableLocales(),
            },
          }));
        }

        if (request.method === "POST" && url.pathname === "/api/analyze") {
          return handleAnalyzeUpload(request);
        }

        if (request.method === "POST" && url.pathname === "/api/analyze-sample") {
          return handleAnalyzeSample(request);
        }

        if (request.method === "GET" && url.pathname.startsWith("/samples/")) {
          return serveStaticFile(request, SAMPLES_DIR, url.pathname.replace("/samples/", ""), "public, max-age=86400");
        }

        return finalizeResponse(request, new Response("Not Found", { status: 404 }), undefined, "no-store");
      } catch (error) {
        logError("fetch handler failed", error);
        return errorResponse(error, undefined, request);
      }
    },
  });

  logInfo(`Star annotator running at http://localhost:${server.port}`);
  return server;
}

async function shutdown(server: Bun.Server<unknown>, signal: string) {
  if (shuttingDown) {
    return;
  }
  shuttingDown = true;

  logInfo("shutting down", { signal });

  destroyWorker(signal);
  await server.stop(true);
}

if (import.meta.main) {
  try {
    const server = await startServer();
    for (const signal of ["SIGINT", "SIGTERM"] as const) {
      process.on(signal, () => {
        void shutdown(server, signal).finally(() => {
          process.exit(0);
        });
      });
    }
  } catch (error) {
    logError("startup failed", error);
    process.exit(1);
  }
}
