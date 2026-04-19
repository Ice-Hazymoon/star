import { existsSync } from "node:fs";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import {
  ASTROMETRY_DIR,
  CATALOG_DIR,
  PYTHON_BIN_CANDIDATES,
  PYTHON_SCRIPT,
  REFERENCE_DIR,
  ROOT_DIR,
  SAMPLE_IMAGES,
  SAMPLES_DIR,
  STARDROID_CONSTELLATIONS_PATH,
  STARDROID_DSO_PATH,
} from "./app-config";

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

const sampleId = process.argv[2] ?? "apod4";
const optionsJson = process.argv[3];
const sample = SAMPLE_IMAGES.find((entry) => entry.id === sampleId);

if (!sample) {
  console.error(`unknown sample: ${sampleId}`);
  process.exit(1);
}

const runId = `${sample.id}-${Date.now()}`;
const workspaceDir = await mkdtemp(path.join(tmpdir(), "star-run-sample-"));

try {
  const outputImagePath = path.join(workspaceDir, `${runId}.png`);
  const outputJsonPath = path.join(workspaceDir, `${runId}.json`);

  const command = [
    resolvePythonBinary(),
    PYTHON_SCRIPT,
    "--input",
    path.join(SAMPLES_DIR, sample.filename),
    "--output-image",
    outputImagePath,
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
  ];

  if (existsSync(STARDROID_CONSTELLATIONS_PATH)) {
    command.push("--constellations", STARDROID_CONSTELLATIONS_PATH);
  }

  if (existsSync(STARDROID_DSO_PATH)) {
    command.push("--dso-catalog", STARDROID_DSO_PATH);
  }

  if (optionsJson) {
    command.push("--options-json", optionsJson);
  }

  const proc = Bun.spawn({
    cmd: command,
    cwd: ROOT_DIR,
    stdout: "pipe",
    stderr: "pipe",
  });

  const [stdout, stderr, exitCode] = await Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
    proc.exited,
  ]);

  if (stderr.trim()) {
    console.error(stderr.trim());
  }

  if (exitCode !== 0) {
    process.exit(exitCode);
  }

  const result = await Bun.file(outputJsonPath).json();
  const outputImageBase64 = Buffer.from(await Bun.file(outputImagePath).arrayBuffer()).toString("base64");
  console.log(JSON.stringify({
    sample: sample.id,
    annotatedImageMimeType: "image/png",
    annotatedImageBase64Length: outputImageBase64.length,
    annotatedImageBase64Preview: outputImageBase64.slice(0, 120),
    summary: result,
  }, null, 2));
} finally {
  await rm(workspaceDir, { recursive: true, force: true });
}
