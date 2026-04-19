import { mkdirSync, existsSync } from "node:fs";
import path from "node:path";
import {
  ASTROMETRY_DIR,
  CATALOG_DIR,
  REFERENCE_ASSETS,
  REFERENCE_DIR,
  REQUIRED_ASTROMETRY_INDEXES,
  SAMPLES_DIR,
  SAMPLE_IMAGES,
} from "./app-config";

async function downloadIfMissing(url: string, destination: string) {
  if (existsSync(destination)) {
    console.log(`skip ${path.basename(destination)}`);
    return;
  }

  console.log(`download ${path.basename(destination)} <- ${url}`);
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`failed to download ${url}: ${response.status} ${response.statusText}`);
  }

  await Bun.write(destination, response);
}

async function main() {
  [
    ASTROMETRY_DIR,
    CATALOG_DIR,
    REFERENCE_DIR,
    SAMPLES_DIR,
  ].forEach((directory) => mkdirSync(directory, { recursive: true }));

  for (const index of REQUIRED_ASTROMETRY_INDEXES) {
    const filename = `index-${index}.fits`;
    await downloadIfMissing(
      `http://data.astrometry.net/4100/${filename}`,
      path.join(ASTROMETRY_DIR, filename),
    );
  }

  for (const asset of REFERENCE_ASSETS) {
    await downloadIfMissing(asset.url, path.join(asset.directory, asset.filename));
  }

  for (const sample of SAMPLE_IMAGES) {
    await downloadIfMissing(sample.url, path.join(SAMPLES_DIR, sample.filename));
  }

  console.log("data bootstrap complete");
}

await main();
