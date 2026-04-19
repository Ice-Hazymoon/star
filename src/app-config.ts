import path from "node:path";

export const ROOT_DIR = path.resolve(import.meta.dir, "..");
export const DATA_DIR = path.join(ROOT_DIR, "data");
export const ASTROMETRY_DIR = path.join(DATA_DIR, "astrometry");
export const CATALOG_DIR = path.join(DATA_DIR, "catalog");
export const REFERENCE_DIR = path.join(DATA_DIR, "reference");
export const SAMPLES_DIR = path.join(ROOT_DIR, "samples");
export const PUBLIC_DIR = path.join(ROOT_DIR, "public");
export const PYTHON_SCRIPT = path.join(ROOT_DIR, "python", "annotate.py");
export const PYTHON_WORKER_SCRIPT = path.join(ROOT_DIR, "python", "annotate_worker.py");
export const STARDROID_CONSTELLATIONS_PATH = path.join(REFERENCE_DIR, "stardroid-constellations.ascii");
export const STARDROID_DSO_PATH = path.join(REFERENCE_DIR, "stardroid-deep_sky_objects.csv");
export const STARDROID_LOCALES_DIR = path.join(REFERENCE_DIR, "stardroid-locales");
export const STARDROID_ENGLISH_LOCALIZATION_PATH = path.join(STARDROID_LOCALES_DIR, "values", "celestial_objects.xml");
export const SUPPLEMENTAL_DSO_PATH = path.join(REFERENCE_DIR, "supplemental-deep-sky-objects.json");
export const PYTHON_BIN_CANDIDATES = [
  path.join(ROOT_DIR, ".venv", "bin", "python"),
  path.join(ROOT_DIR, ".venv", "bin", "python3"),
  "python3",
  "python"
];

export const REQUIRED_ASTROMETRY_INDEXES = Array.from(
  { length: 13 },
  (_, offset) => 4107 + offset,
);

export const SAMPLE_IMAGES = [
  {
    id: "apod4",
    title: "APOD Big Dipper",
    filename: "apod4.jpg",
    url: "https://raw.githubusercontent.com/dstndstn/astrometry.net/master/demo/apod4.jpg",
    note: "34x24 degree field, suitable for testing the Big Dipper / Ursa Major overlay."
  },
  {
    id: "orion-over-pines",
    title: "Orion Over Pine Trees",
    filename: "orion-over-pines.jpg",
    url: "https://upload.wikimedia.org/wikipedia/commons/6/69/Orion%27s_wide_field_over_pine_trees.jpg",
    note: "Earth-view nightscape with foreground trees, useful for testing sky-only crop solving."
  },
  {
    id: "apod5",
    title: "APOD Wide Winter Sky",
    filename: "apod5.jpg",
    url: "https://raw.githubusercontent.com/dstndstn/astrometry.net/master/demo/apod5.jpg",
    note: "Very wide winter-sky stress sample for the plate-solving pipeline."
  }
] as const;

export const REFERENCE_ASSETS = [
  {
    filename: "modern_st.json",
    url: "https://raw.githubusercontent.com/Stellarium/stellarium/master/skycultures/modern_st/index.json",
    directory: REFERENCE_DIR,
  },
  {
    filename: "common_star_names.fab",
    url: "https://raw.githubusercontent.com/Stellarium/stellarium/master/skycultures/common_star_names.fab",
    directory: REFERENCE_DIR,
  },
  {
    filename: "NGC.csv",
    url: "https://raw.githubusercontent.com/dstndstn/astrometry.net/master/catalogs/NGC.csv",
    directory: REFERENCE_DIR,
  },
];
