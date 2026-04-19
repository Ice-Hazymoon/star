export type OverlayLayers = {
  constellation_lines: boolean;
  constellation_labels: boolean;
  contextual_constellation_labels: boolean;
  star_markers: boolean;
  star_labels: boolean;
  deep_sky_markers: boolean;
  deep_sky_labels: boolean;
  label_leaders: boolean;
};

export type OverlayDetail = {
  star_label_limit: number;
  star_magnitude_limit: number;
  star_bright_separation: number;
  star_dim_separation: number;
  dso_label_limit: number;
  dso_magnitude_limit: number;
  dso_spacing_scale: number;
  show_all_constellation_labels: boolean;
  detailed_dso_labels: boolean;
  include_catalog_dsos: boolean;
};

export type OverlayOptions = {
  preset: "balanced" | "detailed" | "max";
  layers: OverlayLayers;
  detail: OverlayDetail;
};

export const DEFAULT_OVERLAY_OPTIONS: OverlayOptions = {
  preset: "max",
  layers: {
    constellation_lines: true,
    constellation_labels: true,
    contextual_constellation_labels: true,
    star_markers: true,
    star_labels: true,
    deep_sky_markers: true,
    deep_sky_labels: true,
    label_leaders: true,
  },
  detail: {
    star_label_limit: 36,
    star_magnitude_limit: 4.8,
    star_bright_separation: 82,
    star_dim_separation: 60,
    dso_label_limit: 48,
    dso_magnitude_limit: 13,
    dso_spacing_scale: 0.58,
    show_all_constellation_labels: true,
    detailed_dso_labels: true,
    include_catalog_dsos: true,
  },
};

export const OVERLAY_PRESETS: Record<OverlayOptions["preset"], Pick<OverlayOptions, "detail">> = {
  balanced: {
    detail: {
      star_label_limit: 18,
      star_magnitude_limit: 3.8,
      star_bright_separation: 105,
      star_dim_separation: 82,
      dso_label_limit: 24,
      dso_magnitude_limit: 11,
      dso_spacing_scale: 0.7,
      show_all_constellation_labels: false,
      detailed_dso_labels: false,
      include_catalog_dsos: false,
    },
  },
  detailed: {
    detail: {
      star_label_limit: 28,
      star_magnitude_limit: 4.4,
      star_bright_separation: 92,
      star_dim_separation: 72,
      dso_label_limit: 36,
      dso_magnitude_limit: 12.2,
      dso_spacing_scale: 0.64,
      show_all_constellation_labels: false,
      detailed_dso_labels: true,
      include_catalog_dsos: true,
    },
  },
  max: {
    detail: {
      star_label_limit: 36,
      star_magnitude_limit: 4.8,
      star_bright_separation: 82,
      star_dim_separation: 60,
      dso_label_limit: 48,
      dso_magnitude_limit: 13,
      dso_spacing_scale: 0.58,
      show_all_constellation_labels: true,
      detailed_dso_labels: true,
      include_catalog_dsos: true,
    },
  },
};

function deepMerge(target: Record<string, unknown>, source: Record<string, unknown>) {
  for (const [key, value] of Object.entries(source)) {
    if (
      value &&
      typeof value === "object" &&
      !Array.isArray(value) &&
      target[key] &&
      typeof target[key] === "object"
    ) {
      deepMerge(target[key] as Record<string, unknown>, value as Record<string, unknown>);
      continue;
    }
    target[key] = value;
  }
  return target;
}

function clampNumber(value: unknown, fallback: number, minimum: number, maximum: number) {
  const numeric = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(numeric)) {
    return fallback;
  }
  return Math.max(minimum, Math.min(maximum, numeric));
}

export function cloneOverlayOptions() {
  return structuredClone(DEFAULT_OVERLAY_OPTIONS);
}

export function normalizeOverlayOptions(input: unknown): OverlayOptions {
  const options = cloneOverlayOptions();
  const payload = input && typeof input === "object" && !Array.isArray(input)
    ? input as Partial<OverlayOptions>
    : {};

  const preset = typeof payload.preset === "string" && payload.preset in OVERLAY_PRESETS
    ? payload.preset as OverlayOptions["preset"]
    : options.preset;

  options.preset = preset;
  deepMerge(
    options as unknown as Record<string, unknown>,
    structuredClone(OVERLAY_PRESETS[preset]) as Record<string, unknown>,
  );
  const { preset: _ignoredPreset, ...payloadWithoutPreset } = payload;
  deepMerge(options as unknown as Record<string, unknown>, payloadWithoutPreset as Record<string, unknown>);

  options.detail.star_label_limit = Math.round(clampNumber(options.detail.star_label_limit, 36, 0, 80));
  options.detail.star_magnitude_limit = clampNumber(options.detail.star_magnitude_limit, 4.8, 0, 8);
  options.detail.star_bright_separation = clampNumber(options.detail.star_bright_separation, 82, 20, 180);
  options.detail.star_dim_separation = clampNumber(options.detail.star_dim_separation, 60, 12, 150);
  options.detail.dso_label_limit = Math.round(clampNumber(options.detail.dso_label_limit, 48, 0, 120));
  options.detail.dso_magnitude_limit = clampNumber(options.detail.dso_magnitude_limit, 13, 0, 20);
  options.detail.dso_spacing_scale = clampNumber(options.detail.dso_spacing_scale, 0.58, 0.1, 1.5);
  options.detail.show_all_constellation_labels = Boolean(options.detail.show_all_constellation_labels);
  options.detail.detailed_dso_labels = Boolean(options.detail.detailed_dso_labels);
  options.detail.include_catalog_dsos = Boolean(options.detail.include_catalog_dsos);

  for (const key of Object.keys(options.layers) as Array<keyof OverlayLayers>) {
    options.layers[key] = Boolean(options.layers[key]);
  }

  return options;
}
