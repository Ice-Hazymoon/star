import type { OverlayOptions } from "./overlay-options";
import type { RenderMode } from "./render-mode";

export type CropDescriptor = {
  name: string;
  x: number;
  y: number;
  width: number;
  height: number;
};

export type SolveSummary = {
  center_ra_deg: number;
  center_dec_deg: number;
  field_width_deg: number;
  field_height_deg: number;
  crop: CropDescriptor | null;
};

export type VisibleNamedStar = {
  hip: number;
  name: string;
  magnitude: number;
  x: number;
  y: number;
};

export type ConstellationSegmentEndpoint = {
  x: number;
  y: number;
  hip?: number;
};

export type ConstellationSegment = {
  start: ConstellationSegmentEndpoint;
  end: ConstellationSegmentEndpoint;
};

export type VisibleConstellation = {
  abbr: string;
  english_name: string;
  native_name: string;
  display_name: string;
  label_x: number;
  label_y: number;
  segments: ConstellationSegment[];
  show_label: boolean;
};

export type VisibleDeepSkyObject = {
  name: string;
  type: string;
  const: string;
  ra_degrees: number;
  dec_degrees: number;
  major_axis_arcmin: number | null;
  magnitude: number | null;
  messier: string | null;
  catalog_id: string | null;
  common_name: string | null;
  common_names: string[];
  label: string;
  curated: boolean;
  x: number;
  y: number;
  display_label: string;
};

export type RgbaTuple = [number, number, number, number];

export type OverlaySceneLine = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  line_width: number;
  rgba: RgbaTuple;
};

export type OverlaySceneMarker = {
  x: number;
  y: number;
  radius: number;
  line_width?: number;
  rgba?: RgbaTuple;
  fill_rgba?: RgbaTuple;
  outline_rgba?: RgbaTuple;
  marker?: string;
};

export type OverlaySceneLeader = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  line_width: number;
  rgba: RgbaTuple;
};

export type OverlaySceneLabel = {
  text: string;
  x: number;
  y: number;
  font_size: number;
  stroke_width: number;
  text_rgba: RgbaTuple;
  stroke_rgba: RgbaTuple;
  leader?: OverlaySceneLeader | null;
};

export type OverlayScene = {
  image_width: number;
  image_height: number;
  crop: CropDescriptor;
  bounds: {
    left: number;
    top: number;
    right: number;
    bottom: number;
  };
  constellation_lines: OverlaySceneLine[];
  constellation_labels: OverlaySceneLabel[];
  deep_sky_markers: OverlaySceneMarker[];
  deep_sky_labels: OverlaySceneLabel[];
  star_markers: OverlaySceneMarker[];
  star_labels: OverlaySceneLabel[];
};

export type RawAnnotationResult = {
  input_image: string;
  output_image: string | null;
  image_width: number;
  image_height: number;
  solve: SolveSummary;
  solve_verification: Record<string, unknown>;
  attempts: Array<Record<string, unknown>>;
  source_analysis: Record<string, unknown>;
  localization: {
    requested_locale: string;
    resolved_locale: string;
    available_locales: string[];
  };
  visible_named_stars: VisibleNamedStar[];
  visible_constellations: VisibleConstellation[];
  visible_deep_sky_objects: VisibleDeepSkyObject[];
  render_options: OverlayOptions;
  overlay_scene: OverlayScene;
  solver_log_tail: string;
  timings_ms: Record<string, number>;
};

export type AnnotationApiResponse = Omit<RawAnnotationResult, "input_image" | "output_image"> & {
  render_mode: RenderMode;
  available_renders: {
    server: boolean;
    client: boolean;
    default_view: RenderMode;
  };
  inputImageUrl: string | null;
  annotatedImageBase64: string | null;
  annotatedImageMimeType: string | null;
  processingMs: number;
};
