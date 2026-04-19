import { describe, expect, test } from "bun:test";
import { cloneOverlayOptions, normalizeOverlayOptions } from "./overlay-options";

describe("overlay options", () => {
  test("clone returns an isolated copy", () => {
    const options = cloneOverlayOptions();
    options.layers.star_labels = false;

    expect(cloneOverlayOptions().layers.star_labels).toBe(true);
  });

  test("preset values are applied before overrides", () => {
    const options = normalizeOverlayOptions({
      preset: "balanced",
      detail: {
        star_label_limit: 22,
      },
    });

    expect(options.preset).toBe("balanced");
    expect(options.detail.star_label_limit).toBe(22);
    expect(options.detail.include_catalog_dsos).toBe(false);
  });

  test("numeric values are clamped", () => {
    const options = normalizeOverlayOptions({
      detail: {
        star_label_limit: 999,
        dso_spacing_scale: -1,
      },
    });

    expect(options.detail.star_label_limit).toBe(80);
    expect(options.detail.dso_spacing_scale).toBe(0.1);
  });
});
