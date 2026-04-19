const uploadForm = document.getElementById("upload-form");
const optionsForm = document.getElementById("options-form");
const presetSelect = document.getElementById("preset");
const requestRenderModeSelect = document.getElementById("request-render-mode");
const requestLocaleSelect = document.getElementById("request-locale");
const viewerModeSelect = document.getElementById("viewer-mode");
const samples = document.getElementById("samples");
const status = document.getElementById("status");
const viewer = document.getElementById("viewer");
const resultImage = document.getElementById("result-image");
const resultCanvas = document.getElementById("result-canvas");
const resultJson = document.getElementById("result-json");

if (viewer instanceof HTMLDivElement) {
  viewer.style.position = "relative";
  viewer.style.display = "inline-block";
}
if (resultImage instanceof HTMLImageElement) {
  resultImage.style.display = "block";
}
if (resultCanvas instanceof HTMLCanvasElement) {
  resultCanvas.style.position = "absolute";
  resultCanvas.style.left = "0";
  resultCanvas.style.top = "0";
  resultCanvas.style.pointerEvents = "none";
  resultCanvas.hidden = true;
}

let overlayDefaults = null;
let overlayPresets = null;
let availableLocales = [];
let currentResult = null;
let currentSourceImageUrl = "";
let currentServerImageUrl = "";
let currentUploadObjectUrl = null;
let imageLoadToken = 0;

const layerIds = {
  constellation_lines: "layer-constellation-lines",
  constellation_labels: "layer-constellation-labels",
  contextual_constellation_labels: "layer-contextual-constellation-labels",
  star_markers: "layer-star-markers",
  star_labels: "layer-star-labels",
  deep_sky_markers: "layer-deep-sky-markers",
  deep_sky_labels: "layer-deep-sky-labels",
  label_leaders: "layer-label-leaders",
};

const detailIds = {
  star_label_limit: "star-label-limit",
  star_magnitude_limit: "star-magnitude-limit",
  star_bright_separation: "star-bright-separation",
  star_dim_separation: "star-dim-separation",
  dso_label_limit: "dso-label-limit",
  dso_magnitude_limit: "dso-magnitude-limit",
  dso_spacing_scale: "dso-spacing-scale",
  show_all_constellation_labels: "show-all-constellation-labels",
  detailed_dso_labels: "detailed-dso-labels",
  include_catalog_dsos: "include-catalog-dsos",
};

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function mergeNested(target, source) {
  Object.entries(source).forEach(([key, value]) => {
    if (value && typeof value === "object" && !Array.isArray(value) && target[key] && typeof target[key] === "object") {
      mergeNested(target[key], value);
      return;
    }
    target[key] = value;
  });
  return target;
}

function checkboxValue(id) {
  const element = document.getElementById(id);
  return element instanceof HTMLInputElement ? element.checked : false;
}

function numberValue(id) {
  const element = document.getElementById(id);
  return element instanceof HTMLInputElement ? Number(element.value) : 0;
}

function applyOptions(options) {
  if (presetSelect instanceof HTMLSelectElement) {
    presetSelect.value = options.preset;
  }

  Object.entries(layerIds).forEach(([key, id]) => {
    const element = document.getElementById(id);
    if (element instanceof HTMLInputElement) {
      element.checked = Boolean(options.layers[key]);
    }
  });

  Object.entries(detailIds).forEach(([key, id]) => {
    const element = document.getElementById(id);
    if (!(element instanceof HTMLInputElement)) {
      return;
    }
    if (element.type === "checkbox") {
      element.checked = Boolean(options.detail[key]);
    } else {
      element.value = String(options.detail[key]);
    }
  });
}

function currentOptions() {
  return {
    preset: presetSelect instanceof HTMLSelectElement ? presetSelect.value : "max",
    layers: Object.fromEntries(
      Object.entries(layerIds).map(([key, id]) => [key, checkboxValue(id)]),
    ),
    detail: {
      star_label_limit: numberValue(detailIds.star_label_limit),
      star_magnitude_limit: numberValue(detailIds.star_magnitude_limit),
      star_bright_separation: numberValue(detailIds.star_bright_separation),
      star_dim_separation: numberValue(detailIds.star_dim_separation),
      dso_label_limit: numberValue(detailIds.dso_label_limit),
      dso_magnitude_limit: numberValue(detailIds.dso_magnitude_limit),
      dso_spacing_scale: numberValue(detailIds.dso_spacing_scale),
      show_all_constellation_labels: checkboxValue(detailIds.show_all_constellation_labels),
      detailed_dso_labels: checkboxValue(detailIds.detailed_dso_labels),
      include_catalog_dsos: checkboxValue(detailIds.include_catalog_dsos),
    },
  };
}

function currentRequestRenderMode() {
  return requestRenderModeSelect instanceof HTMLSelectElement
    ? requestRenderModeSelect.value
    : "server";
}

function currentRequestLocale() {
  if (requestLocaleSelect instanceof HTMLSelectElement && requestLocaleSelect.value) {
    return requestLocaleSelect.value;
  }
  return navigator.language || "en";
}

function pickPreferredLocale(preferredLocale, locales) {
  if (!preferredLocale || !Array.isArray(locales) || locales.length === 0) {
    return "en";
  }

  const normalized = preferredLocale.replaceAll("_", "-").toLowerCase();
  const exactMatch = locales.find((locale) => locale.toLowerCase() === normalized);
  if (exactMatch) {
    return exactMatch;
  }

  if (normalized.startsWith("zh-")) {
    if (normalized.includes("-tw") || normalized.includes("-hk") || normalized.includes("-mo") || normalized.includes("-hant")) {
      return locales.find((locale) => locale === "zh-Hant") || "en";
    }
    if (normalized.includes("-cn") || normalized.includes("-sg") || normalized.includes("-my") || normalized.includes("-hans")) {
      return locales.find((locale) => locale === "zh-Hans") || "en";
    }
  }

  const requestedLanguage = normalized.split("-")[0];
  return locales.find((locale) => locale.split("-")[0].toLowerCase() === requestedLanguage) || "en";
}

function clearCanvas() {
  if (!(resultCanvas instanceof HTMLCanvasElement)) {
    return;
  }
  const context = resultCanvas.getContext("2d");
  if (!context) {
    return;
  }
  context.setTransform(1, 0, 0, 1, 0, 0);
  context.clearRect(0, 0, resultCanvas.width, resultCanvas.height);
}

function setPending(message) {
  status.textContent = message;
  resultJson.textContent = "";
  currentResult = null;
  currentServerImageUrl = "";
  currentSourceImageUrl = "";
  if (resultImage instanceof HTMLImageElement) {
    resultImage.removeAttribute("src");
  }
  if (resultCanvas instanceof HTMLCanvasElement) {
    resultCanvas.hidden = true;
  }
  clearCanvas();
}

function summarizeResult(data) {
  const timings = data.timings_ms || {};
  const renders = data.available_renders || {};
  const localization = data.localization || {};
  const lines = [
    "识别完成",
    `render mode: ${data.render_mode ?? "server"}`,
    `requested locale: ${localization.requested_locale ?? "en"}`,
    `resolved locale: ${localization.resolved_locale ?? "en"}`,
    `solve: ${timings.solve ?? "?"} ms`,
    `scene: ${timings.scene ?? "?"} ms`,
    `overlay scene: ${timings.overlay_scene ?? "?"} ms`,
    `render: ${timings.render ?? "?"} ms`,
    `total: ${timings.total ?? "?"} ms`,
    `constellations: ${(data.visible_constellations || []).length}`,
    `stars: ${(data.visible_named_stars || []).length}`,
    `deep sky objects: ${(data.visible_deep_sky_objects || []).length}`,
    `server image: ${renders.server ? "yes" : "no"}`,
    `client overlay: ${renders.client ? "yes" : "no"}`,
  ];
  return lines.join("\n");
}

function summarizeJsonForDisplay(data) {
  const displayData = clone(data);
  if (typeof displayData.annotatedImageBase64 === "string") {
    displayData.annotatedImageBase64 = `<base64 ${displayData.annotatedImageBase64.length} chars>`;
  }
  return JSON.stringify(displayData, null, 2);
}

function buildOptionsForPreset(preset) {
  if (!overlayDefaults || !overlayPresets) {
    return null;
  }
  const nextOptions = clone(overlayDefaults);
  nextOptions.preset = preset;
  if (overlayPresets[preset]) {
    mergeNested(nextOptions, clone(overlayPresets[preset]));
  }
  return nextOptions;
}

function clearUploadObjectUrl() {
  if (currentUploadObjectUrl) {
    URL.revokeObjectURL(currentUploadObjectUrl);
    currentUploadObjectUrl = null;
  }
}

function createUploadObjectUrl(file) {
  clearUploadObjectUrl();
  currentUploadObjectUrl = URL.createObjectURL(file);
  return currentUploadObjectUrl;
}

function stripLargeImagePayload(data) {
  if (!data || typeof data !== "object") {
    return data;
  }
  if (typeof data.annotatedImageBase64 !== "string" || !data.annotatedImageBase64) {
    return data;
  }
  return {
    ...data,
    annotatedImageBase64: null,
  };
}

function rgbaToCss(rgba) {
  const [r, g, b, a = 255] = Array.isArray(rgba) ? rgba : [255, 255, 255, 255];
  return `rgba(${r}, ${g}, ${b}, ${Math.max(0, Math.min(1, a / 255))})`;
}

function canUseServerView() {
  return Boolean(currentServerImageUrl);
}

function canUseClientView() {
  return Boolean(currentResult?.overlay_scene && currentSourceImageUrl);
}

function syncViewerModeOptions() {
  if (!(viewerModeSelect instanceof HTMLSelectElement)) {
    return;
  }
  for (const option of viewerModeSelect.options) {
    if (option.value === "server") {
      option.disabled = !canUseServerView();
    }
    if (option.value === "client") {
      option.disabled = !canUseClientView();
    }
  }
}

function resolveViewerMode() {
  if (!currentResult) {
    return null;
  }

  const requested = viewerModeSelect instanceof HTMLSelectElement
    ? viewerModeSelect.value
    : "auto";
  const serverAvailable = canUseServerView();
  const clientAvailable = canUseClientView();

  if (requested === "server") {
    if (serverAvailable) {
      return "server";
    }
    return clientAvailable ? "client" : null;
  }

  if (requested === "client") {
    if (clientAvailable) {
      return "client";
    }
    return serverAvailable ? "server" : null;
  }

  const preferred = currentResult?.available_renders?.default_view === "client" ? "client" : "server";
  if (preferred === "server" && serverAvailable) {
    return "server";
  }
  if (preferred === "client" && clientAvailable) {
    return "client";
  }
  if (serverAvailable) {
    return "server";
  }
  if (clientAvailable) {
    return "client";
  }
  return null;
}

async function loadImageSource(sourceUrl) {
  if (!(resultImage instanceof HTMLImageElement)) {
    return false;
  }

  if (!sourceUrl) {
    resultImage.removeAttribute("src");
    return true;
  }

  const absoluteTarget = new URL(sourceUrl, window.location.href).href;
  if (resultImage.currentSrc === absoluteTarget && resultImage.complete && resultImage.naturalWidth > 0) {
    return true;
  }

  const token = ++imageLoadToken;
  await new Promise((resolve, reject) => {
    const cleanup = () => {
      resultImage.removeEventListener("load", handleLoad);
      resultImage.removeEventListener("error", handleError);
    };
    const handleLoad = () => {
      cleanup();
      resolve(token === imageLoadToken);
    };
    const handleError = () => {
      cleanup();
      if (token !== imageLoadToken) {
        resolve(false);
        return;
      }
      reject(new Error(`加载图片失败: ${sourceUrl}`));
    };

    resultImage.addEventListener("load", handleLoad, { once: true });
    resultImage.addEventListener("error", handleError, { once: true });
    resultImage.src = sourceUrl;

    if (resultImage.complete) {
      cleanup();
      if (resultImage.naturalWidth > 0) {
        resolve(token === imageLoadToken);
      } else if (token === imageLoadToken) {
        reject(new Error(`加载图片失败: ${sourceUrl}`));
      } else {
        resolve(false);
      }
    }
  });
  return token === imageLoadToken;
}

function drawDsoMarker(context, marker) {
  const x = marker.x;
  const y = marker.y;
  const radius = marker.radius;
  const color = rgbaToCss(marker.rgba);
  const width = Math.max(1, Number(marker.line_width) || 1);

  context.save();
  context.strokeStyle = color;
  context.lineWidth = width;
  context.lineJoin = "round";
  context.lineCap = "round";

  if (marker.marker === "square") {
    context.strokeRect(x - radius, y - radius, radius * 2, radius * 2);
  } else if (marker.marker === "crossed_circle") {
    context.beginPath();
    context.arc(x, y, radius, 0, Math.PI * 2);
    context.stroke();
    context.beginPath();
    context.moveTo(x - radius, y);
    context.lineTo(x + radius, y);
    context.moveTo(x, y - radius);
    context.lineTo(x, y + radius);
    context.stroke();
  } else if (marker.marker === "ring") {
    context.beginPath();
    context.arc(x, y, radius, 0, Math.PI * 2);
    context.stroke();
    context.beginPath();
    context.arc(x, y, Math.max(2, Math.floor(radius / 2)), 0, Math.PI * 2);
    context.stroke();
  } else if (marker.marker === "x_circle") {
    context.beginPath();
    context.arc(x, y, radius, 0, Math.PI * 2);
    context.stroke();
    context.beginPath();
    context.moveTo(x - radius, y - radius);
    context.lineTo(x + radius, y + radius);
    context.moveTo(x - radius, y + radius);
    context.lineTo(x + radius, y - radius);
    context.stroke();
  } else if (marker.marker === "hexagon") {
    const vertical = radius * 0.86;
    const horizontal = radius * 0.5;
    context.beginPath();
    context.moveTo(x - horizontal, y - vertical);
    context.lineTo(x + horizontal, y - vertical);
    context.lineTo(x + radius, y);
    context.lineTo(x + horizontal, y + vertical);
    context.lineTo(x - horizontal, y + vertical);
    context.lineTo(x - radius, y);
    context.closePath();
    context.stroke();
  } else if (marker.marker === "diamond") {
    context.beginPath();
    context.moveTo(x, y - radius);
    context.lineTo(x + radius, y);
    context.lineTo(x, y + radius);
    context.lineTo(x - radius, y);
    context.closePath();
    context.stroke();
  } else {
    context.beginPath();
    context.arc(x, y, radius, 0, Math.PI * 2);
    context.stroke();
  }

  context.restore();
}

function drawTextLabel(context, label) {
  const fontSize = Math.max(1, Number(label.font_size) || 12);
  const strokeWidth = Math.max(1, Number(label.stroke_width) || 1);

  context.save();
  context.font = `${fontSize}px "PingFang SC", "Hiragino Sans GB", "Noto Sans CJK SC", "Microsoft YaHei", sans-serif`;
  context.textBaseline = "top";
  context.lineJoin = "round";
  context.miterLimit = 2;

  if (label.leader) {
    context.strokeStyle = rgbaToCss(label.leader.rgba);
    context.lineWidth = Math.max(1, Number(label.leader.line_width) || 1);
    context.beginPath();
    context.moveTo(label.leader.x1, label.leader.y1);
    context.lineTo(label.leader.x2, label.leader.y2);
    context.stroke();
  }

  context.strokeStyle = rgbaToCss(label.stroke_rgba);
  context.lineWidth = strokeWidth * 2;
  context.strokeText(label.text, label.x, label.y);
  context.fillStyle = rgbaToCss(label.text_rgba);
  context.fillText(label.text, label.x, label.y);
  context.restore();
}

function drawOverlayScene(scene) {
  if (!(resultCanvas instanceof HTMLCanvasElement) || !scene) {
    return;
  }

  const width = Number(scene.image_width) || 0;
  const height = Number(scene.image_height) || 0;
  if (width <= 0 || height <= 0) {
    clearCanvas();
    return;
  }

  const dpr = window.devicePixelRatio || 1;
  resultCanvas.width = Math.round(width * dpr);
  resultCanvas.height = Math.round(height * dpr);
  resultCanvas.style.width = `${width}px`;
  resultCanvas.style.height = `${height}px`;
  resultCanvas.hidden = false;

  const context = resultCanvas.getContext("2d");
  if (!context) {
    return;
  }

  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  context.clearRect(0, 0, width, height);
  context.imageSmoothingEnabled = true;
  context.lineJoin = "round";
  context.lineCap = "round";

  for (const line of scene.constellation_lines || []) {
    context.strokeStyle = rgbaToCss(line.rgba);
    context.lineWidth = Math.max(1, Number(line.line_width) || 1);
    context.beginPath();
    context.moveTo(line.x1, line.y1);
    context.lineTo(line.x2, line.y2);
    context.stroke();
  }

  for (const marker of scene.deep_sky_markers || []) {
    drawDsoMarker(context, marker);
  }

  for (const marker of scene.star_markers || []) {
    const radius = Math.max(1, Number(marker.radius) || 1);
    context.beginPath();
    context.fillStyle = rgbaToCss(marker.fill_rgba);
    context.strokeStyle = rgbaToCss(marker.outline_rgba);
    context.lineWidth = 1;
    context.arc(marker.x, marker.y, radius, 0, Math.PI * 2);
    context.fill();
    context.stroke();
  }

  for (const label of scene.deep_sky_labels || []) {
    drawTextLabel(context, label);
  }
  for (const label of scene.constellation_labels || []) {
    drawTextLabel(context, label);
  }
  for (const label of scene.star_labels || []) {
    drawTextLabel(context, label);
  }
}

async function updateViewer() {
  if (!currentResult) {
    return;
  }

  syncViewerModeOptions();
  const mode = resolveViewerMode();
  if (!mode) {
    clearCanvas();
    return;
  }

  if (mode === "server" && currentServerImageUrl) {
    const loaded = await loadImageSource(currentServerImageUrl);
    if (!loaded) {
      return;
    }
    if (resultCanvas instanceof HTMLCanvasElement) {
      resultCanvas.hidden = true;
    }
    clearCanvas();
    return;
  }

  if (mode === "client") {
    const loaded = await loadImageSource(currentSourceImageUrl);
    if (!loaded) {
      return;
    }
    drawOverlayScene(currentResult.overlay_scene);
  }
}

async function renderResult(data, sourceImageUrl) {
  currentResult = stripLargeImagePayload(data);
  currentSourceImageUrl = typeof sourceImageUrl === "string" && sourceImageUrl
    ? sourceImageUrl
    : (typeof data.inputImageUrl === "string" ? data.inputImageUrl : "");
  currentServerImageUrl = typeof data.annotatedImageBase64 === "string" && data.annotatedImageBase64
    ? `data:${data.annotatedImageMimeType || "image/png"};base64,${data.annotatedImageBase64}`
    : "";

  status.textContent = summarizeResult(data);
  resultJson.textContent = summarizeJsonForDisplay(data);
  await updateViewer();
}

async function fetchJson(url, init) {
  const response = await fetch(url, init);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || `${response.status} ${response.statusText}`);
  }
  return payload;
}

async function loadOverlayOptions() {
  const payload = await fetchJson("/api/overlay-options");
  overlayDefaults = payload.defaults;
  overlayPresets = payload.presets;
  availableLocales = Array.isArray(payload.localization?.available_locales)
    ? payload.localization.available_locales
    : [];
  applyOptions(clone(payload.defaults));

  if (requestLocaleSelect instanceof HTMLSelectElement) {
    requestLocaleSelect.replaceChildren();
    for (const locale of availableLocales) {
      const option = document.createElement("option");
      option.value = locale;
      option.textContent = locale;
      requestLocaleSelect.appendChild(option);
    }
    const preferred = navigator.language || payload.localization?.default_locale || "en";
    requestLocaleSelect.value = pickPreferredLocale(preferred, availableLocales);
  }
}

async function loadSamples() {
  const sampleItems = await fetchJson("/api/samples");
  samples.replaceChildren();

  sampleItems.forEach((sample) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = `运行样例: ${sample.title}`;
    button.addEventListener("click", async () => {
      clearUploadObjectUrl();
      setPending(`正在处理样例 ${sample.title} ...`);
      try {
        const result = await fetchJson("/api/analyze-sample", {
          method: "POST",
          headers: {
            "content-type": "application/json",
          },
          body: JSON.stringify({
            id: sample.id,
            options: currentOptions(),
            render_mode: currentRequestRenderMode(),
            locale: currentRequestLocale(),
          }),
        });
        const sourceImageUrl = typeof result.inputImageUrl === "string"
          ? result.inputImageUrl
          : `/samples/${sample.filename}`;
        await renderResult(result, sourceImageUrl);
      } catch (error) {
        status.textContent = error instanceof Error ? error.message : "处理失败";
      }
    });

    const note = document.createElement("p");
    note.textContent = sample.note;

    const wrapper = document.createElement("div");
    wrapper.appendChild(button);
    wrapper.appendChild(note);
    samples.appendChild(wrapper);
  });
}

if (presetSelect instanceof HTMLSelectElement) {
  presetSelect.addEventListener("change", () => {
    const options = buildOptionsForPreset(presetSelect.value);
    if (options) {
      applyOptions(options);
    }
  });
}

viewerModeSelect?.addEventListener("change", () => {
  void updateViewer().catch((error) => {
    status.textContent = error instanceof Error ? error.message : "切换查看模式失败";
  });
});

uploadForm?.addEventListener("submit", async (event) => {
  event.preventDefault();
  const fileInput = document.getElementById("image");
  const file = fileInput instanceof HTMLInputElement ? fileInput.files?.[0] : null;
  if (!file) {
    status.textContent = "请先选择图片";
    return;
  }

  const localImageUrl = createUploadObjectUrl(file);
  setPending(`正在处理 ${file.name} ...`);
  const formData = new FormData();
  formData.set("image", file);
  formData.set("options", JSON.stringify(currentOptions()));
  formData.set("render_mode", currentRequestRenderMode());
  formData.set("locale", currentRequestLocale());

  try {
    const result = await fetchJson("/api/analyze", {
      method: "POST",
      body: formData,
    });
    await renderResult(result, localImageUrl);
  } catch (error) {
    status.textContent = error instanceof Error ? error.message : "处理失败";
  }
});

optionsForm?.addEventListener("change", () => {
  status.textContent = "参数已更新，重新运行一次即可看到新的结果";
});

window.addEventListener("beforeunload", () => {
  clearUploadObjectUrl();
});

try {
  await loadOverlayOptions();
  await loadSamples();
  status.textContent = "等待上传";
} catch (error) {
  status.textContent = error instanceof Error ? error.message : "初始化失败";
}
