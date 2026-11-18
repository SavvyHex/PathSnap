const apiInput = document.querySelector("#api-url");
const select = document.querySelector("#trip");
const predictButton = document.querySelector("#predict");
const status = document.querySelector("#status");
const error = document.querySelector("#error");
const mode = document.querySelector("#mode");
const metrics = document.querySelector("#metrics");
const results = document.querySelector("#results");
const mapNote = document.querySelector("#map-note");
const fitButton = document.querySelector("#fit-trip");
const legend = document.querySelector("#map-legend");

apiInput.value = new URLSearchParams(location.search).get("api") || "http://localhost:8000";
const map = window.L ? L.map("map", { preferCanvas: true }).setView([0, 0], 2) : null;
if (map) {
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "&copy; OpenStreetMap contributors",
  }).addTo(map);
}
let layerControl;
let displayedLayers = [];
let tripBounds;

function apiBase() {
  let url;
  try {
    url = new URL(new URLSearchParams(location.search).get("api") || "http://localhost:8000");
  } catch {
    throw new Error("Enter a valid HTTP or HTTPS API address and select Connect.");
  }
  if (!["http:", "https:"].includes(url.protocol) || url.username || url.password
      || url.search || url.hash) {
    throw new Error("Enter an HTTP or HTTPS API address without credentials, a query or a fragment.");
  }
  return url.href.replace(/\/+$/, "");
}

async function request(path, method = "GET") {
  const base = apiBase();
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 180000);
  try {
    let response;
    try {
      response = await fetch(base + path, { method, signal: controller.signal });
    } catch (failure) {
      if (failure.name === "AbortError") throw failure;
      throw new Error("Could not reach " + base + ". Check the API address, network and CORS settings.");
    }
    const data = await response.json().catch(() => null);
    if (controller.signal.aborted) throw new Error("The API request timed out.");
    if (!response.ok) {
      const id = response.headers.get("X-Request-ID");
      const detail = typeof data?.detail === "string" ? data.detail : "HTTP " + response.status;
      throw new Error(detail + (id ? " Request ID " + id : ""));
    }
    if (!data) throw new Error("The API returned an invalid response.");
    return data;
  } catch (failure) {
    if (failure.name === "AbortError") throw new Error("The API request timed out.");
    throw failure;
  } finally {
    clearTimeout(timer);
  }
}

function showError(failure) {
  error.textContent = failure.message;
  error.hidden = false;
}

function showMode(currentMode) {
  const demo = currentMode === "development";
  mode.hidden = !demo;
  mode.textContent = demo ? "Development mode. Synthetic route preview only." : "";
  predictButton.textContent = demo ? "Preview demo route" : "Run prediction";
}

function fitTrip() {
  if (tripBounds) map.fitBounds(tripBounds, { padding: [24, 24], maxZoom: 17 });
}

fitButton.addEventListener("click", fitTrip);

function drawRoute(data) {
  layerControl?.remove();
  displayedLayers.forEach(layer => layer.remove());
  layerControl = null;
  displayedLayers = [];
  tripBounds = null;
  fitButton.disabled = legend.hidden = true;
  mapNote.textContent = map ? "" : "Map unavailable. Leaflet could not load.";
  if (!map || !data) return;
  const demo = data.mode === "development";
  const gps = data.gps_observations ?? data.route;
  const overlays = {};
  if (demo && data.route.length) {
    overlays["Development preview"] = L.polyline(
      data.route.map(point => [point.lat, point.lon]), { color: "#805d0d", dashArray: "6 6" },
    );
  } else if (gps.length) {
    overlays["GPS observations"] = L.featureGroup(gps.map(point =>
      L.circleMarker([point.lat, point.lon], {
        radius: 3, color: "#185aa7", weight: 1, fillOpacity: 0.75,
      }).bindTooltip("Observation " + Number(point.obs_id)),
    ));
  }
  if (!demo && data.geometry?.geojson?.features?.length) {
    overlays["Predicted road arcs"] = L.geoJSON(data.geometry.geojson, {
      style: { color: "#087760", weight: 4, opacity: 0.8 },
    });
  }
  const connected = data.route_completeness === "connected" && data.connected_geometry;
  if (connected) {
    overlays["Connected route"] = L.geoJSON(connected, { style: { color: "#7a3cb1", weight: 5, dashArray: "8 5" } });
  }
  displayedLayers = Object.values(overlays);
  if (displayedLayers.length) {
    displayedLayers.forEach(layer => layer.addTo(map));
    layerControl = L.control.layers({}, overlays, { collapsed: false }).addTo(map);
    tripBounds = L.featureGroup(displayedLayers).getBounds();
    fitButton.disabled = false;
    fitTrip();
  }
  document.querySelector("#legend-gps").hidden = !overlays["GPS observations"];
  document.querySelector("#legend-arcs").hidden = !overlays["Predicted road arcs"];
  document.querySelector("#legend-connected").hidden = !connected;
  legend.hidden = demo || !displayedLayers.length;
  mapNote.textContent = demo ? "Synthetic demo trace" : (data.transition_limitation ?? "");
}

const percent = value => Number.isFinite(value) ? (100 * value).toFixed(1) + "%" : "N/A";

function showResults(data) {
  const demo = data.mode === "development";
  showMode(data.mode);
  const values = [
    ["Observations", data.n_observations],
    ["Route points", data.route.length],
  ];
  if (!demo) {
    const geometry = data.geometry;
    const completeness = { candidate_only: "Candidate sequence", partial: "Partial candidate sequence",
      connected: "Connected route", unavailable: "Unavailable" };
    const decoder = { directed_road_network: "Viterbi · directed road distances",
      classifier_only: "Classifier only", neutral: "Viterbi · equal costs" };
    values.push(
      ["Imputed observations", data.n_imputed],
      ["Scored observations", data.scored_observations],
      ["Top-1 accuracy", percent(data.top1_accuracy)],
      ["Directed-arc overlap (Jaccard)", percent(data.route_overlap?.score)],
      ["Inference time", Number.isFinite(data.inference_s) ? data.inference_s + " s" : "N/A"],
      ["Geometry coverage", geometry ? geometry.covered_observations + " / " + geometry.decoded_observations : "N/A"],
      ["Decoder", decoder[data.transition_model] ?? "N/A"],
      ["Route status", completeness[data.route_completeness] ?? "N/A"],
      ["Disconnected transitions", data.disconnected_transitions ?? "N/A"],
    );
  }
  metrics.replaceChildren();
  for (const [label, value] of values) {
    const term = document.createElement("dt");
    const description = document.createElement("dd");
    term.textContent = label;
    description.textContent = value;
    metrics.append(term, description);
  }
  drawRoute(data);
  results.hidden = false;
  status.textContent = data.route.length ? "Showing " + data.trip_id + "." : "No route points were decoded.";
}

async function loadTrips() {
  status.textContent = "Loading trips...";
  drawRoute();
  try {
    const data = await request("/trips");
    if (!Array.isArray(data.trips)) throw new Error("The API returned an invalid trip list.");
    showMode(data.mode);
    select.replaceChildren(new Option("Select a trip", ""));
    for (const trip of data.trips) select.add(new Option(trip, trip));
    select.disabled = !data.trips.length;
    status.textContent = data.trips.length ? "" : "No evaluator trips are available.";
  } catch (failure) {
    status.textContent = "";
    showError(failure);
  }
}

select.addEventListener("change", () => {
  predictButton.disabled = !select.value;
  results.hidden = true;
  error.hidden = true;
  status.textContent = "";
  drawRoute();
});

document.querySelector("#prediction-form").addEventListener("submit", async event => {
  event.preventDefault();
  if (predictButton.disabled) return;
  const trip = select.value;
  select.disabled = predictButton.disabled = true;
  results.hidden = error.hidden = true;
  status.textContent = "Loading " + trip + "...";
  drawRoute();
  try {
    showResults(await request("/predict/" + encodeURIComponent(trip), "POST"));
  } catch (failure) {
    status.textContent = "";
    showError(failure);
  } finally {
    select.disabled = predictButton.disabled = false;
  }
});

loadTrips();
