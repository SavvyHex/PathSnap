const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const source = fs.readFileSync(path.join(__dirname, "../ui/app.js"), "utf8");
const flush = () => new Promise(resolve => setImmediate(resolve));

function runUI(fetch, search = "") {
  class Element {
    constructor(text = "", value = "") {
      this.textContent = text;
      this.value = value;
      this.children = [];
      this.handlers = {};
      this.disabled = true;
      this.hidden = true;
    }
    replaceChildren(...children) { this.children = children; }
    append(...children) { this.children.push(...children); }
    add(child) { this.append(child); }
    addEventListener(event, listener) { this.handlers[event] = listener; }
  }

  const elements = new Map();
  const document = {
    querySelector(id) {
      if (!elements.has(id)) elements.set(id, new Element());
      return elements.get(id);
    },
    createElement: () => new Element(),
  };
  const drawn = [];
  const geojson = [];
  const map = { setView: () => map, fitBounds: () => {} };
  const leaflet = {
    map: () => map,
    tileLayer: () => ({ addTo() {} }),
    polyline(points, options) {
      drawn.push({ points, options });
      return { addTo() {}, remove() {}, getBounds: () => ({}) };
    },
    circleMarker: () => ({ bindTooltip: () => ({}) }),
    featureGroup: () => ({ addTo() {}, remove() {}, getBounds: () => ({}) }),
    geoJSON(data) {
      geojson.push(data);
      return { addTo() {}, remove() {} };
    },
    control: { layers: () => ({ addTo: () => ({ remove() {} }) }) },
  };
  vm.runInNewContext(source, {
    document, window: { L: leaflet }, L: leaflet, location: { search },
    URLSearchParams, URL, AbortController, setTimeout, clearTimeout,
    Option: Element, fetch,
  });

  return {
    element: document.querySelector,
    drawn,
    geojson,
    selectTrip(trip) {
      const selector = document.querySelector("#trip");
      selector.value = trip;
      selector.handlers.change();
    },
    submit() {
      return document.querySelector("#prediction-form").handlers.submit({ preventDefault() {} });
    },
  };
}

function response(data, status = 200) {
  return { ok: status === 200, status, headers: new Headers(), json: async () => data };
}

test("configured API, development mode, trip selection, metrics and map", async () => {
  const calls = [];
  const ui = runUI(async (url, { method }) => {
    calls.push([url, method]);
    if (method === "GET") {
      return response({ mode: "development", trips: ["development-demo"] });
    }
    return response({
      mode: "development", trip_id: "development-demo", n_observations: 1,
      route: [{ obs_id: 0, segment: null, lat: 12, lon: 77 }],
    });
  }, "?api=http%3A%2F%2F192.168.1.10%3A8000%2F");
  assert.match(ui.element("#status").textContent, /Loading trips/);
  await flush();
  assert.match(ui.element("#mode").textContent, /Development mode/);
  assert.equal(ui.element("#trip").children[1].value, "development-demo");
  assert.equal(ui.element("#predict").disabled, true);
  ui.selectTrip("development-demo");
  const pending = ui.submit();
  assert.equal(ui.element("#trip").disabled, true);
  await pending;
  assert.deepEqual(calls, [
    ["http://192.168.1.10:8000/trips", "GET"],
    ["http://192.168.1.10:8000/predict/development-demo", "POST"],
  ]);
  assert.deepEqual(ui.element("#metrics").children.map(element => element.textContent),
    ["Observations", 1, "Route points", 1]);
  assert.equal(ui.drawn[0].points[0][0], 12);
  assert.equal(ui.element("#map-note").textContent, "Synthetic demo trace");
  assert.equal(ui.element("#predict").disabled, false);
});

test("prediction failures surface the API error and unlock the controls", async () => {
  const ui = runUI(async (url, { method }) => method === "GET"
    ? response({ mode: "evaluation", trips: ["00000007"] })
    : response({ detail: "Model artifacts are unavailable." }, 503));
  await flush();
  ui.selectTrip("00000007");
  await ui.submit();
  assert.equal(ui.element("#results").hidden, true);
  assert.equal(ui.element("#error").hidden, false);
  assert.match(ui.element("#error").textContent, /Model artifacts are unavailable/);
  assert.equal(ui.element("#predict").disabled, false);
});

test("network failures are visible when loading trips", async () => {
  const ui = runUI(async () => { throw new TypeError("offline"); });
  await flush();
  assert.match(ui.element("#error").textContent, /Could not reach http:\/\/localhost:8000/);
  assert.equal(ui.element("#trip").disabled, true);
});

test("evaluation renders only genuine road layers and available metrics", async () => {
  const arcs = { type: "FeatureCollection", features: [{
    type: "Feature", geometry: { type: "LineString", coordinates: [[77, 12], [78, 13]] },
  }] };
  const ui = runUI(async (url, { method }) => method === "GET"
    ? response({ mode: "evaluation", trips: ["00000007"] })
    : response({
      mode: "evaluation", trip_id: "00000007", route: [], n_observations: 2,
      gps_observations: [{ obs_id: 0, lat: 12, lon: 77 }, { obs_id: 1, lat: 13, lon: 78 }],
      geometry: { geojson: arcs, covered_observations: 1, decoded_observations: 2 },
      transition_model: "directed_road_network", route_completeness: "partial",
      connected_geometry: arcs, disconnected_transitions: 1, inference_s: 1.3,
      route_overlap: { score: 0.25 }, top1_accuracy: null,
    }));
  await flush();
  ui.selectTrip("00000007");
  await ui.submit();
  assert.deepEqual(ui.geojson, [arcs, arcs]);
  assert.equal(ui.drawn.length, 0);
  assert.equal(ui.element("#legend-gps").hidden, false);
  assert.equal(ui.element("#legend-connected").hidden, false);
  assert.equal(ui.element("#fit-trip").disabled, false);
  const values = ui.element("#metrics").children.map(element => element.textContent);
  const value = label => values[values.indexOf(label) + 1];
  assert.equal(value("Top-1 accuracy"), "N/A");
  assert.equal(value("Directed-arc overlap (Jaccard)"), "25.0%");
  assert.equal(value("Geometry coverage"), "1 / 2");
  assert.equal(value("Route status"), "Partial route");
});
