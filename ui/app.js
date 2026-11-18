// PathSnap evaluator UI. Set this to the API host (laptop 1).
const PATHSNAP_API = "http://localhost:8000";

const map = L.map("map").setView([12.935, 77.605], 14); // PESU area default
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  attribution: "&copy; OpenStreetMap contributors",
}).addTo(map);

let routeLayer = null;

async function loadTrips() {
  const box = document.getElementById("trips");
  try {
    const res = await fetch(`${PATHSNAP_API}/trips`);
    const data = await res.json();
    box.innerHTML = "";
    data.trips.forEach((t) => {
      const b = document.createElement("button");
      b.textContent = t;
      b.onclick = () => runPrediction(t);
      box.appendChild(b);
    });
  } catch (e) {
    box.textContent = "Could not reach the API at " + PATHSNAP_API;
  }
}

async function runPrediction(tripId) {
  const stats = document.getElementById("stats");
  stats.textContent = `Running ${tripId}...`;
  const res = await fetch(`${PATHSNAP_API}/predict/${tripId}`, { method: "POST" });
  const data = await res.json();
  if (!res.ok) { stats.textContent = data.detail || "prediction failed"; return; }

  if (routeLayer) map.removeLayer(routeLayer);
  const pts = data.route.map((r) => [r.lat, r.lon]);
  routeLayer = L.layerGroup([
    L.polyline(pts, { color: "red", weight: 4 }),
    ...pts.map((p) => L.circleMarker(p, { radius: 3, color: "blue" })),
  ]).addTo(map);
  map.fitBounds(L.polyline(pts).getBounds());

  stats.textContent =
    `Trip: ${data.trip_id}\n` +
    `Observations: ${data.n_observations} (imputed: ${data.n_imputed})\n` +
    `Top-1 accuracy: ${data.top1_accuracy === null ? "n/a" : (100 * data.top1_accuracy).toFixed(1) + "%"}\n` +
    `Pipeline time: ${data.elapsed_s}s`;
}

loadTrips();
