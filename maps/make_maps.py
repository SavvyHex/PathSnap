"""Folium map generation.

Builds the maps required by the brief: raw GPS observations, road
network geometry, reference route and predicted route as separate,
colour-coded layers with a legend and layer control. One representative
map is exported per selected trip plus an overview map.
"""
import folium
import pandas as pd
from folium.plugins import MeasureControl

from src.config import MAPS_DIR, RESULTS


def base_map(lat, lon, zoom=14):
    m = folium.Map(location=[lat, lon], zoom_start=zoom, tiles="OpenStreetMap")
    m.add_child(MeasureControl())
    return m


def add_gps_layer(m, obs, name="GPS observations", color="blue"):
    fg = folium.FeatureGroup(name=name)
    for _, r in obs.iterrows():
        folium.CircleMarker([r["latitude"], r["longitude"]], radius=3,
                            color=color, fill=True, fill_opacity=0.8).add_to(fg)
    fg.add_to(m)
    return fg


def add_route_layer(m, points, name, color, weight=4):
    fg = folium.FeatureGroup(name=name)
    folium.PolyLine(points, color=color, weight=weight, opacity=0.9).add_to(fg)
    fg.add_to(m)
    return fg


def trip_map(obs, reference=None, predicted=None, network=None, out="trip_map.html"):
    m = base_map(obs["latitude"].mean(), obs["longitude"].mean())
    add_gps_layer(m, obs)
    if network is not None:
        add_route_layer(m, network, "Road network", "grey", weight=2)
    if reference is not None:
        add_route_layer(m, reference, "Reference route", "green")
    if predicted is not None:
        add_route_layer(m, predicted, "Predicted route", "red")
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(str(MAPS_DIR / out))
    return MAPS_DIR / out


def main() -> None:
    # Wired up once data/clean is populated: overview map across all
    # trips plus one detailed map per selected student-test trip.
    raise SystemExit("Populate data/clean first, then select trips for the report maps.")


if __name__ == "__main__":
    main()
