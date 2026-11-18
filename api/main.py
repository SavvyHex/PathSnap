"""PathSnap evaluator API. Reference data is used only after decoding."""
import logging
import math
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.evaluator_adapter import TRIP_ID, available_trip_files, trip_files
from api.geometry import geometry_unavailable, index_road_arcs, load_network, resolve_geometry
from api.inference import MODEL_COLUMNS, load_model, predict_route
from api.reconstruction import reconstruct_route
from api.scoring import overlap_unavailable, score_arc_overlap, score_route
from src.config import ARTIFACTS, DATA_CLEAN, HMM_BETA, REFERENCE


log = logging.getLogger("uvicorn.error").getChild("pathsnap")
DEMO_ID = "development-demo"


@dataclass(frozen=True)
class Settings:
    mode: str = "evaluation"
    evaluator_dir: Path = DATA_CLEAN / "evaluator"
    artifacts_dir: Path = ARTIFACTS
    reference_dir: Path = REFERENCE
    cors_origins: tuple[str, ...] = ("http://localhost:8080", "http://127.0.0.1:8080")
    beta: float = HMM_BETA
    network_dir: Path = DATA_CLEAN / "network"
    network_source_dir: Path | None = None
    route_reference_dir: Path | None = None

    @classmethod
    def from_env(cls):
        defaults = cls()
        origins = os.getenv("PATHSNAP_CORS_ORIGINS", ",".join(defaults.cors_origins))
        network_source = os.getenv("PATHSNAP_NETWORK_SOURCE_DIR")
        route_reference = os.getenv("PATHSNAP_ROUTE_REFERENCE_DIR")
        settings = cls(
            mode=os.getenv("PATHSNAP_MODE", defaults.mode),
            evaluator_dir=Path(os.getenv("PATHSNAP_EVALUATOR_DIR", defaults.evaluator_dir)),
            artifacts_dir=Path(os.getenv("PATHSNAP_ARTIFACTS_DIR", defaults.artifacts_dir)),
            reference_dir=Path(os.getenv("PATHSNAP_REFERENCE_DIR", defaults.reference_dir)),
            cors_origins=tuple(origin.strip() for origin in origins.split(",") if origin.strip()),
            beta=float(os.getenv("PATHSNAP_HMM_BETA", defaults.beta)),
            network_dir=Path(os.getenv("PATHSNAP_NETWORK_DIR", defaults.network_dir)),
            network_source_dir=Path(network_source) if network_source else None,
            route_reference_dir=Path(route_reference) if route_reference else None,
        )
        if settings.mode not in {"development", "evaluation"}:
            raise ValueError("PATHSNAP_MODE must be development or evaluation")
        if not math.isfinite(settings.beta) or settings.beta <= 0:
            raise ValueError("PATHSNAP_HMM_BETA must be finite and positive")
        return settings


def create_app(settings=None):
    # Embedded callers can supply standalone classifiers through explicit settings.
    allow_custom_classifier = (
        settings is not None and settings.artifacts_dir.resolve() != ARTIFACTS.resolve()
    )
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app):
        app.state.model_bundle = None
        if settings.mode == "evaluation":
            try:
                app.state.model_bundle = load_model(
                    settings.artifacts_dir,
                    allow_custom_classifier=allow_custom_classifier,
                )
            except Exception:
                log.exception("pathsnap.model_load_failed")
        log.info("pathsnap.startup mode=%s model_loaded=%s",
                 settings.mode, app.state.model_bundle is not None)
        yield

    app = FastAPI(title="PathSnap", version="1.1", lifespan=lifespan)

    @app.middleware("http")
    async def request_log(request: Request, call_next):
        request_id = uuid4().hex
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log.exception("pathsnap.request_failed request_id=%s", request_id)
            response = JSONResponse({"detail": "The server could not complete the request."}, 500)
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["Server-Timing"] = f"app;dur={elapsed_ms:.1f}"
        route = request.scope.get("route")
        log.info("pathsnap.request request_id=%s method=%s route=%s status=%s elapsed_ms=%.1f",
                 request_id, request.method, getattr(route, "path", "unmatched"),
                 response.status_code, elapsed_ms)
        return response

    app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins),
                       allow_methods=["GET", "POST"],
                       expose_headers=["X-Request-ID", "Server-Timing"])

    @app.get("/health")
    def health():
        return dict(status="ok", mode=settings.mode,
                    model_loaded=app.state.model_bundle is not None)

    def available_trips():
        bundle = app.state.model_bundle
        columns = bundle[1] if bundle is not None else MODEL_COLUMNS
        try:
            return available_trip_files(settings.evaluator_dir, columns)
        except FileNotFoundError:
            return {}

    @app.get("/ready")
    def readiness():
        loaded = app.state.model_bundle is not None
        if settings.mode == "development":
            return dict(status="ready", mode=settings.mode, model_loaded=False)
        try:
            has_trips = bool(available_trips())
        except OSError:
            has_trips = False
        ready = loaded and has_trips
        return JSONResponse(dict(status="ready" if ready else "unavailable",
                                 mode=settings.mode, model_loaded=loaded,
                                 trips_available=has_trips), 200 if ready else 503)

    @app.get("/trips")
    def trips():
        if settings.mode == "development":
            return dict(mode=settings.mode, trips=[DEMO_ID])
        return dict(mode=settings.mode, trips=sorted(available_trips()))

    @app.post("/predict/{trip_id}")
    def predict(trip_id: str):
        started = time.perf_counter()
        unknown = HTTPException(404, "Unknown trip.")
        if not TRIP_ID.fullmatch(trip_id):
            raise unknown
        if settings.mode == "development":
            if trip_id != DEMO_ID:
                raise unknown
            points = [(12.9337, 77.5332), (12.9341, 77.5340), (12.9346, 77.5345)]
            route = [
                dict(obs_id=i, segment=None, lat=lat, lon=lon, imputed=False)
                for i, (lat, lon) in enumerate(points)
            ]
            return dict(mode=settings.mode, trip_id=trip_id, route=route,
                        n_observations=len(route), n_imputed=None, top1_accuracy=None,
                        scored_observations=0, elapsed_s=None, transition_model=None,
                        route_kind="development_preview",
                        gps_observations=[], inference_s=None, connected_geometry=None,
                        route_completeness="unavailable", disconnected_transitions=None,
                        connected_transitions=None, unreachable_transitions=None,
                        reconstructed_connecting_arcs=None, reconstructed_distance_m=None,
                        route_distance_m=None,
                        geometry=dict(geometry_unavailable("development_preview"),
                                      covered_observations=0, decoded_observations=0),
                        route_overlap=overlap_unavailable("development_preview"))
        try:
            path = trip_files(settings.evaluator_dir).get(trip_id)
        except FileNotFoundError:
            path = None
        if path is None:
            raise unknown
        if app.state.model_bundle is None:
            raise HTTPException(503, "Model artifacts are unavailable.")
        network = None
        network_error = False
        try:
            network = load_network(settings.network_dir, trip_id, settings.network_source_dir)
        except Exception:
            log.exception("pathsnap.network_load_failed")
            network_error = True
        try:
            inference_started = time.perf_counter()
            route, gps, imputed, transition_model, disconnected, limitation, reconstruction_input = predict_route(
                path, *app.state.model_bundle, settings.beta, network,
            )
            inference_s = round(time.perf_counter() - inference_started, 3)
        except FileNotFoundError:
            raise unknown
        try:
            accuracy, scored = score_route(settings.reference_dir / f"{trip_id}.csv", route)
        except Exception:
            log.exception("pathsnap.reference_scoring_failed")
            accuracy, scored = None, 0
        road_arcs = index_road_arcs(network)
        geometry = (geometry_unavailable("invalid_network_data") if network_error
                    else resolve_geometry(road_arcs, route))
        geojson = geometry["geojson"]
        geometry["covered_observations"] = len(geojson["features"]) if geojson else 0
        geometry["decoded_observations"] = len(route)
        overlap = overlap_unavailable("route_reference_not_configured")
        if settings.route_reference_dir is not None:
            reference = settings.route_reference_dir / trip_id / f"{trip_id}.route"
            try:
                overlap = score_arc_overlap(reference, route, road_arcs)
            except (OSError, UnicodeError):
                log.exception("pathsnap.route_reference_scoring_failed")
                overlap = overlap_unavailable("invalid_route_reference")
        completeness = ("unavailable" if not route else
                        "partial" if disconnected or imputed or geometry["status"] == "partial"
                        else "candidate_only")
        reconstruction = reconstruct_route(*reconstruction_input) if reconstruction_input else None
        if reconstruction:
            if disconnected or reconstruction.geometry_failures or not reconstruction.spans_trip:
                completeness = "partial"
                limitation = ("Some selected transitions are unreachable or lack continuous geometry. "
                              "Only verified road sections are shown.")
            elif reconstruction.geojson:
                completeness = "connected"
            else:
                limitation = "The selected road positions have no nonzero route geometry."
        distance = reconstruction.distance_m if reconstruction else None
        return dict(mode=settings.mode, trip_id=trip_id, route=route, n_observations=len(gps),
                    n_imputed=imputed, top1_accuracy=accuracy, scored_observations=scored,
                    elapsed_s=round(time.perf_counter() - started, 3),
                    transition_model=transition_model, route_kind="candidate_sequence",
                    transition_limitation=limitation, disconnected_transitions=disconnected,
                    gps_observations=gps, inference_s=inference_s,
                    route_completeness=completeness,
                    connected_geometry=reconstruction.geojson if reconstruction else None,
                    connected_transitions=reconstruction.connected_transitions if reconstruction else None,
                    unreachable_transitions=reconstruction.unreachable_transitions if reconstruction else None,
                    reconstructed_connecting_arcs=reconstruction.connecting_arcs if reconstruction else None,
                    reconstructed_distance_m=distance,
                    route_distance_m=distance if completeness == "connected" else None,
                    geometry=geometry, route_overlap=overlap)

    return app


app = create_app()
