"""Prepare unlabeled candidate CSVs offline and discover usable prediction inputs."""
import argparse
import os
import re
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd

from src.config import ARTIFACTS, DATA_CLEAN, DATA_RAW, TEST_DIR, TEST_TRIPS
from src.data.preprocess import generate_candidates
from src.features.build_features import motion_features


TRIP_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")
METADATA = ["obs_id", "candidate_id", "latitude", "longitude"]
RAW_COLUMNS = {
    "track": ["longitude", "latitude", "seconds"],
    "nodes": ["longitude", "latitude"],
    "arcs": ["from_node", "to_node"],
}
STUDENT_OUTPUT = DATA_CLEAN / "evaluator_student_test"
EVALUATOR_OUTPUT = DATA_CLEAN / "evaluator"


def trip_files(directory: Path) -> dict[str, Path]:
    """List prepared CSV paths by exact filename stem."""
    return {
        path.stem: path
        for path in directory.iterdir()
        if path.suffix == ".csv" and TRIP_ID.fullmatch(path.stem)
        and not path.is_symlink() and path.is_file()
    }


def read_prepared_trip(path: Path, columns) -> pd.DataFrame:
    """Read only observation metadata and ordered model inputs."""
    selected = list(dict.fromkeys([*METADATA, *columns]))
    frame = pd.read_csv(path, usecols=selected, dtype={"candidate_id": "string"},
                        keep_default_na=False, na_values=[""])[selected]
    if frame.empty:
        raise ValueError("Empty evaluator CSV")
    obs_ids = pd.to_numeric(frame["obs_id"])
    coordinates = frame[["latitude", "longitude"]].to_numpy(dtype=float)
    if (not pd.api.types.is_integer_dtype(obs_ids) or obs_ids.isna().any()
            or (obs_ids < 0).any() or not np.isfinite(coordinates).all()
            or (np.abs(coordinates) > [90, 180]).any()):
        raise ValueError("Invalid observation IDs or coordinates")
    frame["obs_id"] = obs_ids
    candidates = frame.loc[frame["candidate_id"].notna()]
    if candidates.duplicated(["obs_id", "candidate_id"]).any():
        raise ValueError("Duplicate candidates in observation")
    if frame[["obs_id", "latitude", "longitude"]].drop_duplicates()["obs_id"].duplicated().any():
        raise ValueError("Inconsistent observation coordinates")
    features = candidates[list(columns)].apply(pd.to_numeric)
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite model features")
    return frame


@lru_cache(maxsize=256)
def _usable_trip(path: Path, columns: tuple[str, ...], signature: tuple[int, ...]) -> bool:
    # The signature invalidates cached validation when the CSV is replaced or edited.
    try:
        return bool(read_prepared_trip(path, columns)["candidate_id"].notna().any())
    except (OSError, ValueError):
        return False


def available_trip_files(directory: Path, columns) -> dict[str, Path]:
    """Filter prepared files to trips with at least one valid candidate."""
    available = {}
    for trip_id, path in trip_files(directory).items():
        try:
            info = path.stat()
        except FileNotFoundError:
            continue
        signature = (info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        if _usable_trip(path, tuple(columns), signature):
            available[trip_id] = path
    return available


def discover_source_trips(root: Path, *, student_test: bool = False) -> dict[str, Path]:
    """Discover source folders, including incomplete ones for explicit reporting."""
    if not root.exists():
        return {}
    return {
        path.name: path for path in sorted(root.iterdir())
        if path.is_dir() and not path.is_symlink() and TRIP_ID.fullmatch(path.name)
        and (not student_test or path.name in TEST_TRIPS)
    }


def read_raw_table(folder: Path, extension: str) -> pd.DataFrame:
    """Read one GPS or network input using the repository's positional format."""
    names = RAW_COLUMNS[extension]
    path = folder / f"{folder.name}.{extension}"
    try:
        table = pd.read_csv(path, sep=r"\s+", header=None)
    except pd.errors.EmptyDataError as error:
        raise ValueError(f"Empty input: {path.name}") from error
    if table.shape[1] != len(names):
        raise ValueError(f"Expected {len(names)} columns in {path.name}")
    table.columns = names
    table = table.apply(pd.to_numeric)
    if not np.isfinite(table.to_numpy(dtype=float)).all():
        raise ValueError(f"Missing or non-finite values in {path.name}")
    if extension == "arcs":
        if any(not pd.api.types.is_integer_dtype(table[col]) for col in names) or (table < 0).any().any():
            raise ValueError(f"Invalid node IDs in {path.name}")
    elif (table["longitude"].abs() > 180).any() or (table["latitude"].abs() > 90).any():
        raise ValueError(f"Invalid GPS coordinates in {path.name}")
    return table


def _read_raw_trip(folder: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # The training parser also reads .route. Open only the three inference files.
    track, nodes, arcs = (read_raw_table(folder, extension) for extension in RAW_COLUMNS)
    track["trip_id"] = folder.name
    track["obs_id"] = range(len(track))
    nodes["node_id"] = range(len(nodes))
    arcs["arc_id"] = range(len(arcs))
    return track, nodes, arcs


def prepare_trip(folder: Path, columns) -> pd.DataFrame:
    """Run Member 1's candidate and motion algorithms without reference inputs."""
    if not TRIP_ID.fullmatch(folder.name):
        raise ValueError("Invalid source trip ID")
    track, nodes, arcs = _read_raw_trip(folder)
    try:
        candidates, _ = generate_candidates(track, nodes, arcs)
    except ValueError as error:
        # Member 1 concatenates an empty list when the search grid has no overlap.
        if str(error) != "No objects to concatenate":
            raise
        raise ValueError("No road candidates found") from error
    if candidates.empty:
        raise ValueError("No road candidates found")

    # Match training: derive motion from observations represented by candidates.
    observations = (candidates.drop_duplicates("obs_id")
                    .sort_values("obs_id")[["obs_id", "latitude", "longitude", "seconds"]]
                    .reset_index(drop=True))
    motion = motion_features(observations).assign(obs_id=observations["obs_id"])
    candidates = candidates.merge(motion, on="obs_id", validate="many_to_one")
    if not np.isfinite(candidates[list(columns)].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite generated model features")

    # Carry genuine GPS observations with no candidates as blank candidate rows.
    candidates["candidate_id"] = candidates["candidate_id"].astype("Int64")
    output = track[["trip_id", "obs_id", "latitude", "longitude"]].merge(
        candidates[["obs_id", "candidate_id", *columns]], on="obs_id", how="left",
    )
    return output[["trip_id", *METADATA, *columns]]


def write_prepared_trip(frame: pd.DataFrame, path: Path) -> bool:
    """Publish atomically. An identical output is a no-op; differences are refused."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
        try:
            # Windows rename refuses to overwrite. POSIX rename would replace.
            if os.name == "nt":
                temporary.rename(path)
            else:
                os.link(temporary, path)
        except FileExistsError:
            if not path.is_symlink() and path.is_file() and path.read_bytes() == temporary.read_bytes():
                return False
            raise FileExistsError(f"Existing output differs: {path}") from None
        return True
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=["evaluator", "student_test"])
    parser.add_argument("--source-dir", type=Path)
    args = parser.parse_args()

    # Local import avoids circular dependency and keeps module execution simple.
    from api.inference import read_feature_manifest

    student_test = args.source == "student_test"
    root = args.source_dir or (TEST_DIR if student_test else DATA_RAW / "evaluator")
    if not student_test and root.name.casefold() in {"student_test", "train"}:
        parser.error("Select the student_test source for student-test data; training data is excluded.")
    output = STUDENT_OUTPUT if student_test else EVALUATOR_OUTPUT
    trips = discover_source_trips(root, student_test=student_test)
    if not trips:
        print(f"No {args.source} source trips found in {root}")
        return 1
    columns = read_feature_manifest(ARTIFACTS)
    succeeded = 0
    for trip_id, folder in trips.items():
        try:
            frame = prepare_trip(folder, columns)
            created = write_prepared_trip(frame, output / f"{trip_id}.csv")
        except (OSError, ValueError) as error:
            print(f"{trip_id}: skipped: {error}")
            continue
        succeeded += 1
        candidates = frame["candidate_id"].notna()
        gaps = frame.loc[~candidates, "obs_id"].nunique()
        print(f"{trip_id}: {'prepared' if created else 'unchanged'}; "
              f"{candidates.sum()} candidates, {gaps} observations without candidates")
    print(f"{succeeded}/{len(trips)} {args.source} trips prepared in {output}")
    return int(succeeded != len(trips))


if __name__ == "__main__":
    raise SystemExit(main())
