"""Offline dataset loaders for tests and examples (data lives under tests/data/)."""

from pathlib import Path
import tarfile

import numpy as np
import pandas as pd

CALIFORNIA_HOUSING_FEATURE_NAMES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]
CALIFORNIA_HOUSING_TARGET_NAMES = ["MedHouseVal"]


def _tests_data_dir():
    return Path(__file__).resolve().parent / "data"


def _load_california_housing_from_tgz(bundle_path: Path):
    """Load California housing using sklearn's column transforms from the bundled archive."""
    with tarfile.open(bundle_path, mode="r:gz") as archive:
        cal_housing = np.loadtxt(
            archive.extractfile("CaliforniaHousing/cal_housing.data"),
            delimiter=",",
        )
    columns_index = [8, 7, 2, 3, 4, 5, 6, 1, 0]
    cal_housing = cal_housing[:, columns_index]
    target, data = cal_housing[:, 0], cal_housing[:, 1:].copy()
    data[:, 2] /= data[:, 5]
    data[:, 3] /= data[:, 5]
    data[:, 5] = data[:, 4] / data[:, 5]
    target = target / 100000.0
    return {
        "data": data,
        "target": target,
        "feature_names": CALIFORNIA_HOUSING_FEATURE_NAMES,
        "target_names": CALIFORNIA_HOUSING_TARGET_NAMES,
    }


def load_california_housing():
    """
    Load the California housing dataset from tests/data/.

    Avoids sklearn's network download (``fetch_california_housing``), which can
    fail in offline or restricted SSL environments.
    """
    data_dir = _tests_data_dir()
    npz_path = data_dir / "california_housing.npz"
    if npz_path.exists():
        cached = np.load(npz_path, allow_pickle=True)
        return {
            "data": cached["data"],
            "target": cached["target"],
            "feature_names": list(cached["feature_names"]),
            "target_names": list(cached["target_names"]),
        }

    tgz_path = data_dir / "cal_housing.tgz"
    if tgz_path.exists():
        return _load_california_housing_from_tgz(tgz_path)

    raise FileNotFoundError(
        "Bundled California housing data not found. Expected one of: "
        f"{npz_path}, {tgz_path}"
    )


def load_california_housing_frames():
    source = load_california_housing()
    data = pd.DataFrame(source["data"], columns=source["feature_names"])
    label = pd.Series(source["target"], name=source["target_names"][0])
    return data, label
