from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing

from evidently import Report
from evidently.presets import DataDriftPreset


RANDOM_STATE = 42


def make_production_data(reference: pd.DataFrame, mode: str = "drift") -> pd.DataFrame:

    rng = np.random.default_rng(RANDOM_STATE)

    prod = reference.sample(n=min(4000, len(reference)), random_state=RANDOM_STATE).copy()

    if mode == "nodrift":
        # small noise only
        noise_cols = ["MedInc", "AveRooms", "AveOccup"]
        for col in noise_cols:
            prod[col] = prod[col] + rng.normal(0, prod[col].std() * 0.02, size=len(prod))
        return prod

    if mode == "drift":
        # 1) income shift upward
        prod["MedInc"] = prod["MedInc"] * 1.25 + rng.normal(0, prod["MedInc"].std() * 0.05, size=len(prod))

        # 2) occupancy shift upward
        prod["AveOccup"] = prod["AveOccup"] * 1.15 + rng.normal(0, prod["AveOccup"].std() * 0.05, size=len(prod))

        # 3) geographic shift (slight move south-east)
        prod["Latitude"] = prod["Latitude"] - 0.6 + rng.normal(0, 0.05, size=len(prod))
        prod["Longitude"] = prod["Longitude"] + 0.7 + rng.normal(0, 0.05, size=len(prod))

        # 4) extra noise on rooms/bedrooms
        prod["AveRooms"] = prod["AveRooms"] + rng.normal(0, prod["AveRooms"].std() * 0.08, size=len(prod))
        prod["AveBedrms"] = prod["AveBedrms"] + rng.normal(0, prod["AveBedrms"].std() * 0.08, size=len(prod))

        return prod

    raise ValueError("mode must be 'nodrift' or 'drift'")


def main() -> None:
    X, y = fetch_california_housing(return_X_y=True, as_frame=True)
    # reference = "training-like" data
    reference = X.sample(n=min(8000, len(X)), random_state=RANDOM_STATE).copy()

    # production simulation
    production_mode = "drift"  # change to "nodrift" to compare
    production = make_production_data(reference=reference, mode=production_mode)

    # Evidently drift report (Evidently 0.7.x)
    report = Report([DataDriftPreset()])

    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"data_drift_report_{production_mode}.html"

    result = report.run(reference_data=reference, current_data=production)
    result.save_html(str(out_path))

    print(f"[OK] Drift report generated: {out_path.resolve()}")


if __name__ == "__main__":
    main()
