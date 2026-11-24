"""Dựng lại mô hình đã khóa trong thư mục tạm và đối chiếu kết quả."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.release_experiment import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    args = parser.parse_args()
    base = args.output.resolve()
    started = perf_counter()
    with TemporaryDirectory(prefix="equityrank-verify-") as folder:
        fresh = Path(folder)
        invoke = [sys.executable, str(ROOT / "scripts/run_jpx_project.py")]
        options = ["--raw-dir", str(args.raw_dir.resolve()), "--output", str(fresh)]
        subprocess.run([*invoke, "prepare", *options], check=True)
        shutil.copyfile(base / "selected_model.json", fresh / "selected_model.json")
        for stage in ("train", "rank", "evaluate", "backtest"):
            subprocess.run([*invoke, stage, *options], check=True)
        for file in ("ridge_predictions.csv", "predictions.csv"):
            expected = pd.read_csv(base / file, float_precision="round_trip")
            actual = pd.read_csv(fresh / file, float_precision="round_trip")
            pd.testing.assert_frame_equal(expected.drop(columns="Score"), actual.drop(columns="Score"), check_exact=True)
            pd.testing.assert_series_equal(expected["Score"], actual["Score"], rtol=1e-6, atol=1e-8)
        for file in ("control_scores.csv", "historical_daily.csv", "historical_summary.csv", "historical_matched_summary.csv",
                     "backtest_daily.csv", "backtest_costs.csv", "target_weight_proxy.csv", "execution_daily.csv",
                     "execution_orders.csv", "execution_costs.csv", "execution_coverage.csv"):
            pd.testing.assert_frame_equal(pd.read_csv(base / file), pd.read_csv(fresh / file), rtol=1e-8, atol=1e-10)
        for name in ("ridge_metadata.json", "selected_metadata.json"):
            assert json.loads((base / name).read_text(encoding="utf-8")) == json.loads(
                (fresh / name).read_text(encoding="utf-8")
            )
        selection = json.loads((fresh / "selected_model.json").read_text(encoding="utf-8"))
        metadata = json.loads((fresh / "selected_metadata.json").read_text(encoding="utf-8"))
        if metadata["last_label"] >= selection["cutoff"]:
            raise ValueError("A training label crossed the locked historical boundary.")
    result = {"status": "completed", "seconds": round(perf_counter() - started, 2), "exact_ranks": True}
    write_json(base / "reproduction.json", result)
    manifest = json.loads((base / "run_manifest.json").read_text(encoding="utf-8"))
    manifest["steps"]["reproduce"] = result
    write_json(base / "run_manifest.json", manifest)
    print("Đã dựng lại từ dữ liệu gốc. Thứ hạng và các bảng đánh giá khớp.")


if __name__ == "__main__":
    main()
