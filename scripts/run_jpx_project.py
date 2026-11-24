"""Chạy quy trình nghiên cứu và sinh hạng cổ phiếu JPX."""

import argparse
from hashlib import file_digest
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.final_pipeline import atomic_write
from src.release_data import read_release_inputs
from src.release_experiment import PROTOCOL, develop, source_hash, write_json


COMMANDS = ("prepare", "develop", "select", "train", "rank", "evaluate", "backtest")


def sha256(path):
    with path.open("rb") as file:
        return file_digest(file, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--pilot", action="store_true", help="Chạy thử lượt walk-forward đầu tiên.")
    args = parser.parse_args()
    if args.pilot and args.command != "develop":
        parser.error("--pilot chỉ dùng với develop.")
    out = args.output
    started = perf_counter()
    manifest_path = out / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"steps": {}}
    if args.command == "prepare":
        manifest = {"protocol": PROTOCOL, "steps": {}}
        prices, financial = read_release_inputs(args.raw_dir)
        for name, table in (("prices", prices), ("financial", financial)):
            atomic_write(out / f"{name}.pkl", table.to_pickle)
        inputs = {f"{folder}/{file}": sha256(args.raw_dir / folder / file)
                  for folder in ("train_files", "supplemental_files") for file in ("stock_prices.csv", "financials.csv")}
        manifest["inputs"] = inputs
        manifest["rows"] = {"prices": len(prices), "financial": len(financial)}
        manifest["prepared"] = {n: sha256(out / f"{n}.pkl") for n in ("prices", "financial")}
        manifest["packages"] = {name: version(name) for name in
                                ("numpy", "pandas", "scikit-learn", "lightgbm", "xgboost", "torch")}
        print(f"Đã nạp {len(prices):,} dòng giá và {len(financial):,} công bố.", flush=True)
    else:
        if manifest.get("protocol") != PROTOCOL or "prepared" not in manifest:
            parser.error("Chạy prepare trước để tạo dữ liệu và manifest.")
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            try:
                torch.empty(1, device=device).fill_(1).item()
            except RuntimeError as error:
                print(f"Kiểm tra CUDA thất bại, chuyển sang CPU. {error}", flush=True)
                device = "cpu"
        packages = {name: version(name) for name in manifest["packages"]}
        if packages != manifest["packages"]:
            raise ValueError("Phiên bản thư viện đã thay đổi sau bước prepare.")
        identity = {"inputs": manifest["inputs"], "source": source_hash(), "device": device,
                    "packages": packages}
        if args.command == "rank":
            for name, expected in manifest["inputs"].items():
                if sha256(args.raw_dir / name) != expected:
                    raise ValueError(f"Dữ liệu gốc {name} khác manifest.")
        prepared = ("financial",) if args.command == "rank" else (
            ("prices",) if args.command in {"evaluate", "backtest"} else ("prices", "financial")
        )
        for name in prepared:
            if sha256(out / f"{name}.pkl") != manifest["prepared"][name]:
                raise ValueError(f"Dữ liệu đã chuẩn bị {name} khác manifest.")
        if args.command == "develop":
            develop(pd.read_pickle(out / "prices.pkl"), pd.read_pickle(out / "financial.pkl"),
                    out, identity, device, pilot=args.pilot)
        else:
            from src.release_delivery import run_step

            run_step(args.command, args.raw_dir, out, identity, device)
    key = "pilot" if args.pilot and args.command == "develop" else args.command
    manifest["steps"][key] = {"seconds": round(perf_counter() - started, 3), "status": "completed"}
    write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
