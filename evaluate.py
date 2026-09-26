import argparse
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from data.loader import build_loader, load_market
from train import build_model, get_device
from training import test_mae


def evaluate(run):
    cfg = OmegaConf.load(run / "config.yaml")
    cfg.dataset.num_workers = 0
    device = get_device(cfg)
    model = build_model(cfg).load(run / "model.pt").to(device)
    data = load_market(cfg.market, cfg.dataset.processed_dir)

    grid = [(0.0, 0)]
    if cfg.variant != "ablated":
        grid += [(r, s) for r in cfg.evaluation.rates for s in cfg.evaluation.mask_seeds]

    rows = []
    for rate, mask_seed in grid:
        loader = build_loader(data, "test", cfg, cfg.window.stride_eval, rate=rate, seed=mask_seed)
        rows.append({"market": cfg.market, "model": cfg.name, "variant": cfg.variant,
                     "seed": cfg.seed, "rate": rate, "mask_seed": mask_seed,
                     "MAE": test_mae(model, loader, data["scaler"], device)})
    pd.DataFrame(rows).to_csv(run / "metrics.csv", index=False)
    print(f"{run} | MAE(0) = {rows[0]['MAE']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="./runs")
    ap.add_argument("--market", default="*")
    ap.add_argument("--model", default="*")
    ap.add_argument("--variant", default="*")
    args = ap.parse_args()

    pattern = f"{args.market}/{args.model}/{args.variant}/seed*/model.pt"
    for ckpt in sorted(Path(args.runs).glob(pattern)):
        evaluate(ckpt.parent)


if __name__ == "__main__":
    main()
