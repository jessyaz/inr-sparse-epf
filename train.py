import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

from data.loader import MARKETS, build_loader, load_market
from models.dnn import DNN
from models.epf_transformer import EPFTransformer
from models.inr import INR
from models.lear import LEAR
from models.transformer import Transformer
from training import train

MODELS = ["lear", "dnn", "epf_transformer", "transformer_imputed", "transformer_masked", "inr"]
ARCHS = {"lear": LEAR, "dnn": DNN, "epf_transformer": EPFTransformer,
         "transformer": Transformer, "inr": INR}
VARIANTS = {
    "full": {},
    "ablated": {"model": {"use_lookback": False}},
    "static": {"model": {"static_exog": True}},
    "noexog": {"model": {"use_exog": False}},
}


def make_config(name, market, seed, variant, overrides=()):
    cfg = OmegaConf.merge(OmegaConf.load("conf/base.yaml"), OmegaConf.load(f"conf/{name}.yaml"))
    per_market = cfg.pop("markets", {})
    cfg = OmegaConf.merge(cfg, per_market.get(market, {}), VARIANTS[variant],
                          {"name": name, "market": market, "seed": seed, "variant": variant},
                          OmegaConf.from_dotlist(list(overrides)))
    return cfg


def run_dir(cfg):
    return Path(cfg.out_dir) / cfg.market / cfg.name / cfg.variant / f"seed{cfg.seed}"


def get_device(cfg):
    if cfg.device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return cfg.device


def build_model(cfg):
    return ARCHS[cfg.arch](cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=MODELS)
    ap.add_argument("--market", default="DE", choices=MARKETS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variant", default="full", choices=list(VARIANTS))
    args, overrides = ap.parse_known_args()

    cfg = make_config(args.model, args.market, args.seed, args.variant, overrides)
    out = run_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out / "config.yaml")

    torch.manual_seed(cfg.seed)
    device = get_device(cfg)
    data = load_market(cfg.market, cfg.dataset.processed_dir)
    train_loader = build_loader(data, "train", cfg, cfg.window.stride_train, shuffle=True)
    val_loader = build_loader(data, "val", cfg, cfg.window.stride_eval)
    model = build_model(cfg).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"{out} | device={device} | {n_params:,} parameters")

    if isinstance(model, LEAR):
        model.fit(train_loader, val_loader, build_loader(data, "test", cfg, cfg.window.stride_eval))
    else:
        train(model, train_loader, val_loader, device, cfg)
    model.save(out / "model.pt")


if __name__ == "__main__":
    main()
