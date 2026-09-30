"""Entry point: python train_opd.py --config configs/alfworld_opd.yaml [key=value ...]"""

import argparse
import logging

import yaml

from opd import OPDConfig, OPDTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*", help="key=value overrides of OPDConfig fields")
    args = ap.parse_args()
    with open(args.config) as f:
        values = yaml.safe_load(f) or {}
    for kv in args.overrides:
        k, v = kv.split("=", 1)
        values[k] = yaml.safe_load(v)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", force=True)
    for name in ("absl", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)
    OPDTrainer(OPDConfig.from_dict(values)).train()


if __name__ == "__main__":
    main()
