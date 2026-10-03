"""Save the exact training state (fp32 params, Adam state, step) of a RUNNING tunix trainer.

For runs started before trainers exported this themselves at full checkpoints. Runs inside
the live trainer worker via Ray's __ray_call__ (between train steps; the run keeps going).
Refuses unless the worker is exactly at STEP.

  python scripts/tpu/export_live_state.py CONFIG.yaml STEP
"""

import os
import sys
import time

import ray

from trinity.common.config import load_config
from trinity.trainer.tunix import resume


def main(config_path: str, step: int) -> None:
    cfg = load_config(config_path)
    namespace = f"{cfg.project}/{cfg.name}"  # Trinity's default ray_namespace
    job_dir = os.path.join(os.path.abspath(cfg.checkpoint_root_dir), cfg.project, cfg.name)
    root = (os.path.join(cfg.trainer.sync_checkpoint_dir, cfg.project, cfg.name)
            if cfg.trainer.sync_checkpoint_dir else job_dir)
    out_dir = resume.resume_state_dir(root, step)
    ray.init(address="auto", namespace=namespace, log_to_driver=False)
    worker = ray.get_actor(f"{cfg.trainer.name}_tunix_worker", namespace=namespace)
    t0 = time.time()
    info = ray.get(worker.__ray_call__.remote(resume.export_state, out_dir, step))
    print(f"EXPORTED {info} -> {out_dir} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
