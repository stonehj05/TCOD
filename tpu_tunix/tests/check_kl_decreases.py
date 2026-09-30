"""Sanity check: OPD updates on one fixed batch must pull the student toward the teacher.

Collects one rollout batch, then repeatedly (score -> update) on that same batch
and prints the student-teacher token KL, which should fall steadily.
Run: python tests/check_kl_decreases.py --config configs/smoke.yaml lr=1e-5
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from opd import Mode, OPDConfig, OPDTrainer  # noqa: E402

cfg_path, overrides = sys.argv[2], sys.argv[3:]
values = yaml.safe_load(open(cfg_path))
values.update({k: yaml.safe_load(v) for k, v in (o.split("=", 1) for o in overrides)})
values["eval_file"] = None
trainer = OPDTrainer(OPDConfig.from_dict(values))
roll = trainer.run_episodes(trainer.rng.sample(trainer.train_tasks, trainer.cfg.batch_size), Mode.TRAIN)
batch = trainer.build_batch(roll["turns"])
mask = batch["mask_np"]
kls = []
for it in range(int(os.environ.get("ITERS", 8))):
    teacher, student = trainer.score(batch)
    kl = float((np.asarray(student - teacher) * mask).sum() / mask.sum())
    kls.append(kl)
    print(f"iter {it}: token KL(student||teacher) estimate = {kl:.4f}", flush=True)
    trainer.update(batch, teacher, student)
assert kls[-1] < kls[0], f"KL did not decrease: {kls}"
print("OK: KL decreased", kls[0], "->", kls[-1])
