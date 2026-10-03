"""Pre-flight 3: does the Tunix trainer of CONFIG fit and step at full sequence length?
Runs the trainer worker in this process (no Ray) on the first N chips of THIS host with a
synthetic batch of train_batch_size rows whose prompts go up to max_prompt_tokens, and reports
step time and peak HBM per chip. Run it on the trainer host with its chips free. The first
step compiles one program per sequence-length bucket (several minutes).

  python scripts/tpu/checks/check_trainer_memory.py CONFIG.yaml [--chips N] [--plan]
"""
import sys
import time

from _common import local_chips, raw_plan, set_inprocess_tpu_env, validated_config


def main(config_path: str, chips: int, plan_only: bool) -> None:
    p = raw_plan(config_path)
    cfg0 = p["cfg"]
    chips = chips or p["trainer_chips"]
    print(f"PLAN  trainer {cfg0.model.model_path} on {chips} chip(s) of this host ({local_chips()} present), "
          f"batch {cfg0.buffer.train_batch_size}, prompts up to {cfg0.model.max_prompt_tokens} + {cfg0.model.max_response_tokens} tokens")
    if chips > local_chips():
        sys.exit(f"this host has {local_chips()} chips; the trainer must fit on one host")
    if plan_only:
        return
    set_inprocess_tpu_env(chips)
    import numpy as np
    from transformers import AutoTokenizer

    from trinity.trainer.tunix.hf_io import resolve_model_dir
    from trinity.trainer.tunix.worker import TunixActorWorker
    from trinity.trainer.tunix_trainer import TunixTrainerWrapper

    cfg = validated_config(config_path)
    wcfg = TunixTrainerWrapper(cfg).worker_cfg
    wcfg["rows_per_micro_batch"] = chips
    tok = AutoTokenizer.from_pretrained(resolve_model_dir(cfg.model.model_path))
    wcfg["pad_token_id"], wcfg["eos_token_id"] = tok.pad_token_id, tok.convert_tokens_to_ids("<|im_end|>")
    t = time.time()
    w = TunixActorWorker(wcfg)
    print(f"INIT {time.time() - t:.0f}s, buckets {wcfg['seq_buckets']}", flush=True)
    rng = np.random.default_rng(0)
    top = cfg.model.max_prompt_tokens

    def row(p_len):
        r = int(rng.integers(20, min(300, cfg.model.max_response_tokens)))
        return {"prompt": rng.integers(1000, 100000, p_len).astype(np.int32), "response": rng.integers(1000, 100000, r).astype(np.int32),
                "response_mask": np.ones(r, bool), "teacher_logprobs": -rng.random(r).astype(np.float32) * 3}

    # full-memory-like mix: prompt length grows with the turn index up to the cap
    rows = [row(int(min(top, 700 + (top // 31) * (i % 31)))) for i in range(cfg.buffer.train_batch_size)]
    for s in range(2):
        t = time.time()
        m = w.train_step(rows)
        print(f"STEP {s}: {time.time() - t:.0f}s (old logprobs {m['time/old_log_prob']:.0f}s, update {m['time/update_actor']:.0f}s, "
              f"{m['tunix/micro_batches']:.0f} micro-batches)", flush=True)
    import jax

    peak = [d.memory_stats()["peak_bytes_in_use"] / 2**30 for d in jax.local_devices()]
    limit = [d.memory_stats()["bytes_limit"] / 2**30 for d in jax.local_devices()]
    print("HBM peak GiB per chip:", [round(x, 1) for x in peak], "of", round(limit[0], 1))
    print("TRAINER OK" if max(peak) < 0.95 * limit[0] else "TRAINER TIGHT: peak above 95% of HBM")


if __name__ == "__main__":
    n = int(sys.argv[sys.argv.index("--chips") + 1]) if "--chips" in sys.argv else 0
    main(sys.argv[1], n, "--plan" in sys.argv)
