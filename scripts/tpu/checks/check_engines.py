"""Pre-flight 2: bring up every vLLM-TPU engine of CONFIG through Trinity (models ARE loaded)
and exercise the calls OPD workflows make: a student turn with logprobs, a teacher generation,
and teacher scoring of the student's turn, plus an in-place weight reload on one student.
Needs the Ray cluster up and the chips free; takes a few minutes (first calls compile).

  python scripts/tpu/checks/check_engines.py CONFIG.yaml [--plan]
"""
import sys
import time

from _common import raw_plan, validated_config


def main(config_path: str, plan_only: bool) -> None:
    p = raw_plan(config_path)
    cfg0 = p["cfg"]
    print(f"PLAN  students: {len(p['students'])} x {cfg0.model.model_path} | teachers: "
          f"{[(a.model_path, a.engine_num, a.tensor_parallel_size) for a in cfg0.explorer.auxiliary_models]}")
    if plan_only:
        return
    import ray

    from trinity.common.models import create_inference_models

    cfg = validated_config(config_path)
    ray.init(address="auto", namespace=cfg.ray_namespace, log_to_driver=False)
    t = time.time()
    students, aux = create_inference_models(cfg)
    teachers = [m for group in aux for m in group]
    ray.get([m.prepare.remote() for m in students + teachers])
    print(f"ENGINES {len(students)} student(s) + {len(teachers)} teacher(s) ready in {time.time() - t:.0f}s", flush=True)
    msgs = [{"role": "user", "content": "You are in a kitchen. Admissible actions: 'go to fridge 1', 'open cabinet 2'. "
                                          "Reason briefly, then give the action in <action></action> tags."}]
    e = ray.get(students[0].chat.remote(msgs, n=1, temperature=1.0))[0]
    n_resp = len(e.tokens) - e.prompt_length
    assert e.logprobs is not None and len(e.logprobs) == n_resp, "student must return one logprob per response token"
    print(f"STUDENT turn: {n_resp} tokens with logprobs: {e.response_text[:100]!r}", flush=True)
    for i, tch in enumerate(teachers):
        q = msgs + [{"role": "assistant", "content": e.response_text},
                    {"role": "user", "content": "Did the last step make progress? Answer yes or no, then explain briefly."}]
        t = time.time()
        r = ray.get(tch.chat.remote(q, n=1, temperature=0.0, max_tokens=128))[0]
        print(f"TEACHER{i} generation ({time.time() - t:.1f}s): {r.response_text[:100]!r}", flush=True)
        t = time.time()
        lp = ray.get(tch.logprobs.remote(e.tokens.tolist(), temperature=1.0))[e.prompt_length - 1:]
        assert len(lp) == n_resp, f"teacher returned {len(lp)} response logprobs, expected {n_resp}"
        print(f"TEACHER{i} scoring ({time.time() - t:.1f}s): {len(lp)} logprobs, mean {float(lp.mean()):.3f}", flush=True)
    # In-place weight reload (what happens after every training step), using the base weights.
    from trinity.trainer.tunix.hf_io import resolve_model_dir

    n = ray.get(students[0]._collective_rpc.remote("update_weight_from_checkpoint", args=(resolve_model_dir(cfg.model.model_path),)))
    print(f"RELOAD student 0: {n} tensors replaced in place", flush=True)
    for m in students + teachers:
        ray.get(m.shutdown.remote())
        ray.kill(m)
    print("ENGINES OK")


if __name__ == "__main__":
    main(sys.argv[1], "--plan" in sys.argv)
