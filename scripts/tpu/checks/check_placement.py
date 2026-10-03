"""Pre-flight 1: can Ray place every engine and the trainer of CONFIG on this cluster, and does
the TPU initialize in each? Starts placeholder actors with exactly the resources the run will
request (no models are loaded) and prints where each landed. Needs the Ray cluster up
(scripts/tpu/start_cluster.sh) and the chips free.

  python scripts/tpu/checks/check_placement.py CONFIG.yaml [--plan]
"""
import socket
import sys

from _common import raw_plan


def main(config_path: str, plan_only: bool) -> None:
    p = raw_plan(config_path)
    print(f"PLAN  total chips {p['total']} = {len(p['students'])} student engine(s) + {len(p['teachers'])} teacher engine(s) "
          f"({p['engine_chips']} chips) + trainer {p['trainer_chips']} chips")
    if p["trainer_chips"] <= 0:
        sys.exit("trainer would get no chips: raise cluster.node_num/gpu_per_node or lower engine_num")
    if plan_only:
        return
    import ray

    from trinity.common.models.tpu_env import chips_per_host, whole_host_runtime_env

    ray.init(address="auto", log_to_driver=False)
    res = ray.cluster_resources()
    labels = {k: v for k, v in res.items() if k == "TPU" or k.endswith("_tpu")}
    print("CLUSTER", labels, "| chips per host", chips_per_host())
    student_label = "explorer_tpu" if "explorer_tpu" in res else None
    teacher_label = "teacher_tpu" if "teacher_tpu" in res else student_label

    @ray.remote
    class Probe:
        def check(self):
            import os

            import jax

            return socket.gethostname(), os.environ.get("TPU_VISIBLE_CHIPS"), jax.device_count()

    specs = [(role, n, student_label) for role, n in p["students"]] + [(role, n, teacher_label) for role, n in p["teachers"]]
    specs.append(("trainer", p["trainer_chips"], "trainer_tpu" if "trainer_tpu" in res else None))
    actors = []
    for role, n, label in specs:
        resources = {"TPU": n, **({label: n} if label else {})}
        actors.append((role, n, Probe.options(num_cpus=0, resources=resources, runtime_env=whole_host_runtime_env(n)).remote()))
    ok = True
    for role, n, a in actors:
        try:
            host, chips, devices = ray.get(a.check.remote(), timeout=300)
            good = devices == n
            print(f"{'OK  ' if good else 'BAD '} {role:8s} host {host} chips {chips} jax devices {devices} (wanted {n})")
            ok &= good
        except Exception as e:  # not schedulable (wrong layout) or TPU init hang
            print(f"FAIL {role:8s} wanting {n} chip(s): {type(e).__name__} (not placed within 5 min, or TPU init hung)")
            ok = False
        ray.kill(a)
    print("PLACEMENT OK" if ok else "PLACEMENT FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(sys.argv[1], "--plan" in sys.argv)
