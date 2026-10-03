"""Shared helpers for the TPU pre-flight checks."""
import glob
import os


def raw_plan(config_path: str) -> dict:
    """Chip plan straight from the yaml (no validation side effects)."""
    from trinity.common.config import load_config

    cfg = load_config(config_path)
    r = cfg.explorer.rollout_model
    students = [("student", r.tensor_parallel_size)] * r.engine_num
    teachers = [("teacher", a.tensor_parallel_size) for a in cfg.explorer.auxiliary_models for _ in range(a.engine_num)]
    total = (cfg.cluster.node_num or 0) * (cfg.cluster.gpu_per_node or 0)
    engine_chips = sum(n for _, n in students + teachers)
    return {"cfg": cfg, "students": students, "teachers": teachers, "total": total,
            "engine_chips": engine_chips, "trainer_chips": total - engine_chips}


def validated_config(config_path: str):
    """Fully validated config, redirected to a scratch run dir so no stray run folder is
    created next to real runs."""
    from trinity.common.config import load_config

    cfg = load_config(config_path)
    cfg.name = cfg.name + "_preflight"
    cfg.checkpoint_root_dir = "/dev/shm/tcod_preflight"
    cfg.continue_from_checkpoint = False
    cfg.check_and_update()
    return cfg


def local_chips() -> int:
    return len(glob.glob("/dev/accel*"))


def set_inprocess_tpu_env(n_chips: int) -> None:
    """TPU env for running a JAX program on the first n_chips of THIS host, outside Ray.
    Must be called before the TPU backend initializes."""
    from trinity.common.models.tpu_env import whole_host_runtime_env

    os.environ.setdefault("TCOD_TPU_CHIPS_PER_HOST", str(local_chips() or 4))
    env = whole_host_runtime_env(n_chips).get("env_vars")
    if env is None:  # sub-host allocation: the settings Ray would give such an actor
        env = {"TPU_VISIBLE_CHIPS": ",".join(str(i) for i in range(n_chips)),
               "TPU_CHIPS_PER_HOST_BOUNDS": f"1,{n_chips},1", "TPU_HOST_BOUNDS": "1,1,1"}
    os.environ.update(env)
