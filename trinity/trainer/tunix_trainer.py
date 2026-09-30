"""TPU trainer backend (trainer_type: tunix).

Plays the role of VerlPPOTrainerWrapper: `train_step(exps)` turns experiences
into arrays and hands them to a `TunixActorWorker` Ray actor that owns
`cluster.trainer_gpu_num` TPU chips (verl's actor worker group analogue).

Weight sync is `sync_method: checkpoint`: `save_state_dict` writes a HF
safetensors checkpoint to `<checkpoint_job_dir>/global_step_N/actor/huggingface`
and bumps `latest_state_dict_iteration.txt`; the Synchronizer publishes that
path and the vLLM-TPU engines load it (vllm_tpu_worker.py).
"""

import os
from collections import defaultdict
from typing import Dict, List

import numpy as np
import ray

from trinity.common.config import Config
from trinity.common.experience import Experience
from trinity.trainer.trainer import TrainEngineWrapper
from trinity.utils.log import get_logger


def hf_checkpoint_dir(checkpoint_job_dir: str, step: int) -> str:
    return os.path.join(checkpoint_job_dir, f"global_step_{step}", "actor", "huggingface")


def sync_checkpoint_root(config: Config) -> str:
    """Directory holding this run's weight-sync checkpoints (see TrainerConfig.sync_checkpoint_dir)."""
    root = config.trainer.sync_checkpoint_dir
    if not root:
        return config.checkpoint_job_dir
    return os.path.join(root, config.project, os.path.basename(os.path.normpath(config.checkpoint_job_dir)))


def sync_checkpoint_dir(config: Config, step: int) -> str:
    return hf_checkpoint_dir(sync_checkpoint_root(config), step)


class TunixTrainerWrapper(TrainEngineWrapper):
    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger(__name__, in_ray_actor=True)
        algo = config.algorithm
        if algo.algorithm_type != "on_policy_distill":
            raise NotImplementedError(
                f"tunix trainer implements on_policy_distill only, got {algo.algorithm_type}")
        loss_args = {"clip_range": 0.2, "clip_ratio_c": 3.0, "loss_agg_mode": "token-mean",
                     **(algo.policy_loss_fn_args or {})}
        adv_args = {"kl_coef": 1.0, **(algo.advantage_fn_args or {})}
        model = config.model
        from trinity.trainer.tunix.worker import default_seq_buckets

        block = 512
        self.worker_cfg = {
            "model_path": model.model_path,
            "max_response_tokens": model.max_response_tokens,
            "temperature": model.temperature,
            "kl_coef": adv_args["kl_coef"],
            "clip_range_low": loss_args.get("clip_range_low") or loss_args["clip_range"],
            "clip_range_high": loss_args.get("clip_range_high") or loss_args["clip_range"],
            "clip_ratio_c": loss_args["clip_ratio_c"],
            "loss_agg_mode": loss_args["loss_agg_mode"],
            "ppo_epochs": 1,
            "rows_per_micro_batch": config.cluster.trainer_gpu_num,  # one row per chip
            "flash_block_size": block,
            "remat": "DECODER",
            "seq_buckets": default_seq_buckets(model.max_prompt_tokens, model.max_response_tokens, block),
            "optimizer": {
                "lr": algo.optimizer.lr,
                "betas": list(algo.optimizer.betas),
                "weight_decay": algo.optimizer.weight_decay,
                "clip_grad": config.trainer.grad_clip,
            },
        }
        if algo.optimizer.lr_scheduler_type != "constant" or algo.optimizer.lr_warmup_steps_ratio:
            raise NotImplementedError("tunix trainer supports a constant learning rate only")
        self.default_local_dir = config.checkpoint_job_dir
        self.local_latest_state_dict_iteration = os.path.join(
            self.default_local_dir, "latest_state_dict_iteration.txt")
        self.local_latest_checkpointed_iteration = os.path.join(
            self.default_local_dir, "latest_checkpointed_iteration.txt")
        self._train_step_num = 0
        self._last_state_dict_step = 0
        self._last_checkpoint_step = 0
        self.worker = None

    async def prepare(self) -> None:
        from transformers import AutoTokenizer

        from trinity.trainer.tunix.hf_io import resolve_model_dir
        from trinity.trainer.tunix.worker import TunixActorWorker

        tok = AutoTokenizer.from_pretrained(resolve_model_dir(self.config.model.model_path))
        self.worker_cfg["pad_token_id"] = tok.pad_token_id
        self.worker_cfg["eos_token_id"] = tok.convert_tokens_to_ids("<|im_end|>")
        if self.config.continue_from_checkpoint and os.path.exists(self.local_latest_checkpointed_iteration):
            raise NotImplementedError("resuming a tunix run from a checkpoint is not implemented yet")
        os.makedirs(self.default_local_dir, exist_ok=True)
        n_chips = self.config.cluster.trainer_gpu_num
        resources = {"TPU": n_chips}
        if "trainer_tpu" in ray.cluster_resources():  # multi-host: see _create_tpu_inference_models
            resources["trainer_tpu"] = n_chips
        runtime_env = {}
        if n_chips == 4:
            # A whole v4 host: Ray sets no per-host bounds, so libtpu would wait for every
            # host of the slice to join. Declare a single-process 2x2x1 slice instead.
            runtime_env = {"env_vars": {
                "TPU_CHIPS_PER_PROCESS_BOUNDS": "2,2,1", "TPU_PROCESS_BOUNDS": "1,1,1",
                "TPU_VISIBLE_CHIPS": "0,1,2,3", "TPU_PROCESS_PORT": "8476",
                "TPU_PROCESS_ADDRESSES": "localhost:8476"}}
        self.worker = (
            ray.remote(TunixActorWorker)
            .options(num_cpus=1, resources=resources, runtime_env=runtime_env,
                     name=f"{self.config.trainer.name}_tunix_worker", namespace=self.config.ray_namespace)
            .remote(self.worker_cfg)
        )
        await self.worker.ready.remote()

    @property
    def train_step_num(self) -> int:
        return self._train_step_num

    @staticmethod
    def _to_row(exp: Experience) -> Dict[str, np.ndarray]:
        tokens = np.asarray(exp.tokens, dtype=np.int32)
        response = tokens[exp.prompt_length:]
        teacher = np.asarray(exp.teacher_logprobs, dtype=np.float32)
        if len(teacher) != len(response):
            raise ValueError(f"teacher_logprobs ({len(teacher)}) != response length ({len(response)})")
        mask = (np.ones(len(response), bool) if exp.action_mask is None
                else np.asarray(exp.action_mask, dtype=bool))
        return {"prompt": tokens[:exp.prompt_length], "response": response,
                "response_mask": mask, "teacher_logprobs": teacher}

    async def train_step(self, batch_exps: List[Experience]) -> Dict:
        rows = [self._to_row(e) for e in batch_exps]
        max_resp = self.config.model.max_response_tokens
        for r in rows:  # responses are generated with max_tokens=max_response_tokens
            if len(r["response"]) > max_resp:
                raise ValueError(f"response of {len(r['response'])} tokens > max_response_tokens={max_resp}")
        metrics = await self.worker.train_step.remote(rows)
        # MultiTurnOpdAdvantage's trajectory metrics: per-turn KL (old - teacher) summed per run.
        row_kl = metrics.pop("_row_kl")
        traj = defaultdict(float)
        for e, kl in zip(batch_exps, row_kl):
            traj[(e.eid.batch, e.eid.task, e.eid.run)] += kl
        if traj:
            vals = list(traj.values())
            metrics["kl/trajectory_mean"] = float(np.mean(vals))
            metrics["kl/trajectory_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        self._train_step_num += 1
        return metrics

    def _write_hf(self, output_dir: str) -> str:
        return ray.get(self.worker.save_hf.remote(output_dir))

    def save_state_dict(self) -> None:
        """Checkpoint used only for weight sync (removed by the Synchronizer once superseded)."""
        if self.train_step_num == self._last_state_dict_step:
            return
        self._write_hf(sync_checkpoint_dir(self.config, self.train_step_num))
        self._last_state_dict_step = self.train_step_num
        with open(self.local_latest_state_dict_iteration, "w") as f:
            f.write(str(self.train_step_num))

    def save_checkpoint(self, block_until_saved: bool = False, save_as_hf: bool = False) -> None:
        """Full checkpoint (HF weights; optimizer state is not saved yet)."""
        if self.train_step_num == self._last_checkpoint_step:
            return
        step_dir = os.path.join(self.default_local_dir, f"global_step_{self.train_step_num}")
        full_dir = hf_checkpoint_dir(self.default_local_dir, self.train_step_num)
        if not os.path.exists(os.path.join(full_dir, "model.safetensors")):
            self._write_hf(full_dir)
        # The worker created step_dir on its own host; over NFS this host may still hold a
        # cached "does not exist" for it. mkdir goes to the server and refreshes the view.
        os.makedirs(step_dir, exist_ok=True)
        with open(os.path.join(step_dir, ".full_checkpoint"), "w") as f:
            f.write("")
        with open(self.local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.train_step_num))
        self._last_checkpoint_step = self.train_step_num

    def sync_weight(self) -> None:
        raise NotImplementedError("tunix trainer supports sync_method: checkpoint only")

    def upload_state_dict(self) -> None:
        raise NotImplementedError("tunix trainer supports sync_method: checkpoint only")

