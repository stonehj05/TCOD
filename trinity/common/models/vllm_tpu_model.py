"""vLLM-TPU rollout model (engine_type: vllm_tpu).

Same `chat` / `generate` / `logprobs` behaviour as `vLLMRolloutModel` (inherited
unchanged, including the client-side prompt-truncation handling), running on
vLLM's TPU backend (vllm-tpu / tpu_inference). Differences from the GPU class:

- The engine is built without the CUDA/NCCL worker extension and vLLM patches;
  `TPUWorkerExtension` adds checkpoint-based weight loading instead.
- Weight sync is `sync_method: checkpoint` only: the synchronizer's state-dict
  payload for a TPU trainer is the path of a Hugging Face safetensors
  checkpoint, which every engine worker loads in place.
- Chip placement comes from Ray's `TPU` resource (Ray sets TPU_VISIBLE_CHIPS
  for the actor); see `create_inference_models`.
"""

import asyncio
import os

from trinity.common.config import InferenceModelConfig
from trinity.common.models.model import BaseInferenceModel
from trinity.common.models.vllm_model import vLLMRolloutModel


METRICS_BASE_PORT = 8431  # libtpu's default runtime-metrics port (what `tpu-info` reads)


def set_tpu_metrics_port() -> None:
    """Give this process its own libtpu runtime-metrics port: 8431 + its first TPU chip.

    Several engines share a VM (one per chip); by default all of them serve metrics on 8431,
    so tools like `tpu-info` only reach one. Must run before the TPU backend initializes;
    vLLM's EngineCore subprocess inherits the environment.
    """
    chips = os.environ.get("TPU_VISIBLE_CHIPS", "")  # set by Ray for the actor's chips
    if not chips or "runtime_metric_service_port" in os.environ.get("LIBTPU_INIT_ARGS", ""):
        return
    port = METRICS_BASE_PORT + int(chips.split(",")[0])
    os.environ["LIBTPU_INIT_ARGS"] = (
        os.environ.get("LIBTPU_INIT_ARGS", "") + f" --runtime_metric_service_port={port}"
    ).strip()



def _text_only_args(model_path: str) -> dict:
    """Gemma 4 checkpoints carry vision / audio towers; our games are text-only, so serve without
    them (as tested on TPU v4, 2026-10-26). Empty for every other model."""
    import json
    import os

    cfg = os.path.join(model_path, "config.json")
    if os.path.isfile(cfg):
        with open(cfg) as f:
            if json.load(f).get("architectures", [""])[0].startswith("Gemma4"):
                return {"limit_mm_per_prompt": {"image": 0, "audio": 0}}
    return {}

class vLLMTPURolloutModel(vLLMRolloutModel):
    def __init__(self, config: InferenceModelConfig) -> None:
        BaseInferenceModel.__init__(self, config)  # skip the CUDA-specific engine setup
        set_tpu_metrics_port()
        # Gemma 4 MoE (26B-A4B): vLLM model path + gmm_v2 tile patch (tpu_env.moe_env_vars). Resolved here, on
        # the engine's own host (the driver's host may hold no copy of the weights). The env reaches a spawned
        # engine core (PYTHONPATH -> sitecustomize); the in-process import covers a forked one.
        from trinity.common.models.tpu_env import moe_env_vars

        moe_env = moe_env_vars(config.model_path)
        if moe_env:
            import importlib.util

            os.environ.update(moe_env)
            site = os.path.join(moe_env["PYTHONPATH"].split(":")[0], "sitecustomize.py")
            spec = importlib.util.spec_from_file_location("tpu_moe_sitecustomize", site)
            spec.loader.exec_module(importlib.util.module_from_spec(spec))

        import vllm
        from vllm.sampling_params import RequestOutputKind

        # vLLM-TPU cannot serialize the worker's live JAX arrays over collective_rpc
        # otherwise; also needed for the in-process weight loader.
        os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
        self.use_v1 = True
        self.vllm_version = None
        self.logprobs_no_prefix_cache = True
        self.default_sampling_params = vllm.SamplingParams(
            n=1,
            temperature=config.temperature,
            max_tokens=config.max_response_tokens,
            min_tokens=config.min_response_tokens,
            skip_special_tokens=True,
            include_stop_str_in_output=False,
            output_kind=RequestOutputKind.FINAL_ONLY,
            logprobs=config.logprobs,
            top_p=config.top_p,
            top_k=config.top_k,
            ignore_eos=config.ignore_eos,
        )
        self.ray_namespace = config.ray_namespace
        self.request_id = 0
        self.enable_lora = False
        self.default_lora_path = None
        engine_args = vllm.AsyncEngineArgs(
            model=config.model_path,
            enforce_eager=config.enforce_eager,
            worker_extension_cls="trinity.common.models.vllm_tpu_worker.TPUWorkerExtension",
            tensor_parallel_size=config.tensor_parallel_size,
            seed=config.seed,
            max_model_len=config.max_model_len,
            enable_prefix_caching=config.enable_prefix_caching,
            enable_chunked_prefill=config.enable_chunked_prefill,
            dtype=config.dtype,
            trust_remote_code=True,
            gpu_memory_utilization=config.gpu_memory_utilization,
            override_generation_config={
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "max_new_tokens": config.max_response_tokens,
                "repetition_penalty": config.repetition_penalty,
            },
            disable_log_stats=True,
            logprobs_mode="processed_logprobs",
            enable_log_requests=config.enable_log_requests,
            reasoning_parser=config.reasoning_parser,
            async_scheduling=False,
            **_text_only_args(config.model_path),
        )
        self.async_llm = vllm.AsyncLLMEngine.from_engine_args(engine_args)
        self.processor = None
        self.state_dict_meta = None
        self.model_version = 0
        self.api_server_host = None
        self.api_server_port = None
        self.api_server = None
        self._prepared = False
        self.async_lock = asyncio.Lock()
        self._synchronizer = None

    async def _initialize_tokenizer(self):
        if self.tokenizer is None:
            tokenizer = self.async_llm.get_tokenizer()  # sync in newer vLLM, async in older
            self.tokenizer = await tokenizer if asyncio.iscoroutine(tokenizer) else tokenizer
        self.tokenizer.truncation_side = "left"

    def _create_sampling_params(self, **kwargs):
        params = super()._create_sampling_params(**kwargs)
        if params.logprobs == 0:
            # vLLM-TPU returns no logprobs at all for logprobs=0 ("sampled token
            # only" on GPU). With 1 it returns the sampled token first, followed by
            # the top-1, which is what `generate` reads.
            if params is self.default_sampling_params:
                params = params.clone()
            params.logprobs = 1
        return params

    async def prepare(self) -> None:
        async with self.async_lock:
            if self._prepared:
                return
            if self.config.enable_openai_api:
                await self.run_api_server()
            self._prepared = True

    async def sync_model(self, model_version: int, weight_source: str = "student") -> int:
        """Load the checkpoint the synchronizer published for `model_version`."""
        from trinity.manager.synchronizer import Synchronizer

        if self._synchronizer is None:
            self._synchronizer = Synchronizer.get_actor(namespace=self.ray_namespace)
        checkpoint_dir, version = await self._synchronizer.get_model_state_dict.remote(weight_source)
        if not isinstance(checkpoint_dir, str):
            raise RuntimeError(
                "vllm_tpu engines only support sync_method=checkpoint with a TPU trainer "
                f"(expected a checkpoint path, got {type(checkpoint_dir).__name__})."
            )
        await self.async_llm.reset_prefix_cache()
        n = await self._collective_rpc("update_weight_from_checkpoint", args=(checkpoint_dir,))
        self.logger.info(f"Loaded {n} tensors from {checkpoint_dir} (model version {version}).")
        self.model_version = model_version
        return model_version

    async def prompt_token_ids(self, messages, add_generation_prompt: bool = True):
        """`messages` rendered with this engine's chat template and thinking setting, as token ids (used by
        the deferred teacher gate to compare the teacher's generation prompt with the student's)."""
        if self.tokenizer is None:
            await self._initialize_tokenizer()
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt,
            chat_template=self.chat_template or self.tokenizer.get_chat_template(),
            enable_thinking=self.enable_thinking,
        )
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    async def init_process_group(self, *args, **kwargs):
        raise NotImplementedError("NCCL weight sync is not available on TPU; use sync_method: checkpoint.")
