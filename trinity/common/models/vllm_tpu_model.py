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


class vLLMTPURolloutModel(vLLMRolloutModel):
    def __init__(self, config: InferenceModelConfig) -> None:
        BaseInferenceModel.__init__(self, config)  # skip the CUDA-specific engine setup

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

    async def init_process_group(self, *args, **kwargs):
        raise NotImplementedError("NCCL weight sync is not available on TPU; use sync_method: checkpoint.")
