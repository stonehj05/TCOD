# Source before running on a single v4 host (4 local chips). Unset these for
# full-slice multi-host runs, where JAX discovers all 16 chips itself.
export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1
export TPU_PROCESS_BOUNDS=1,1,1
export TPU_VISIBLE_CHIPS=0,1,2,3
export TPU_PROCESS_PORT=8476
export TPU_PROCESS_ADDRESSES=localhost:8476
export HF_HUB_ENABLE_HF_TRANSFER=1
