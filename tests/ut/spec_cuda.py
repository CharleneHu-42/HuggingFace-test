import torch

# torch.use_deterministic_algorithms(True)
DEVICE_NAME = "cuda"

MANUAL_SEED_FN = torch.cuda.manual_seed
EMPTY_CACHE_FN = torch.cuda.empty_cache
DEVICE_COUNT_FN = torch.cuda.device_count
