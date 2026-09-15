# -----------------------------------------------------------------------------
# Fine-tune GPT-2 Large (774M) for chat
#
# Single-GPU launch:
#     python train.py
#
# Multi-GPU DDP launch (e.g. 4 GPUs):
#     torchrun --standalone --nproc_per_node=4 train.py
# -----------------------------------------------------------------------------

import os
import math
import time

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from model import GPT, GPTConfig
from dataloader import ConversationDataLoader
from data_sanitizer.utils import enc, special_tokens



# =============================================================================
# 1. Utilities
# =============================================================================
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def mprint(*args, **kwargs):
    """Print only from the master process (rank 0)."""
    if master_process:
        print(*args, **kwargs)


# =============================================================================
# 2. Variables & Parameters
# =============================================================================
device          = get_device()
backend         = 'nccl' # 'nccl', 'gloo', etc.

# =============================================================================
# 3. DDP / device setup
# =============================================================================

ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank            = int(os.environ['RANK'])
    ddp_local_rank      = int(os.environ['LOCAL_RANK'])
    ddp_world_size      = int(os.environ['WORLD_SIZE'])
    device              = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process      = True
    seed_offset         = 0
    ddp_world_size      = 1
    ddp_rank            = 0
    ddp_local_rank      = 0
    ddp_world_size      = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
mprint(f"tokens per iteration will be: {tokens_per_iter:,}")

# =============================================================================
# 5. Learning-rate schedule (cosine decay with linear warm-up)
# =============================================================================

def get_lr(step: int) -> float:
    # 1. Linear warm-up
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    # 2. Past decay horizon → hold at min_lr
    if step > max_steps:
        return min_lr
    # 3. Cosine decay
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0.0 <= decay_ratio <= 1.0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # 1 → 0
    return min_lr + coeff * (max_lr - min_lr)


# =============================================================================
# 6. Data loaders
# =============================================================================

train_loader = ConversationDataLoader(
    batch_size=B,
    sequence_length=T,
    process_rank=ddp_rank,
    num_processes=ddp_world_size,
    split='train',
)
val_loader = ConversationDataLoader(
    batch_size=B,
    sequence_length=T,
    process_rank=ddp_rank,
    num_processes=ddp_world_size,
    split='val',
)
