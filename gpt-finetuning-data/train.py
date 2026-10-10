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

from contextlib import nullcontext

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

out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
max_lr = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_steps = 2000 # how many steps to warm up for
max_steps = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
# This section allows us to override the values defined in the
# variables section above. If we have a file named `configurator.py`
# and any of the variables defined above are also defined in that file,
# the value from `configurator.py` takes priority. The value defined
# in this file is then ignored and replaced by the value from `configurator.py`.

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------



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
    sequence_length=1024,
    process_rank=ddp_rank,
    num_processes=ddp_world_size,
    split='train',
)
val_loader = ConversationDataLoader(
    batch_size=B,
    sequence_length=1024,
    process_rank=ddp_rank,
    num_processes=ddp_world_size,
    split='val',
)




if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = get_device() # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
