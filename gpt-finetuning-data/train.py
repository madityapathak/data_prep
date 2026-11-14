"""
GPT-2 770M Fine-tuning Script for User-Assistant Conversational Model

This training script supports both single-GPU and distributed data parallel (DDP) training.

Single GPU example:
$ python train.py

DDP on multiple GPUs (4 GPUs, 1 node):
$ torchrun --standalone --nproc_per_node=4 train.py

DDP on multiple nodes:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=<IP> --master_port=1234 train.py
"""

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


if os.environ.get("SINGLE_GPU_DDP_TEST") == "1":
    os.environ["NCCL_MULTI_RANK_GPU_ENABLE"] = "1"

# FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated.
# Please use `torch.amp.GradScaler('cuda', args...)` instead.


# batch_size = 5  # Per-GPU batch size: 480 (global batch) / (12 gradient_accumulation_steps * 8 GPUs)
# max_iters = 673  # (323,956 training sequences // 480 global batch size) - 1
# warmup_iters = 34  # int((max_iters + 1) * 0.05)  # 5% of total training iterations
# lr_decay_iters = max_iters  # Cosine learning-rate decay completes at the final training iteration 

# =============================================================================
# Hyperparameters & Configuration
# =============================================================================

# I/O
out_dir = 'out'
log_dir = 'log'
eval_interval = 2000    # Run train/validation loss evaluation and checkpoint-saving checks every eval_interval (2000) training iterations.
# log_interval = 1
eval_iters = 200    # Number of batches evaluated per split (train and val); total samples evaluated = B * eval_iters.
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = True  # if True, always save a checkpoint after each eval
init_from = 'scratch' if os.environ.get("SINGLE_GPU_DDP_TEST") == "1" else "gpt2-large"  # 'scratch' or 'resume' or 'gpt2*'

# wandb logging
wandb_log = False  # disabled by default
wandb_project = 'gpt2-finetuning'
wandb_run_name = 'gpt2-770m-chat'

# data
total_batch_size =  131072
batch_size = 8  # micro-batch size per GPU
block_size = 1024  # context length

# model - GPT-2 770M configuration (24 layers, 16 heads, 1280 embedding dim)
n_layer = 24
n_head = 16
n_embd = 1280
dropout = 0.1  # for fine-tuning, use 0.1-0.2 dropout to prevent overfitting
bias = False  # do we use bias inside LayerNorm and Linear layers?

# adamw optimizer
learning_rate = 3e-4  # max learning rate (lower than pretraining for fine-tuning)
max_iters = 2535  # total number of training iterations (total tokens in dataset // total_batch_size)
weight_decay = 1e-2  # Penalizes larger weights more than smaller ones by applying a proportional reduction.
                      # Weight decay: w_decay = w_old - (weight_decay * w_old)
                      # Updated weight: w_new = w_old - (learning_rate * gradient) - (weight_decay * w_old)
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0

# learning rate decay settings (cosine decay with warmup)
decay_lr = True  # whether to decay the learning rate
warmup_iters = 100  # how many steps to warm up for (4% of max_iters)
lr_decay_iters = 2540  # should be ~= max_iters 
min_lr = 3e-5  # minimum learning rate, should be ~= learning_rate/10

# system
backend = 'nccl'  # 'nccl', 'gloo', etc.
device = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True  # use PyTorch 2.0 to compile the model to be faster


# =============================================================================
# DDP Setup
# =============================================================================

ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])

    # Override GPU selection only for single-GPU testing
    gpu_id = 0 if os.environ.get("SINGLE_GPU_DDP_TEST") == "1" else ddp_local_rank

    device = f'cuda:{gpu_id}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed
    

    assert total_batch_size % (batch_size * block_size * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
    gradient_accumulation_steps = total_batch_size // (batch_size * block_size * ddp_world_size)
else:
    # Single GPU training
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
    ddp_rank = 0
    ddp_local_rank = 0

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
if master_process:
    print("gradient accumulation steps : ", gradient_accumulation_steps)
    print(f"tokens per iteration will be: {tokens_per_iter:,}")
    print(f"Training GPT-2 770M ({n_layer}L, {n_head}H, {n_embd}D)")


# =============================================================================
# Setup
# =============================================================================

if master_process:
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast

# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

torch.set_float32_matmul_precision('high')


# =============================================================================
# Data Loaders
# =============================================================================

train_loader = ConversationDataLoader(
    batch_size=batch_size,
    sequence_length=block_size,
    process_rank=ddp_rank,
    num_processes=ddp_world_size,
    split='train',
)

val_loader = ConversationDataLoader(
    batch_size=batch_size,
    sequence_length=block_size,
    process_rank=ddp_rank,
    num_processes=ddp_world_size,
    split='val',
)


def get_batch(split):
    if split == 'train':
        x, y = train_loader.next_batch()
    elif split == 'val':
        x, y = val_loader.next_batch()
    else:
        raise ValueError(f"Invalid split: {split}")

    # Cross entropy requires int64 targets.
    if y.dtype != torch.long:
        y = y.long()

    if x.dtype != torch.long:
        x = x.long()

    if device_type == 'cuda':
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)

    return x, y


# =============================================================================
# Model Initialization
# =============================================================================

# init these up here, can override if init_from='resume'
iter_num = 0
best_val_loss = 1e9

# Model arguments
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=50304,  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    dropout=dropout
)

if init_from == 'scratch':
    # Initialize a new model from scratch
    if master_process:
        print("Initializing a new model from scratch")
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
    
elif init_from.startswith('gpt2'):
    if master_process:
        print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # Initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    
    # Read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)

# Crop down the model block size if desired
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size  # so that the checkpoint will have the right value

model.to(device)

# Initialize a GradScaler. If enabled=False scaler is a no-op
# scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))

# Optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

checkpoint = None  # free up memory

# Compile the model
if compile:
    if master_process:
        print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0

# Wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[gpu_id])


# =============================================================================
# Training Utilities
# =============================================================================

@torch.no_grad()
def estimate_loss():
    """
    Estimate an arbitrarily accurate loss over either split using many batches.
    """
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_lr(it):
    """
    Learning rate decay scheduler (cosine with warmup).
    """
    # 1) Linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    
    # 2) If it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    
    # 3) In between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 1..0
    return min_lr + coeff * (learning_rate - min_lr)


# =============================================================================
# Logging
# =============================================================================

if wandb_log and master_process:
    import wandb
    config = {k: v for k, v in globals().items() 
              if not k.startswith('_') and isinstance(v, (int, float, bool, str))}
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass

# =============================================================================
# Training Loop
# =============================================================================

# Fetch the very first batch
X, Y = get_batch('train')

t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model  # unwrap DDP container if needed
running_mfu = -1.0

if master_process:
    print(f"Starting training at iteration {iter_num}")

while True:

    # Determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu * 100,  # convert to percentage
            })
        
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': {
                        'learning_rate': learning_rate,
                        'max_iters': max_iters,
                        'batch_size': batch_size,
                        'block_size': block_size,
                        'n_layer': n_layer,
                        'n_head': n_head,
                        'n_embd': n_embd,
                    },
                }
                if master_process:
                    print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    
    if iter_num == 0 and eval_only:
        break

    loss_accum = 0.0
    # Forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # In DDP training we only need to sync gradients at the last micro step
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps  # scale the loss to account for gradient accumulation
        loss_accum += loss.detach()
        # Immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        
        # Backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()

    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        
    # Clip the gradient
    norm = 0.0
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    # Step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    
    # Flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # Timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    try: 
        tokens_processed = batch_size * block_size * gradient_accumulation_steps * ddp_world_size
        step = iter_num
        tokens_per_sec = tokens_processed / dt
        if master_process:
            print("batch_size * block_size * gradient_accumulation_steps * ddp_world_size",batch_size, block_size, gradient_accumulation_steps, ddp_world_size)
            print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")
    except:
        pass
    
    iter_num += 1
    local_iter_num += 1

    # Termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()

if master_process:
    print(f"Training complete. Best validation loss: {best_val_loss:.4f}")