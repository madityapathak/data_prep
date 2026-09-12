# -----------------------------------------------------------------------------
# Fine-tune GPT-2 Large (774M) for chat
#
# Single-GPU launch:
#     python train_model.py
#
# Multi-GPU DDP launch (e.g. 4 GPUs):
#     torchrun --standalone --nproc_per_node=4 train_model.py
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
# 1. DDP / device setup
# =============================================================================

ddp = int(os.environ.get('RANK', -1)) != -1

if ddp:
    init_process_group(backend='nccl')
    ddp_rank       = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device         = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank       = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"

device_type = "cuda" if device.startswith("cuda") else "cpu"


# =============================================================================
# 2. Helper: master-process-only print
# =============================================================================

def mprint(*args, **kwargs):
    """Print only from the master process (rank 0)."""
    if master_process:
        print(*args, **kwargs)


# =============================================================================
# 3. Reproducibility seeds
# =============================================================================

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)


# =============================================================================
# 4. Training hyper-parameters
# =============================================================================

# --- Batch / sequence dims ---
total_batch_size = 524288   # ~0.5M tokens per gradient update (2**19)
B = 4                       # micro-batch size (sequences per GPU step)
T = 1024                    # sequence / context length

assert total_batch_size % (B * T * ddp_world_size) == 0, \
    "total_batch_size must be divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

# --- Learning-rate schedule ---
max_lr       = 6e-5         # lower than pre-training; fine-tuning needs smaller steps
min_lr       = max_lr * 0.1
warmup_steps = 100
max_steps    = 5000         # adjust to your dataset size / desired epochs

# --- Evaluation & checkpointing ---
eval_interval       = 250   # run validation every N steps
eval_val_steps      = 20    # number of validation batches to average
checkpoint_interval = 1000  # save a checkpoint every N steps


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

mprint(f"total desired batch size : {total_batch_size:,} tokens")
mprint(f"gradient accumulation steps: {grad_accum_steps}")
mprint(f"train shards : {len(train_loader.shard_files)}")
mprint(f"val   shards : {len(val_loader.shard_files)}")


# =============================================================================
# 7. Model — load GPT-2 Large (774M) from pre-trained weights
# =============================================================================

torch.set_float32_matmul_precision('high')

mprint("Loading gpt2-large weights from HuggingFace ...")
model = GPT.from_pretrained("gpt2-large")
model.to(device)

use_compile = False   # torch.compile can conflict with generation; enable when stable
if use_compile:
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model   # unwrapped model (for saving checkpoints)


# =============================================================================
# 8. Optimizer
# =============================================================================

optimizer = raw_model.configure_optimizers(
    weight_decay=0.1,
    learning_rate=max_lr,
    device_type=device_type,
)


# =============================================================================
# 9. Logging / checkpoint directory
# =============================================================================

log_dir  = "log"
log_file = os.path.join(log_dir, "log.txt")

if master_process:
    os.makedirs(log_dir, exist_ok=True)
    with open(log_file, "w") as f:
        pass   # clear the log file


# =============================================================================
# 10. Chat-style generation helper (used for qualitative eval during training)
# =============================================================================

def generate_chat_response(
    prompt: str,
    max_new_tokens: int = 200,
    top_k: int = 50,
    temperature: float = 0.8,
) -> str:
    """
    Wrap a user prompt in the conversation format and greedily/sample decode
    an ASSISTANT response.  Runs only on the master process.

    Format:  <|BOS|><|USER|>{prompt}<|EOS|><|ASSISTANT|>
    """
    bos  = special_tokens["<|BOS|>"]
    user = special_tokens["<|USER|>"]
    eos  = special_tokens["<|EOS|>"]
    asst = special_tokens["<|ASSISTANT|>"]

    # Encode the prompt text (no special tokens inside the user message)
    prompt_ids = enc.encode(prompt, allowed_special=set())

    # Build the full prefix sequence
    prefix = [bos, user] + prompt_ids + [eos, asst]
    tokens = torch.tensor(prefix, dtype=torch.long, device=device).unsqueeze(0)

    model.eval()
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42)

    eot_id = special_tokens["<|EOT|>"]

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Truncate to model's block size if needed
            ctx = tokens if tokens.size(1) <= raw_model.config.block_size else \
                  tokens[:, -raw_model.config.block_size:]

            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, _ = model(ctx)

            logits = logits[:, -1, :] / temperature          # (1, vocab_size)
            top_k_logits, top_k_ids = torch.topk(logits, top_k, dim=-1)
            probs = F.softmax(top_k_logits, dim=-1)
            ix    = torch.multinomial(probs, 1, generator=sample_rng)
            next_token = torch.gather(top_k_ids, -1, ix)      # (1, 1)

            tokens = torch.cat((tokens, next_token), dim=1)

            # Stop at end-of-turn
            if next_token.item() == eot_id or next_token.item() == eos:
                break

    # Decode only the newly generated assistant tokens
    generated_ids = tokens[0, len(prefix):].tolist()
    return enc.decode(generated_ids, errors='replace')


# =============================================================================
# 11. Training loop
# =============================================================================

mprint(f"\nStarting fine-tuning for {max_steps} steps ...")

for step in range(max_steps):
    t0 = time.time()
    is_last_step = (step == max_steps - 1)

    # -------------------------------------------------------------------------
    # 11a. Validation
    # -------------------------------------------------------------------------
    if step % eval_interval == 0 or is_last_step:
        model.eval()
        val_loader.reset()

        with torch.no_grad():
            val_loss_accum  = 0.0
            for _ in range(eval_val_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    _, loss = model(x, y)
                val_loss_accum += loss.detach() / eval_val_steps

        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)

        mprint(f"step {step:5d} | val loss: {val_loss_accum.item():.4f}")

        if master_process:
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")

    # -------------------------------------------------------------------------
    # 11b. Checkpoint
    # -------------------------------------------------------------------------
    if master_process and step > 0 and (step % checkpoint_interval == 0 or is_last_step):
        checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
        checkpoint = {
            'model':    raw_model.state_dict(),
            'config':   raw_model.config,
            'step':     step,
            'val_loss': val_loss_accum.item() if step % eval_interval == 0 else None,
            'optimizer': optimizer.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)
        mprint(f"Saved checkpoint → {checkpoint_path}")

    # -------------------------------------------------------------------------
    # 11c. Qualitative chat generation sample (master process only)
    # -------------------------------------------------------------------------
    if master_process and step > 0 and step % eval_interval == 0 and not use_compile:
        response = generate_chat_response(
            prompt="What is the capital of France?",
            max_new_tokens=100,
        )
        mprint(f"\n--- generation sample (step {step}) ---")
        mprint(f"Response: {response}")
        mprint(f"--- end sample ---\n")

    # -------------------------------------------------------------------------
    # 11d. Training step (with gradient accumulation)
    # -------------------------------------------------------------------------
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0

    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)

        # In DDP, only sync gradients on the final micro-step
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)

        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            _, loss = model(x, y)

        # Scale loss for gradient accumulation (keeps effective loss as a mean)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()

    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    # Gradient clipping to prevent exploding gradients
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # Apply the cosine-decayed learning rate
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()

    if device_type == "cuda":
        torch.cuda.synchronize()

    t1 = time.time()
    dt = t1 - t0

    tokens_processed = B * T * grad_accum_steps * ddp_world_size
    tokens_per_sec   = tokens_processed / dt

    mprint(
        f"step {step:5d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | "
        f"norm: {norm:.4f} | dt: {dt * 1000:.1f}ms | tok/s: {tokens_per_sec:.0f}"
    )

    if master_process:
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")


# =============================================================================
# 12. Cleanup
# =============================================================================

if ddp:
    destroy_process_group()

mprint("Training complete.")
