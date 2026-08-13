"""
=============================================================================
Distributed / torchrun test for ConversationDataLoader
=============================================================================

HOW TO RUN
----------
Always run from the PROJECT ROOT (gpt-finetuning-data/):

  # Single-process (CPU, no torchrun – quick sanity check):
  python tests/test_dataloader_distributed.py

  # 2 processes – mirrors a 2-GPU DDP setup on a single machine:
  torchrun --standalone --nproc_per_node=2 tests/test_dataloader_distributed.py

  # 4 processes:
  torchrun --standalone --nproc_per_node=4 tests/test_dataloader_distributed.py

  # Verbose shard-level output:
  torchrun --standalone --nproc_per_node=2 tests/test_dataloader_distributed.py --verbose

No GPU required – uses gloo backend (CPU).

WHAT IS TESTED
--------------
Section 1  – Construction & startup position per rank
Section 2  – First batch shape, dtype, device, token range, causal shift
Section 3  – Cross-rank no-overlap  (THE KEY DISTRIBUTED CHECK)
             Each rank records (shard_idx, start, end) for every batch it reads.
             rank-0 gathers all windows via dist.all_gather_object and checks
             that no two ranks ever read overlapping token windows.
Section 4  – Within-rank no-duplicate positions over a full epoch
Section 5  – Pool exhaustion: every shard is visited, no exception raised
Section 6  – reset() determinism: same batch after reset across ranks
Section 7  – Step-size: position advances by B*T*WORLD_SIZE each call
Section 8  – Raw shard bytes match what the batch contains (rank-0 only)
Section 9  – Tokenizer sanity (rank-0 only)
Section 10 – Val split construction works for every rank
=============================================================================
"""

import os
import sys
import argparse
from collections import Counter

import numpy as np
import torch
import torch.distributed as dist

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataloader import ConversationDataLoader, load_shard_tokens
from data_sanitizer.utils import enc, special_tokens

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--verbose", action="store_true")
ARGS, _ = parser.parse_known_args()

# ── Distributed init ──────────────────────────────────────────────────────────
# torchrun injects RANK / LOCAL_RANK / WORLD_SIZE into the environment.
# Plain `python` has none of those → single-process fallback.
TORCHRUN   = "RANK" in os.environ and "WORLD_SIZE" in os.environ

if TORCHRUN:
    dist.init_process_group(backend="gloo")   # gloo = CPU-safe
    RANK       = dist.get_rank()
    WORLD_SIZE = dist.get_world_size()
else:
    RANK       = 0
    WORLD_SIZE = 1

IS_MASTER = (RANK == 0)

# ── Config ────────────────────────────────────────────────────────────────────
BATCH_SIZE  = 2
SEQ_LEN     = 1024
BT          = BATCH_SIZE * SEQ_LEN          # tokens per batch step per rank
SHARDS_ROOT = os.path.join(PROJECT_ROOT, "shards")
VOCAB_SIZE  = enc.n_vocab


# =============================================================================
# Helpers
# =============================================================================

def log(msg: str):
    """Print with rank prefix; always flushed so lines don't interleave."""
    print(f"[rank {RANK}/{WORLD_SIZE}] {msg}", flush=True)


def vlog(msg: str):
    """Verbose-only log."""
    if ARGS.verbose:
        log(msg)


def barrier(section_title: str = ""):
    """Synchronise all ranks; master prints a section header."""
    if TORCHRUN:
        dist.barrier()
    if IS_MASTER and section_title:
        print(f"\n{'─'*62}\n  {section_title}\n{'─'*62}", flush=True)


def all_gather_obj(obj):
    """Gather any picklable Python object from every rank onto all ranks."""
    if not TORCHRUN:
        return [obj]
    bucket = [None] * WORLD_SIZE
    dist.all_gather_object(bucket, obj)
    return bucket


def compute_full_epoch_steps(split: str, nproc: int) -> int:
    """
    Calculate exactly how many next_batch() calls rank-0 must make to drain
    every shard at least once.  Uses actual shard sizes from disk.
    """
    shard_dir = os.path.join(SHARDS_ROOT, split)
    total = 0
    for fname in sorted(os.listdir(shard_dir)):
        if fname.startswith(split) and fname.endswith(".npy"):
            n    = len(np.load(os.path.join(shard_dir, fname)))
            step = BT * nproc
            total += max(1, n // step)
    return total + 20   # safety margin


# ── Test registry ─────────────────────────────────────────────────────────────
_results: dict[str, tuple[bool, str]] = {}


def record(name: str, passed: bool, detail: str = ""):
    """Record one test result and print it immediately with PASS/FAIL."""
    _results[name] = (passed, detail)
    tag = "PASS" if passed else "FAIL"
    msg = f"  [{tag}] {name}"
    if detail and (not passed or ARGS.verbose):
        msg += f"  ← {detail}"
    log(msg)


# =============================================================================
# SECTION 1 – Construction & starting-position check
# =============================================================================
barrier("SECTION 1 – Construction & starting position")

try:
    loader = ConversationDataLoader(
        batch_size=BATCH_SIZE,
        sequence_length=SEQ_LEN,
        process_rank=RANK,
        num_processes=WORLD_SIZE,
        split="train",
        data_root=SHARDS_ROOT,
    )
    record("s1_construction_no_exception", True)
except Exception as exc:
    record("s1_construction_no_exception", False, str(exc))
    if TORCHRUN:
        dist.destroy_process_group()
    sys.exit(1)

# Each rank must start at its own non-overlapping position: rank * B * T
expected_start = RANK * BT
record("s1_starting_position_is_rank_times_BT",
       loader.current_position == expected_start,
       f"got={loader.current_position} expected={expected_start}")

record("s1_shard_files_found",  len(loader.shard_files) > 0)
record("s1_shard_files_sorted", loader.shard_files == sorted(loader.shard_files))

stats = loader.get_stats()
record("s1_get_stats_keys_present",
       {"split","num_shards","total_tokens","shard_info","avg_tokens_per_shard"}.issubset(stats))
record("s1_total_tokens_positive", stats["total_tokens"] > 0)

vlog(f"  shards found : {len(loader.shard_files)}")
vlog(f"  total tokens : {stats['total_tokens']:,}")


# =============================================================================
# SECTION 2 – First batch: shape / dtype / device / range / causal shift
# =============================================================================
barrier("SECTION 2 – First batch properties")

inp, tgt = loader.next_batch()

record("s2_input_shape",  tuple(inp.shape) == (BATCH_SIZE, SEQ_LEN), str(tuple(inp.shape)))
record("s2_target_shape", tuple(tgt.shape) == (BATCH_SIZE, SEQ_LEN), str(tuple(tgt.shape)))
record("s2_input_dtype_long",  inp.dtype == torch.long, str(inp.dtype))
record("s2_target_dtype_long", tgt.dtype == torch.long, str(tgt.dtype))
record("s2_input_on_cpu",  inp.device.type == "cpu", str(inp.device))
record("s2_target_on_cpu", tgt.device.type == "cpu", str(tgt.device))

in_range = bool(
    (inp >= 0).all() and (inp < VOCAB_SIZE).all() and
    (tgt >= 0).all() and (tgt < VOCAB_SIZE).all()
)
record("s2_token_ids_in_valid_range", in_range, f"vocab_size={VOCAB_SIZE}")

# Causal shift: within each sequence row, inputs[b, 1:] must equal targets[b, :-1]
causal_ok = all(torch.equal(inp[b, 1:], tgt[b, :-1]) for b in range(BATCH_SIZE))
record("s2_causal_shift_inputs_1_eq_targets_minus1", causal_ok)

vlog(f"  inp[0,:5] = {inp[0,:5].tolist()}")
vlog(f"  tgt[0,:5] = {tgt[0,:5].tolist()}")


# =============================================================================
# SECTION 3 – Cross-rank no-overlap  ← THE KEY DISTRIBUTED CHECK
#
# Each rank records every (shard_idx, token_start, token_end) window it reads.
# rank-0 gathers all windows via dist.all_gather_object then checks that no
# two *different* ranks ever read overlapping token ranges within the same shard.
# =============================================================================
barrier("SECTION 3 – Cross-rank no-overlap (distributed check)")

STRESS_STEPS = max(
    500,
    compute_full_epoch_steps("train", WORLD_SIZE) // max(1, len(loader.shard_files)) + 100,
)

loader.reset()
my_windows: list[tuple[int, int, int]] = []   # (shard_idx, start, end)

for step in range(STRESS_STEPS):
    shard = loader.current_shard_idx
    start = loader.current_position
    end   = start + BT
    my_windows.append((shard, start, end))
    loader.next_batch()
    if ARGS.verbose and step < 3:
        vlog(f"  step {step}: shard={shard} pos=[{start}, {end})")

all_windows = all_gather_obj(my_windows)   # list[list[tuple]] – one list per rank

if IS_MASTER:
    # Flatten to (rank, shard, start, end)
    tagged = [
        (r, s, st, en)
        for r, wins in enumerate(all_windows)
        for (s, st, en) in wins
    ]

    overlaps = []
    for i in range(len(tagged)):
        ri, si, sti, eni = tagged[i]
        for j in range(i + 1, len(tagged)):
            rj, sj, stj, enj = tagged[j]
            if ri == rj:
                continue                  # same rank → expected
            if si == sj and sti < enj and stj < eni:
                overlaps.append((tagged[i], tagged[j]))

    passed = len(overlaps) == 0
    record("s3_cross_rank_no_overlapping_windows",
           passed,
           f"{len(overlaps)} overlapping window pairs" if not passed else "")

    if not passed:
        log("  First 3 overlapping pairs:")
        for a, b in overlaps[:3]:
            log(f"    rank{a[0]} shard{a[1]} [{a[2]},{a[3]}) overlaps "
                f"rank{b[0]} shard{b[1]} [{b[2]},{b[3]})")
    else:
        vlog(f"  Checked {len(tagged)} windows across {WORLD_SIZE} ranks – no overlaps.")

barrier()   # non-master ranks wait for master to finish the overlap check


# =============================================================================
# SECTION 4 – Within-rank: no duplicate (shard, position) across a full epoch
# =============================================================================
barrier("SECTION 4 – Within-rank no-duplicate positions")

loader.reset()
my_positions: list[tuple[int, int]] = []

for _ in range(STRESS_STEPS):
    my_positions.append((loader.current_shard_idx, loader.current_position))
    loader.next_batch()

counts = Counter(my_positions)
dupes  = {k: v for k, v in counts.items() if v > 1}
record("s4_within_rank_no_duplicate_shard_positions",
       len(dupes) == 0,
       f"{len(dupes)} duplicate (shard,pos) pairs" if dupes else "")


# =============================================================================
# SECTION 5 – Pool exhaustion: all shards visited, no exception
# =============================================================================
barrier("SECTION 5 – Pool exhaustion & shard cycling")

loader.reset()
n_shards       = len(loader.shard_files)
full_epoch_steps = compute_full_epoch_steps("train", WORLD_SIZE)
visited_shards = set()

try:
    for _ in range(full_epoch_steps):
        visited_shards.add(loader.current_shard_idx)
        loader.next_batch()
    record("s5_pool_exhaustion_no_exception", True)
except Exception as exc:
    record("s5_pool_exhaustion_no_exception", False, str(exc))

record("s5_all_shards_visited",
       visited_shards == set(range(n_shards)),
       f"visited={sorted(visited_shards)} expected={list(range(n_shards))}")

record("s5_shard_index_wraps_within_range",
       loader.current_shard_idx in range(n_shards),
       f"shard_idx={loader.current_shard_idx}")

vlog(f"  visited shards: {sorted(visited_shards)}")


# =============================================================================
# SECTION 6 – reset() determinism: rank's first batch is reproducible
# =============================================================================
barrier("SECTION 6 – reset() determinism")

loader.reset()
inp_epoch1, tgt_epoch1 = loader.next_batch()

# Advance past all shards, then reset and pull first batch again
for _ in range(full_epoch_steps):
    loader.next_batch()

loader.reset()
inp_epoch2, tgt_epoch2 = loader.next_batch()

record("s6_reset_reproduces_first_input_batch",  torch.equal(inp_epoch1, inp_epoch2))
record("s6_reset_reproduces_first_target_batch", torch.equal(tgt_epoch1, tgt_epoch2))

# reset() called twice → same starting state
loader.reset()
pos1 = loader.current_position
shard1 = loader.current_shard_idx
loader.reset()
pos2 = loader.current_position
shard2 = loader.current_shard_idx
record("s6_reset_is_idempotent", pos1 == pos2 and shard1 == shard2)


# =============================================================================
# SECTION 7 – Step-size: position advances by B*T*WORLD_SIZE per call
# =============================================================================
barrier("SECTION 7 – Step-size & consecutive batch uniqueness")

loader.reset()
pos_a = loader.current_position
inp_a, tgt_a = loader.next_batch()
pos_b = loader.current_position
inp_b, tgt_b = loader.next_batch()

expected_step = BT * WORLD_SIZE
actual_step   = pos_b - pos_a

record("s7_step_size_is_BT_times_world_size",
       actual_step == expected_step,
       f"actual={actual_step} expected={expected_step}")

record("s7_consecutive_batches_differ",
       not torch.equal(inp_a, inp_b) and not torch.equal(tgt_a, tgt_b))


# =============================================================================
# SECTION 8 – Raw shard bytes match the batch tensor (rank-0 only)
# =============================================================================
barrier("SECTION 8 – Raw shard bytes match batch")

if IS_MASTER:
    loader.reset()
    raw_tokens     = load_shard_tokens(loader.shard_files[0])
    start          = loader.current_position
    expected_slice = raw_tokens[start : start + BT + 1]
    inp_raw, tgt_raw = loader.next_batch()

    flat_inp  = inp_raw.reshape(-1)
    last_tgt  = tgt_raw[-1, -1].unsqueeze(0)
    recon     = torch.cat([flat_inp, last_tgt])

    record("s8_raw_shard_slice_matches_batch", torch.equal(recon, expected_slice))
    vlog(f"  shard[{start}:{start+BT+1}] == cat(flat_inp, last_tgt): {torch.equal(recon, expected_slice)}")

barrier()   # let other ranks wait


# =============================================================================
# SECTION 9 – Tokenizer sanity (rank-0 only – avoids redundant work)
# =============================================================================
barrier("SECTION 9 – Tokenizer sanity")

if IS_MASTER:
    # Encode-decode roundtrip
    sample_text = "Hello, GPT fine-tuning!"
    decoded     = enc.decode(enc.encode(sample_text))
    record("s9_tokenizer_roundtrip", decoded == sample_text, repr(decoded))

    # All custom special tokens must have IDs >= base GPT-2 vocab size (50257)
    base_vocab = 50257
    bad = {k: v for k, v in special_tokens.items() if v < base_vocab}
    record("s9_special_tokens_above_base_vocab", len(bad) == 0, str(bad))

    # Each special token encodes to exactly one ID
    for tok_name, tok_id in special_tokens.items():
        ids = enc.encode(tok_name, allowed_special="all")
        record(f"s9_{tok_name.strip('<|>')}_encodes_to_single_id",
               len(ids) == 1 and ids[0] == tok_id,
               f"ids={ids}")

    # BOS and EOT must appear in training shards
    first_arr = np.load(loader.shard_files[0]).astype(np.int32)
    record("s9_BOS_present_in_shard_0", int(special_tokens["<|BOS|>"]) in first_arr)
    record("s9_EOT_present_in_shard_0", int(special_tokens["<|EOT|>"]) in first_arr)

    # uint16 dtype on disk
    for path in loader.shard_files:
        arr = np.load(path)
        if arr.dtype != np.uint16:
            record("s9_shard_dtype_uint16", False,
                   f"{os.path.basename(path)} has dtype {arr.dtype}")
            break
    else:
        record("s9_shard_dtype_uint16", True)

barrier()


# =============================================================================
# SECTION 10 – Val split works on every rank
# =============================================================================
barrier("SECTION 10 – Val split")

try:
    val_loader = ConversationDataLoader(
        batch_size=BATCH_SIZE,
        sequence_length=SEQ_LEN,
        process_rank=RANK,
        num_processes=WORLD_SIZE,
        split="val",
        data_root=SHARDS_ROOT,
    )
    v_inp, v_tgt = val_loader.next_batch()
    record("s10_val_construction_no_exception", True)
    record("s10_val_batch_shape_correct",
           tuple(v_inp.shape) == (BATCH_SIZE, SEQ_LEN),
           str(tuple(v_inp.shape)))
except Exception as exc:
    record("s10_val_construction_no_exception", False, str(exc))
    record("s10_val_batch_shape_correct",       False, "loader failed")


# =============================================================================
# FINAL REPORT – rank-0 collects results from all ranks and prints summary
# =============================================================================
barrier("FINAL REPORT")

all_results = all_gather_obj(_results)

if IS_MASTER:
    print("\n" + "═" * 66)
    print(f"  Distributed DataLoader Test Report")
    print(f"  nproc={WORLD_SIZE}  B={BATCH_SIZE}  T={SEQ_LEN}  "
          f"stress_steps={STRESS_STEPS}")
    print("═" * 66)

    # Merge: a test FAILS globally if it fails on ANY rank
    merged: dict[str, tuple[bool, str]] = {}
    for rank_result in all_results:
        for name, (ok, detail) in rank_result.items():
            if name not in merged:
                merged[name] = (ok, detail)
            elif not ok:
                merged[name] = (ok, detail)   # one bad rank poisons the test

    total  = len(merged)
    n_pass = sum(1 for ok, _ in merged.values() if ok)
    n_fail = total - n_pass

    print(f"\n  {'Test':<52} Result")
    print(f"  {'─'*52} ──────")
    for name, (ok, detail) in merged.items():
        icon = "✅" if ok else "❌"
        line = f"  {name:<52} {icon} {'PASS' if ok else 'FAIL'}"
        if not ok and detail:
            line += f"\n{'':56}↳ {detail}"
        print(line)

    print(f"\n  {'─'*58}")
    print(f"  Total: {total}   PASSED: {n_pass}   FAILED: {n_fail}")
    if n_fail == 0:
        print("  🎉  ALL TESTS PASSED – dataloader is ready for cloud GPU training!")
    else:
        print("  ⚠️   SOME TESTS FAILED – fix before moving to GPU training.")
    print("═" * 66 + "\n")

if TORCHRUN:
    dist.destroy_process_group()

# Exit with non-zero if any test failed (useful in CI)
all_passed = all(ok for ok, _ in _results.values())
sys.exit(0 if all_passed else 1)
