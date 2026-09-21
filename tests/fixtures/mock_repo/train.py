"""Mock train.py: deterministic stand-in for GPU training (G19 dry-run).

val_bpb is a seeded function of this file's own source hash, so any applied
patch changes the metric deterministically. Behavior can be overridden with
env vars (whitelisted by the runner): MOCK_TRAIN_BEHAVIOR in
{ok,crash,oom,nan,timeout}, MOCK_VAL_BPB=<float>.
Contains the same PROTECTED lines as the real train.py so Guard checks behave
identically against both.
"""
import hashlib
import os
import sys
import time

from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb  # PROTECTED
import prepare

torch = prepare.torch
torch.manual_seed(42)  # PROTECTED

DEVICE_BATCH_SIZE = 64
PREFLIGHT_STEPS = int(os.environ.get("AUTORESEARCH_PREFLIGHT_STEPS", "0"))
BEHAVIOR = os.environ.get("MOCK_TRAIN_BEHAVIOR", "ok")


def _seeded_bpb():
    with open(__file__, "rb") as f:
        digest = int(hashlib.sha256(f.read()).hexdigest(), 16)
    return 0.9 + (digest % 10000) / 10000.0 * 0.4


if BEHAVIOR == "crash":
    print("Traceback (most recent call last):", file=sys.stderr)
    print("RuntimeError: simulated crash", file=sys.stderr)
    sys.exit(1)
if BEHAVIOR == "oom":
    print("torch.OutOfMemoryError: CUDA out of memory", file=sys.stderr)
    sys.exit(1)
if BEHAVIOR == "nan":
    print("FAIL")
    sys.exit(1)
if BEHAVIOR == "timeout":
    time.sleep(3600)

model = None
tokenizer = Tokenizer()
step = 0
total_training_time = 0.0
while True:
    step += 1
    total_training_time += 0.001
    if step > 10 and total_training_time >= TIME_BUDGET:  # PROTECTED
        break
    if PREFLIGHT_STEPS and step >= PREFLIGHT_STEPS:
        print("PREFLIGHT_OK", flush=True)
        raise SystemExit(0)

val = float(os.environ.get("MOCK_VAL_BPB", _seeded_bpb()))
prepare._BPB_OVERRIDE = val
val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)  # PROTECTED

print("---")
print(f"val_bpb:          {val_bpb:.6f}")  # PROTECTED
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {total_training_time + 5:.1f}")
print(f"peak_vram_mb:     {1234.5:.1f}")
print(f"num_steps:        {step}")
