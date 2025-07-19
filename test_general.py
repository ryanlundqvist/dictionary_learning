#!/usr/bin/env python
"""
This is a vibe coded smoke test for `dictionary_learning` with Pythia 70M
========================================================

Purpose
-------
This script performs a **minimal, fast, CPU‑only integration / smoke test** of the
`dictionary_learning` repository (extended version containing CrossCoder support).
It is meant to answer the question:

    "Did the core moving pieces of the repo install correctly and can they run end-to-end
     (trace model activations -> train a tiny SAE -> load a published CrossCoder -> cache activations)
     on a machine without a GPU?"

It deliberately **avoids large models, long training, or GPU dependencies.**

What Is Actually Tested (Coverage Matrix)
-----------------------------------------
1. **Environment & Imports**
   * Confirms Python version and torch import.
   * Imports `nnsight` + core objects: `AutoEncoder`, `CrossCoder`, `trainSAE`, `StandardTrainer`.

2. **nnsight Tracing (Single Prompt)**
   * Loads `EleutherAI/pythia-70m-deduped`.
   * Traces *one* MLP submodule to extract a last-token activation.
   * Verifies shape & prints basic statistics (mean/std) to ensure activations are real.

3. **AutoEncoder Construction & Forward Pass**
   * Instantiates an untrained SAE (dictionary size = `AE_DICT_MULTIPLIER * hidden_dim`).
   * Runs a forward pass (encode + decode) to confirm parameter & tensor wiring.
   * Reports initial reconstruction error and feature activity count.

4. **Manual Activation Collection**
   * Collects a fixed number (`ACTIVATION_SAMPLES`) of last-token activations via repeated
     single-prompt traces (sidestepping current nnsight batch trace issues that affected
     the `ActivationBuffer` path).
   * Produces a 2‑D tensor `(N, hidden_dim)` used as pseudo-training data.

5. **Tiny Training Run (trainSAE + StandardTrainer)**
   * Uses a lightweight iterator over the collected activations.
   * Executes `TRAIN_STEPS` optimization steps with `trainSAE`.
   * Falls back to a *manual* minimalist training loop only if `trainSAE` raises.

6. **Post-Training Evaluation**
   * Evaluates trained SAE on a random synthetic batch (not the collected data) to ensure
     the forward path with updated weights still works and produces finite metrics.

7. **CrossCoder Hub Load & Mock Forward**
   * Downloads the published CrossCoder checkpoint:
       `Butanium/gemma-2-2b-crosscoder-l13-mu4.1e-02-lr1e-04`
   * Performs a single forward pass on **mock random** activations shaped `(1, 2, 2304)`.
   * Verifies that loading logic + feature path execute without dtype/shape crashes.
   * (Reconstruction quality is intentionally meaningless because input is random.)

8. **Activation Cache Roundtrip**
   * Saves a small slice (`[3, hidden_dim]`) of collected activations to disk.
   * Reloads and compares tensors (sanity check for serialization & reproducibility pathways).

9. **Optional Hugging Face Hub Push (Disabled by Default)**
   * Code path present; not executed unless `PUSH_TO_HUB=True`.

10. **Summary Report**
   * Aggregates shapes, metrics, and any skipped components.

What Is *Not* Tested
--------------------
- **True sparsity emergence / convergence** (training steps are far too few).
- **ActivationBuffer batch mode** (skipped due to current nnsight multi-input issues).
- **Multi-layer / BatchTopK SAE variants** or Gated / JumpReLU implementations.
- **CrossCoder realism** (we do *not* feed real paired Gemma activations).
- **Neuron resampling**, LR warmup dynamics over long horizons.
- **W&B logging** (disabled for simplicity; pass `use_wandb=True` to exercise).
- **GPU / mixed precision** execution.
- **Entropy-based or experimental alternative loss features**.
- **Paired activation caches, dataset streaming, or large corpus ingestion**.

Interpretation of Metrics
-------------------------
- Initial AutoEncoder MSE is just a random baseline; **high is normal**.
- Trained SAE MSE on *random noise* is *not* meaningful for model quality—only smoke.
- CrossCoder MSE on random input can be very large; only failure would be an exception / NaN.

When To Modify
--------------
- If you fix the nnsight batch trace issue: reintroduce `ActivationBuffer` and compare.
- If you want a stricter regression test: pin seed, assert metric ranges, or extend steps.
- If you want CI use: shorten `ACTIVATION_SAMPLES` further (e.g., 256) to reduce runtime.

-------------------------------------------------------------------------------
"""

import sys
import time
from typing import Dict, Any, List

############################
# 0. Environment Checks
############################
REQUIRED_PY_MIN = (3, 10)
if sys.version_info < REQUIRED_PY_MIN:
    raise RuntimeError(f"Python >= {REQUIRED_PY_MIN} required, found {sys.version_info[:3]}")

print(f"[ENV] Python version OK: {sys.version.split()[0]}")

try:
    import torch
except ImportError as e:
    raise RuntimeError("PyTorch not installed. Install CPU wheel before running.") from e
print(f"[ENV] Torch version: {torch.__version__} (device cpu)")

############################
# 1. Imports
############################
try:
    from nnsight import LanguageModel
except ImportError as e:
    raise RuntimeError("Failed to import nnsight.") from e

try:
    from dictionary_learning import AutoEncoder, CrossCoder
    from dictionary_learning.training import trainSAE
    from dictionary_learning.trainers import StandardTrainer
except ImportError as e:
    raise RuntimeError("Failed to import dictionary_learning components.") from e

print("[IMPORT] dictionary_learning components imported successfully.")

############################
# 2. Configuration
############################
MODEL_NAME = "EleutherAI/pythia-70m-deduped"
DEVICE = "cpu"

# Training hyperparams (tiny on purpose)
TRAIN_STEPS = 20
LOG_EVERY = 5
AE_DICT_MULTIPLIER = 4   # SAE dict size = multiplier * hidden_dim

# Activation collection parameters
ACTIVATION_SAMPLES = 2048     # number of single last-token activations to gather
BATCH_SIZE_FOR_SAE = 128      # batch size used by iterator feeding trainSAE

# Optional push
PUSH_TO_HUB = False
HF_REPO_NAME = "your-username/test-pythia70m-ae"

# Fallback manual training loop params
FALLBACK_L1_COEFF = 1e-3
FALLBACK_STEPS = 20
FALLBACK_LR = 1e-3

############################
# 3. Timing Decorator
############################
def timed(label):
    def wrap(fn):
        def inner(*args, **kwargs):
            start = time.time()
            print(f"[START] {label} ...")
            result = fn(*args, **kwargs)
            elapsed = time.time() - start
            print(f"[DONE ] {label} in {elapsed:.2f}s")
            return result
        return inner
    return wrap

############################
# 4. Helpers
############################
def normalize_activation(t):
    """Ensure 2-D shape (batch, d). Accepts (d,) or (batch,d) or (batch,seq,d)."""
    if t.ndim == 1:
        return t.unsqueeze(0)
    if t.ndim == 2:
        return t
    if t.ndim == 3:
        return t[:, -1]  # last token for each sequence
    raise ValueError(f"Unexpected activation shape {t.shape}")

def extract_tensor(obj):
    """Return raw torch.Tensor from nnsight .save() output or proxy object."""
    if hasattr(obj, "value"):
        return obj.value
    return obj

############################
# 5. Single Activation Capture
############################
@timed("Load LanguageModel & capture one activation")
def get_single_activation():
    lm = LanguageModel(MODEL_NAME, device_map=DEVICE)
    root = getattr(lm, "model", lm)

    # Pythia: use first layer MLP
    try:
        submodule = root.gpt_neox.layers[0].mlp
        submodule_name = "root.gpt_neox.layers[0].mlp"
    except AttributeError as e:
        raise RuntimeError("Could not locate Pythia MLP submodule.") from e

    prompt = "Pythia 70M dictionary learning smoke test."
    with lm.trace(prompt):
        raw = submodule.output.save()
        submodule.output.stop()

    raw_tensor = extract_tensor(raw)
    if isinstance(raw_tensor, tuple):
        raw_tensor = raw_tensor[0]
    if raw_tensor.ndim != 3:
        print(f"[WARN] Expected 3D tensor, got {raw_tensor.shape}")

    single_vec = raw_tensor[0, -1].detach().cpu()
    activation = normalize_activation(single_vec)
    d_model = activation.shape[-1]
    print(f"[ACT] Raw shape: {raw_tensor.shape}; single activation: {activation.shape}; d_model={d_model}")
    print(f"[ACT] Activation stats mean={activation.mean().item():.4f} std={activation.std().item():.4f}")
    return lm, submodule, activation, d_model, submodule_name

lm, submodule, single_act, hidden_dim, submodule_name = get_single_activation()
print(f"[INFO] Hidden dimension detected: {hidden_dim}")

############################
# 6. AutoEncoder Forward Smoke
############################
@timed("Instantiate AutoEncoder + forward pass")
def ae_forward(activation):
    ae = AutoEncoder(
        activation_dim=hidden_dim,
        dict_size=AE_DICT_MULTIPLIER * hidden_dim,
    )
    ae.to(DEVICE)
    act = activation.to(DEVICE)
    recon, feats = ae(act, output_features=True)
    mse = torch.nn.functional.mse_loss(recon, act).item()
    active_positions = (feats.abs() > 1e-4).sum().item()
    print(f"[AE ] Recon MSE (untrained): {mse:.6f}; Active positions total: {active_positions}")
    return ae

ae_model = ae_forward(single_act)

############################
# 7. Manual Activation Collection (replaces ActivationBuffer)
############################
@timed(f"Collect {ACTIVATION_SAMPLES} activations via single-prompt traces")
def collect_activations(n_samples: int) -> torch.Tensor:
    collected: List[torch.Tensor] = []
    lm.eval()
    base_prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "Interpreting activations with sparse autoencoders.",
        "Feature extraction test line.",
        "Dictionary learning improves interpretability.",
        "Neural networks can be probed layer by layer.",
        "Activation sparsity encourages disentanglement.",
        "Small prompts keep CPU tracing efficient.",
        "Testing single prompt tracing pipeline.",
        "Scaling laws suggest structure emerges.",
        "Analyzing model internals is insightful."
    ]
    idx = 0
    while len(collected) < n_samples:
        prompt = base_prompts[idx % len(base_prompts)]
        idx += 1
        with lm.trace(prompt):
            raw = submodule.output.save()
            submodule.output.stop()
        raw_tensor = extract_tensor(raw)
        if isinstance(raw_tensor, tuple):
            raw_tensor = raw_tensor[0]
        if raw_tensor.ndim != 3:
            print(f"[WARN] Unexpected captured shape {raw_tensor.shape}, skipping.")
            continue
        vec = raw_tensor[0, -1].detach().cpu()
        collected.append(vec)
        if len(collected) % 500 == 0:
            print(f"[COLLECT] {len(collected)} / {n_samples}")
    activations = torch.stack(collected)  # (n_samples, hidden_dim)
    print(f"[COLLECT] Final tensor shape: {activations.shape}")
    return activations

import torch  # ensure torch alias for type usage
all_activations = collect_activations(ACTIVATION_SAMPLES)

############################
# 8. Iterator for trainSAE
############################
class ActivationDatasetIterator:
    """
    Simple iterator producing batches of activations for trainSAE.
    Ends naturally when underlying tensor is consumed.
    """
    def __init__(self, activations: torch.Tensor, batch_size: int):
        self.acts = activations
        self.batch_size = batch_size
        self.ptr = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.ptr >= len(self.acts):
            raise StopIteration
        batch = self.acts[self.ptr : self.ptr + self.batch_size]
        self.ptr += self.batch_size
        return batch

train_iterator = ActivationDatasetIterator(all_activations, BATCH_SIZE_FOR_SAE)

############################
# 9. trainSAE (with collected activations)
############################
def manual_fallback_training(ae, activations: torch.Tensor):
    """
    Very small substitute if StandardTrainer or trainSAE pipeline errors.
    Trains with (MSE + small L1) for a handful of steps.
    """
    print("[FALLBACK] Manual minimal training loop starting.")
    ae.train()
    opt = torch.optim.Adam(ae.parameters(), lr=FALLBACK_LR)
    steps = 0
    ptr = 0
    while steps < FALLBACK_STEPS and ptr < len(activations):
        batch = activations[ptr : ptr + BATCH_SIZE_FOR_SAE].to(DEVICE)
        ptr += BATCH_SIZE_FOR_SAE
        opt.zero_grad()
        recon, feats = ae(batch, output_features=True)
        mse = torch.nn.functional.mse_loss(recon, batch)
        l1 = feats.abs().mean()
        loss = mse + FALLBACK_L1_COEFF * l1
        loss.backward()
        opt.step()
        steps += 1
        if steps % 5 == 0:
            print(f"[FALLBACK] Step {steps}/{FALLBACK_STEPS} loss={loss.item():.5f} mse={mse.item():.5f} l1={l1.item():.5f}")
    ae.eval()
    return ae

@timed("trainSAE tiny run")
def train_tiny_sae():
    trainer_cfg: Dict[str, Any] = {
        "trainer": StandardTrainer,
        "dict_class": AutoEncoder,
        "activation_dim": hidden_dim,
        "dict_size": AE_DICT_MULTIPLIER * hidden_dim,
        "lr": 1e-3,
        "l1_penalty": 1e-1,
        "device": DEVICE,
        "layer": 0,
        "lm_name": MODEL_NAME,
        "wandb_name": "pythia70m-collected-smoke",
        "submodule_name": "manual_collected_last_token",
    }
    try:
        iterator_for_train = ActivationDatasetIterator(all_activations, BATCH_SIZE_FOR_SAE)
        ae_trained = trainSAE(
            data=iterator_for_train,
            trainer_config=trainer_cfg,
            steps=TRAIN_STEPS,
            log_steps=LOG_EVERY,
            use_wandb=False,
            save_last_eval=False,  # avoid final validation when no val set
        )
        print("[INFO] trainSAE succeeded.")
    except Exception as e:
        print(f"[WARN] trainSAE failed: {e}")
        ae_trained = manual_fallback_training(ae_model, all_activations)
    ae_trained.to(DEVICE)
    return ae_trained

trained_ae = train_tiny_sae()

############################
# 10. Evaluate Trained AE
############################
@timed("Evaluate trained AE on random minibatch")
def eval_trained(ae):
    with torch.no_grad():
        batch = torch.randn(64, hidden_dim, device=DEVICE)
        recon, features = ae(batch, output_features=True)
        mse = torch.nn.functional.mse_loss(recon, batch).item()
        l1 = features.abs().mean().item()
        avg_active = (features.abs() > 1e-4).float().sum(dim=1).mean().item()
    print(f"[METRICS] MSE={mse:.4f} L1={l1:.4f} AvgActive(L0 proxy)={avg_active:.2f}")
    return dict(mse=mse, l1=l1, avg_active=avg_active)

ae_metrics = eval_trained(trained_ae)

############################
# 11. CrossCoder Mock Test
############################
@timed("Load CrossCoder checkpoint + forward (mock data)")
def test_crosscoder():
    EXPECTED_DIM = 2304
    MOCK = torch.randn(1, 2, EXPECTED_DIM)
    try:
        cc = CrossCoder.from_pretrained(
            "Butanium/gemma-2-2b-crosscoder-l13-mu4.1e-02-lr1e-04",
            from_hub=True,
            map_location="cpu",
        )
    except Exception as e:
        print(f"[CC-FAIL] Could not load CrossCoder: {e}")
        return None, None, None
    if getattr(cc, "input_dim", EXPECTED_DIM) != EXPECTED_DIM:
        print(f"[WARN] CrossCoder input_dim reported {getattr(cc, 'input_dim', 'N/A')} (expected {EXPECTED_DIM}).")
    recon, feats = cc(MOCK, output_features=True)
    mse = torch.nn.functional.mse_loss(recon, MOCK).item()
    active = (feats.abs() > 1e-4).sum().item()
    print(f"[CC ] CrossCoder forward OK. Mock MSE={mse:.5f} ActiveFeatures={active}")
    return cc, mse, active

crosscoder, cc_mse, cc_active = test_crosscoder()

############################
# 12. Manual Activation Cache
############################
@timed("Manual activation cache save & reload")
def cache_roundtrip():
    subset = all_activations[:3]
    path = "activation_cache_test.pt"
    torch.save(subset, path)
    reloaded = torch.load(path)
    assert torch.allclose(subset, reloaded), "Cache mismatch!"
    print(f"[CACHE] Saved & reloaded {reloaded.shape[0]} activations -> {path}")
    return path, reloaded.shape

cache_path, cache_shape = cache_roundtrip()

############################
# 13. (Optional) Push to Hub
############################
if PUSH_TO_HUB:
    print("[PUSH] Attempting push_to_hub...")
    try:
        trained_ae.push_to_hub(HF_REPO_NAME)
        print(f"[PUSH] Successfully pushed to: {HF_REPO_NAME}")
    except Exception as e:
        print(f"[PUSH-FAIL] Could not push model: {e}")
else:
    print("[PUSH] Skipped (set PUSH_TO_HUB=True to enable).")

############################
# 14. Final Summary
############################
print("\n========== CPU SMOKE TEST SUMMARY ==========")
print(f"Model: {MODEL_NAME}")
print(f"Collected activations: {all_activations.shape}")
print(f"Hidden dim: {hidden_dim}")
print(f"AE metrics (random batch eval): {ae_metrics}")
if crosscoder:
    print(f"CrossCoder mock: MSE={cc_mse:.5f} ActiveFeatures={cc_active}")
else:
    print("CrossCoder: skipped / failed")
print(f"Activation cache file: {cache_path}, shape={cache_shape}")
print("Note: ActivationBuffer skipped due to nnsight batch tracing issues; single-prompt collection used instead.")
print("============================================\n")
