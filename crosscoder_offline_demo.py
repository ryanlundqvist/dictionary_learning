#!/usr/bin/env python
"""
CrossCoder Offline Mini Demo 

Demonstrates core *usage* patterns of CrossCoders without needing a GPU or huge
activation caches. It focuses on:
  1. Loading (or constructing) a CrossCoder model.
  2. Loading or synthesizing latent statistics (dec_norm_diff, etc.).
  3. Creating paired activations shaped (B, 2, d_model) from either:
       a) Real tiny model forward traces (Pythia 70M) OR
       b) Random mock tensors.
  4. Running a forward pass and reporting sparsity / reconstruction stats.
  5. Listing top-k “base-biased” vs “chat-biased” latents by a norm-difference metric.

USAGE:
  python crosscoder_offline_demo.py \
      --load-large-crosscoder 0 \
      --use-real-tiny-pair 1

FLAGS:
  --load-large-crosscoder {0,1}  : If 1, attempt to load real Gemma-2-2b crosscoder (~3GB).
  --use-real-tiny-pair {0,1}     : If 1, trace Pythia 70M twice to get actual layer activations.
  --topk 10                      : How many latent rows to display per direction.
  --seed 42

REQUIREMENTS:
  - dictionary_learning
  - nnsight
  - huggingface_hub
  - torch
"""

import argparse, time, sys, os
from typing import Optional, Tuple
import json
import torch
from huggingface_hub import hf_hub_download, list_repo_files

from dictionary_learning import CrossCoder
from nnsight import LanguageModel

# --------------------------------------------------
# Argument Parsing
# --------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--load-large-crosscoder", type=int, default=0,
                   help="If 1, attempt to load real Gemma-2-2b crosscoder from HF.")
    p.add_argument("--large-crosscoder-repo", type=str,
                   default="Butanium/gemma-2-2b-crosscoder-l13-mu4.1e-02-lr1e-04",
                   help="HF repo id for the big crosscoder.")
    p.add_argument("--use-real-tiny-pair", type=int, default=1,
                   help="If 1, trace Pythia 70M twice to get actual layer activations.")
    p.add_argument("--tiny-model", type=str, default="EleutherAI/pythia-70m-deduped")
    p.add_argument("--layer-idx", type=int, default=0,
                   help="Layer index to capture MLP output for tiny model.")
    p.add_argument("--activation-samples", type=int, default=64,
                   help="Number of token positions (approx) to feed to crosscoder.")
    p.add_argument("--topk", type=int, default=10,
                   help="Top-K latents per direction to display.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

# --------------------------------------------------
# Utilities
# --------------------------------------------------
def log(msg): print(f"[LOG] {msg}")

def extract_tensor(x):
    """Return raw torch.Tensor from nnsight .save() output or proxy."""
    return x.value if hasattr(x, "value") else x

def try_load_latent_stats(repo_id: str):
    try:
        files = list_repo_files(repo_id)
    except Exception as e:
        log(f"Could not list repo files: {e}")
        return None
    candidates_csv = [f for f in files if f.lower().endswith(".csv")]
    candidates_json = [f for f in files if f.lower().endswith(".json")]
    picked = None
    for c in candidates_csv + candidates_json:
        lc = c.lower()
        if any(k in lc for k in ["latent", "stat"]):
            picked = c
            break
    if picked is None:
        log("No obvious latent stats file found; will synthesize stats later.")
        return None
    local_path = hf_hub_download(repo_id=repo_id, filename=picked)
    log(f"Downloaded latent stats candidate: {picked}")
    try:
        if picked.endswith(".json"):
            with open(local_path, "r") as f:
                data = json.load(f)
            return ("json", data)
        else:
            with open(local_path, "r") as f:
                header = f.readline().strip()
                first = f.readline().strip()
            return ("csv_head", {"header": header, "first_row": first, "path": local_path})
    except Exception as e:
        log(f"Could not parse stats file {picked}: {e}")
        return None

def build_mock_latent_table(cc: CrossCoder, topk: int):
    with torch.no_grad():
        W = cc.decoder.weight  # (L, F, D)
        layer_norms = torch.norm(W, dim=2)  # (L, F)
        if layer_norms.size(0) >= 2:
            base_norm = layer_norms[0]
            chat_norm = layer_norms[1]
        else:
            base_norm = layer_norms[0]
            chat_norm = torch.zeros_like(base_norm)
        eps = 1e-8
        frac = base_norm / (base_norm + chat_norm + eps)
        rows = []
        for i in range(len(frac)):
            rows.append({
                "latent_id": i,
                "base_norm": base_norm[i].item(),
                "chat_norm": chat_norm[i].item(),
                "dec_norm_diff": frac[i].item(),
                "total_norm": (base_norm[i] + chat_norm[i]).item()
            })
        rows_sorted_low = sorted(rows, key=lambda r: r["dec_norm_diff"])[:topk]
        rows_sorted_high = sorted(rows, key=lambda r: r["dec_norm_diff"], reverse=True)[:topk]
    return rows_sorted_low, rows_sorted_high

def construct_mock_input(d_model: int, n_tokens: int):
    return torch.randn(n_tokens, 2, d_model)

def capture_tiny_pair(
    model_name: str,
    layer_idx: int,
    n_tokens: int,
    d_model: Optional[int] = None,
) -> Tuple[torch.Tensor, int]:
    log(f"Loading tiny model '{model_name}' for paired activation capture (CPU).")
    lm_base = LanguageModel(model_name, device_map="cpu")
    lm_chat = LanguageModel(model_name, device_map="cpu")
    root_b = getattr(lm_base, "model", lm_base)
    root_c = getattr(lm_chat, "model", lm_chat)
    try:
        sub_b = root_b.gpt_neox.layers[layer_idx].mlp
        sub_c = root_c.gpt_neox.layers[layer_idx].mlp
    except AttributeError:
        raise RuntimeError("Could not locate gpt_neox.layers[].mlp in tiny model.")

    prompts = [
        "Hello world interpretability test.",
        "Sparse feature probing sample.",
        "Analyzing internal representations.",
        "Crosscoder pairing demonstration.",
        "Activation geometry matters.",
        "Interpreting latent structure.",
        "Short prompt for speed.",
        "Token dynamics exploration."
    ]

    collected_base = []
    collected_chat = []
    p_idx = 0
    while len(collected_base) < n_tokens:
        prompt = prompts[p_idx % len(prompts)]
        p_idx += 1
        with lm_base.trace(prompt):
            out_b = sub_b.output.save()
            sub_b.output.stop()
        with lm_chat.trace(prompt):
            out_c = sub_c.output.save()
            sub_c.output.stop()

        tb = extract_tensor(out_b)
        tc = extract_tensor(out_c)
        if isinstance(tb, tuple): tb = tb[0]
        if isinstance(tc, tuple): tc = tc[0]
        vb = tb[0, -1].detach().cpu()
        vc = tc[0, -1].detach().cpu()
        collected_base.append(vb)
        collected_chat.append(vc)

        if len(collected_base) % 32 == 0:
            log(f"Captured {len(collected_base)} / {n_tokens} token pairs.")

    base_tensor = torch.stack(collected_base)[:n_tokens]
    chat_tensor = torch.stack(collected_chat)[:n_tokens]
    if d_model is None:
        d_model = base_tensor.shape[-1]
    paired = torch.stack([base_tensor, chat_tensor], dim=1)
    log(f"Paired activations shape: {paired.shape}")
    return paired, d_model

# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    log("=== CrossCoder Offline Mini Demo ===")
    log(f"Args: {args}")

    # 1. Load or create CrossCoder
    if args.load_large_crosscoder:
        log("Attempting to load LARGE Gemma CrossCoder (this may take time & RAM).")
        try:
            t0 = time.time()
            cc = CrossCoder.from_pretrained(
                args.large_crosscoder_repo,
                from_hub=True,
                map_location="cpu",
            )
            log(f"Loaded large CrossCoder in {time.time()-t0:.1f}s "
                f"(layers={cc.num_layers}, dict={cc.dict_size}, d={cc.activation_dim}).")
        except Exception as e:
            log(f"Failed to load large checkpoint: {e}")
            log("Falling back to synthetic tiny CrossCoder.")
            cc = CrossCoder(
                activation_dim=256,
                dict_size=4096,
                num_layers=2,
                init_with_transpose=True,
            )
    else:
        log("Creating a *synthetic tiny* CrossCoder (no large download).")
        cc = CrossCoder(
            activation_dim=256,
            dict_size=4096,
            num_layers=2,
            init_with_transpose=True,
        )

    # 2. Try latent stats
    stats_payload = None
    if args.load_large_crosscoder:
        stats_payload = try_load_latent_stats(args.large_crosscoder_repo)

    if stats_payload is not None:
        kind, content = stats_payload
        log(f"Found latent stats file type={kind}. Showing a snippet / summary:")
        if kind == "csv_head":
            print("  CSV Header:", content["header"])
            print("  First Row :", content["first_row"])
        elif kind == "json":
            keys = list(content.keys())[:5]
            print("  JSON top keys:", keys)
    else:
        log("No latent stats loaded (or synthetic model). Will create mock norm-diff table.")
        low, high = build_mock_latent_table(cc, args.topk)
        print(f"\nLowest {args.topk} dec_norm_diff (more 'chat‑leaning' if diff~0):")
        for r in low:
            print(f"  id={r['latent_id']:5d} diff=0.{'':<0}{r['dec_norm_diff']:.4f} base_norm={r['base_norm']:.3f} chat_norm={r['chat_norm']:.3f}")
        print(f"\nHighest {args.topk} dec_norm_diff (more 'base‑leaning' if diff~1):")
        for r in high:
            print(f"  id={r['latent_id']:5d} diff={r['dec_norm_diff']:.4f} base_norm={r['base_norm']:.3f} chat_norm={r['chat_norm']:.3f}")
        print()

    # 3. Build input (paired activations)
    if args.use_real_tiny_pair:
        log("Capturing *real* tiny model paired activations (base vs 'chat'=same model).")
        paired, d_model_real = capture_tiny_pair(
            args.tiny_model,
            layer_idx=args.layer_idx,
            n_tokens=args.activation_samples,
        )
        if d_model_real != cc.activation_dim:
            log(f"[WARN] Tiny pair d_model={d_model_real} != CrossCoder.d={cc.activation_dim}. Adjusting.")
            if d_model_real > cc.activation_dim:
                paired = paired[..., : cc.activation_dim]
            else:
                pad = cc.activation_dim - d_model_real
                paired = torch.nn.functional.pad(paired, (0, pad), "constant", 0.0)
    else:
        log("Using purely random mock paired activations.")
        paired = construct_mock_input(cc.activation_dim, args.activation_samples)

    # 4. Forward pass
    log("Running CrossCoder forward pass on paired activations...")
    with torch.no_grad():
        recon, feats = cc(paired, output_features=True)
        mse = torch.mean((recon - paired) ** 2).item()
        l0 = (feats.abs() > 1e-4).float().sum(dim=-1).mean().item()
        l1 = feats.abs().mean().item()

    print("\n=== Forward Pass Metrics ===")
    print(f"Input shape             : {paired.shape}")
    print(f"Reconstruction shape    : {recon.shape}")
    print(f"Features shape          : {feats.shape}")
    print(f"MSE (paired space)      : {mse:.6f}")
    print(f"Mean active (L0 proxy)  : {l0:.2f}")
    print(f"L1 mean                 : {l1:.6f}")

    top_feat_vals, top_feat_idx = torch.topk(feats.abs().mean(dim=0), 10)
    print("\nTop 10 average-activation feature indices:")
    for rank, (idx_, val_) in enumerate(zip(top_feat_idx.tolist(), top_feat_vals.tolist()), 1):
        print(f"  {rank:2d}. feature={idx_:6d} mean|f|={val_:.5f}")

    log("Demo complete. (For full interactive dashboards, a GPU + large models are needed.)")

if __name__ == "__main__":
    main()
