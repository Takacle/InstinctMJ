#!/usr/bin/env python3
"""Verify ONNX model I/O shapes and compare PyTorch vs ONNX inference.

Usage:
  # Inspect ONNX shapes and run dummy inference (no checkpoint needed):
  python InstinctMJ/src/instinct_mj/tasks/parkour/scripts/verify_onnx deploy/policy/

  # Full PyTorch vs ONNX comparison (requires checkpoint + environment):
  python InstinctMJ/src/instinct_mj/tasks/parkour/scripts/verify_onnx deploy/policy/ \
      --checkpoint logs/instinct_rl/v11_parkour/2026-04-08_15-56-14/model_50000.pt

  # Compare with specific task ID:
  python InstinctMJ/src/instinct_mj/tasks/parkour/scripts/verify_onnx deploy/policy/ \
      --checkpoint logs/instinct_rl/v11_parkour/2026-04-08_15-56-14/model_50000.pt \
      --task-id Instinct-Parkour-Target-Amp-V11-Play-v0
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Phase 1: ONNX shape inspection
# ---------------------------------------------------------------------------


def inspect_onnx(model_dir: Path, verbose: bool = True) -> dict:
    """Load ONNX files, print shapes, return structured info."""
    import onnxruntime as ort

    info: dict = {"encoder": {}, "actor": {}}

    encoder_path = model_dir / "0-depth_encoder.onnx"
    actor_path = model_dir / "actor.onnx"
    metadata_path = model_dir / "metadata.json"

    # --- metadata ---
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        info["metadata"] = metadata
        if verbose:
            print("=" * 60)
            print("metadata.json")
            print("=" * 60)
            print(json.dumps(metadata, indent=2, ensure_ascii=False))
    else:
        print(f"[WARN] metadata.json not found in {model_dir}")
        info["metadata"] = {}

    # --- encoder ---
    if encoder_path.exists():
        session = ort.InferenceSession(str(encoder_path), providers=["CPUExecutionProvider"])
        inp = session.get_inputs()[0]
        out = session.get_outputs()[0]
        info["encoder"]["input_name"] = inp.name
        info["encoder"]["input_shape"] = inp.shape
        info["encoder"]["input_type"] = inp.type
        info["encoder"]["output_name"] = out.name
        info["encoder"]["output_shape"] = out.shape
        info["encoder"]["output_type"] = out.type
        if verbose:
            print()
            print("=" * 60)
            print(f"Encoder: {encoder_path}")
            print("=" * 60)
            print(f"  Input:  name={inp.name}, shape={inp.shape}, type={inp.type}")
            print(f"  Output: name={out.name}, shape={out.shape}, type={out.type}")
    else:
        print(f"[ERROR] Encoder not found: {encoder_path}")
        sys.exit(1)

    # --- actor ---
    if actor_path.exists():
        session = ort.InferenceSession(str(actor_path), providers=["CPUExecutionProvider"])
        inp = session.get_inputs()[0]
        out = session.get_outputs()[0]
        info["actor"]["input_name"] = inp.name
        info["actor"]["input_shape"] = inp.shape
        info["actor"]["input_type"] = inp.type
        info["actor"]["output_name"] = out.name
        info["actor"]["output_shape"] = out.shape
        info["actor"]["output_type"] = out.type
        if verbose:
            print()
            print("=" * 60)
            print(f"Actor: {actor_path}")
            print("=" * 60)
            print(f"  Input:  name={inp.name}, shape={inp.shape}, type={inp.type}")
            print(f"  Output: name={out.name}, shape={out.shape}, type={out.type}")
    else:
        print(f"[ERROR] Actor not found: {actor_path}")
        sys.exit(1)

    # --- normalizer ---
    normalizer_path = model_dir / "policy_normalizer.npz"
    if normalizer_path.exists():
        data = np.load(normalizer_path)
        info["normalizer"] = {
            "mean_shape": data["mean"].shape,
            "std_shape": data["std"].shape,
            "eps": float(data["eps"]) if "eps" in data else None,
            "mean_sample": data["mean"].flatten()[:5].tolist(),
            "std_sample": data["std"].flatten()[:5].tolist(),
        }
        if verbose:
            print()
            print("=" * 60)
            print(f"Normalizer: {normalizer_path}")
            print("=" * 60)
            print(f"  mean shape: {data['mean'].shape}")
            print(f"  std shape:  {data['std'].shape}")
            print(f"  eps:        {float(data['eps']) if 'eps' in data else 'N/A'}")
            print(f"  mean[:5]:   {data['mean'].flatten()[:5]}")
            print(f"  std[:5]:    {data['std'].flatten()[:5]}")
    else:
        if verbose:
            print()
            print("[INFO] No policy_normalizer.npz found (normalization not used)")

    return info


# ---------------------------------------------------------------------------
# Phase 2: Dummy inference (validate shapes are consistent)
# ---------------------------------------------------------------------------


def _get_fixed_batch(onnx_shape: list) -> int:
    """Extract fixed batch dimension from ONNX shape annotation."""
    batch = onnx_shape[0]
    if isinstance(batch, int) and batch > 0:
        return batch
    # Dynamic batch (string like 'batch' or negative)
    return 1


def run_dummy_inference(model_dir: Path, info: dict, verbose: bool = True) -> dict:
    """Run ONNX inference with random data, verify shape consistency."""
    import onnxruntime as ort

    results: dict = {}

    obs_format = info.get("metadata", {}).get("obs_format", {})
    policy_format = obs_format.get("policy", {})

    # Build obs_segments from metadata
    obs_segments: OrderedDict[str, tuple] = OrderedDict()
    for name, shape_list in policy_format.items():
        obs_segments[name] = tuple(shape_list)

    # Compute sizes
    obs_sizes = {name: int(np.prod(shape)) for name, shape in obs_segments.items()}
    total_obs = sum(obs_sizes.values())

    if verbose:
        print()
        print("=" * 60)
        print("Shape Analysis (from metadata)")
        print("=" * 60)
        for name, shape in obs_segments.items():
            print(f"  {name}: shape={shape}, size={obs_sizes[name]}")
        print(f"  Total obs size: {total_obs}")

    # Encoder inference
    encoder_session = ort.InferenceSession(
        str(model_dir / "0-depth_encoder.onnx"), providers=["CPUExecutionProvider"]
    )
    enc_input = encoder_session.get_inputs()[0]
    enc_output = encoder_session.get_outputs()[0]

    depth_shape = obs_segments.get("depth_image", None)
    if depth_shape is None:
        print("[ERROR] 'depth_image' not found in obs_format!")
        sys.exit(1)

    # Use fixed batch size from ONNX model
    enc_batch = _get_fixed_batch(enc_input.shape)
    dummy_depth = np.random.randn(enc_batch, *depth_shape).astype(np.float32)
    depth_embed = encoder_session.run(None, {enc_input.name: dummy_depth})[0]

    results["depth_input_shape"] = dummy_depth.shape
    results["depth_embed_shape"] = depth_embed.shape

    if verbose:
        print()
        print("=" * 60)
        print("Dummy Encoder Inference")
        print("=" * 60)
        print(f"  batch_size:      {enc_batch} (from ONNX shape)")
        print(f"  depth_input:     {dummy_depth.shape}")
        print(f"  depth_embedded:  {depth_embed.shape}")
        print(f"  encoder expects: {enc_input.shape} -> {enc_output.shape}")

    # Actor inference
    actor_session = ort.InferenceSession(
        str(model_dir / "actor.onnx"), providers=["CPUExecutionProvider"]
    )
    act_input = actor_session.get_inputs()[0]
    act_output = actor_session.get_outputs()[0]

    # Use fixed batch size from actor ONNX model
    act_batch = _get_fixed_batch(act_input.shape)
    # Build proprio (everything except depth_image)
    proprio_size = total_obs - obs_sizes.get("depth_image", 0)
    dummy_proprio = np.random.randn(act_batch, proprio_size).astype(np.float32)

    # Match depth_embed batch to actor batch
    if enc_batch == act_batch:
        depth_for_actor = depth_embed
    else:
        # Repeat or slice to match
        depth_for_actor = np.tile(depth_embed, (act_batch // enc_batch + 1, 1))[:act_batch]

    actor_input = np.concatenate([dummy_proprio, depth_for_actor], axis=1)
    action_output = actor_session.run(None, {act_input.name: actor_input})[0]

    results["proprio_shape"] = dummy_proprio.shape
    results["actor_input_shape"] = actor_input.shape
    results["action_output_shape"] = action_output.shape

    if verbose:
        print()
        print("=" * 60)
        print("Dummy Actor Inference")
        print("=" * 60)
        print(f"  batch_size:      {act_batch} (from ONNX shape)")
        print(f"  proprio:         {dummy_proprio.shape}")
        print(f"  actor_input:     {actor_input.shape}")
        print(f"  action_output:   {action_output.shape}")
        print(f"  actor expects:   {act_input.shape} -> {act_output.shape}")

    # Consistency check
    if verbose:
        print()
        print("=" * 60)
        print("Consistency Check")
        print("=" * 60)

    # Check actor input size matches expected
    expected_actor_input = proprio_size + int(np.prod(depth_for_actor.shape[1:]))
    actual_actor_input = actor_input.shape[1]
    ok_actor_input = expected_actor_input == actual_actor_input
    if verbose:
        status = "✅" if ok_actor_input else "❌"
        print(f"  {status} Actor input size: expected={expected_actor_input}, actual={actual_actor_input}")

    # Check batch sizes are consistent
    ok_batch = enc_batch == act_batch
    if verbose:
        status = "✅" if ok_batch else "⚠️"
        print(f"  {status} Batch size: encoder={enc_batch}, actor={act_batch}")

    # Check depth encoder output size matches expected actor input
    actor_expected_feature = act_input.shape[1] if isinstance(act_input.shape[1], int) else "dynamic"
    proprio_plus_latent = proprio_size + depth_embed.shape[1]
    if verbose:
        print(f"  ℹ️  Proprio size:        {proprio_size}")
        print(f"  ℹ️  Depth latent size:   {depth_embed.shape[1]}")
        print(f"  ℹ️  Sum (actor input):   {proprio_plus_latent}")
        print(f"  ℹ️  Actor input dim:     {actor_expected_feature}")
        ok_feature = proprio_plus_latent == actor_expected_feature
        status = "✅" if ok_feature else "❌"
        print(f"  {status} Feature dim match: {proprio_plus_latent} vs {actor_expected_feature}")

    results["consistent"] = ok_actor_input
    return results


# ---------------------------------------------------------------------------
# Phase 3: Full PyTorch vs ONNX comparison
# ---------------------------------------------------------------------------


def compare_pytorch_onnx(
    model_dir: Path,
    checkpoint: str,
    task_id: str,
    num_samples: int = 10,
    verbose: bool = True,
) -> dict:
    """Load PyTorch model from checkpoint, compare inference with ONNX stage by stage."""
    import onnxruntime as ort
    import torch

    from instinct_rl.utils.utils import get_subobs_by_components, get_subobs_size

    import instinct_mj.tasks  # noqa: F401
    from instinct_mj.envs import InstinctRlEnv
    from instinct_mj.rl import InstinctRlVecEnvWrapper
    from instinct_mj.tasks.registry import load_env_cfg, load_instinct_rl_cfg, load_runner_cls

    from instinct_rl.runners import OnPolicyRunner

    # Load config
    env_cfg = load_env_cfg(task_id, play=True)
    agent_cfg = load_instinct_rl_cfg(task_id)

    # Create environment
    env = InstinctRlEnv(cfg=env_cfg, device="cpu")
    vec_env = InstinctRlVecEnvWrapper(
        env,
        policy_group=agent_cfg.policy_observation_group,
        critic_group=agent_cfg.critic_observation_group,
    )

    # Load runner and checkpoint
    runner_cls = load_runner_cls(task_id) or OnPolicyRunner
    runner = runner_cls(vec_env, agent_cfg.to_dict(), log_dir=None, device="cpu")
    runner.load(checkpoint)
    print(f"[INFO] Loaded checkpoint: {checkpoint}")

    # Check if normalizer is active
    has_normalizer = "policy" in runner.normalizers
    normalizer = runner.normalizers.get("policy", None)
    print(f"[INFO] Has policy normalizer: {has_normalizer}")

    # Get obs format
    obs_format = vec_env.get_obs_format()
    obs_segments = obs_format["policy"]
    print(f"[INFO] obs_segments: {obs_segments}")

    # Extract encoder config details
    encoder_configs = agent_cfg.policy.encoder_configs
    depth_components = encoder_configs.depth_encoder.component_names

    proprio_components = [
        name for name in obs_segments if name not in depth_components
    ]
    proprio_size = get_subobs_size(obs_segments, proprio_components)
    depth_shape = obs_segments.get("depth_image", None)

    print(f"[INFO] depth_components: {depth_components}")
    print(f"[INFO] proprio_components: {proprio_components}")
    print(f"[INFO] proprio_size: {proprio_size}")
    print(f"[INFO] depth_shape: {depth_shape}")

    # Load ONNX models
    ort_providers = ["CPUExecutionProvider"]
    encoder_session = ort.InferenceSession(
        str(model_dir / "0-depth_encoder.onnx"), providers=ort_providers
    )
    actor_session = ort.InferenceSession(
        str(model_dir / "actor.onnx"), providers=ort_providers
    )
    enc_input_name = encoder_session.get_inputs()[0].name
    act_input_name = actor_session.get_inputs()[0].name

    # Get ONNX fixed batch sizes
    enc_batch = _get_fixed_batch(encoder_session.get_inputs()[0].shape)
    act_batch = _get_fixed_batch(actor_session.get_inputs()[0].shape)
    batch_size = min(enc_batch, act_batch)
    print(f"[INFO] ONNX batch sizes: encoder={enc_batch}, actor={act_batch}, using={batch_size}")

    # Load normalizer from npz for ONNX path
    onnx_normalizer_mean = None
    onnx_normalizer_std = None
    onnx_normalizer_eps = 0.0
    normalizer_path = model_dir / "policy_normalizer.npz"
    if normalizer_path.exists():
        data = np.load(normalizer_path)
        onnx_normalizer_mean = data["mean"]  # shape: (1, obs_dim)
        onnx_normalizer_std = data["std"]    # shape: (1, obs_dim)
        onnx_normalizer_eps = float(data.get("eps", 1e-2))
        print(f"[INFO] Loaded normalizer from {normalizer_path}")
        print(f"       mean shape: {onnx_normalizer_mean.shape}, std shape: {onnx_normalizer_std.shape}")
    elif has_normalizer:
        print("[WARN] PyTorch has normalizer but no policy_normalizer.npz found!")

    # Get PyTorch model components
    model = runner.alg.actor_critic
    model.eval()

    # Collect errors
    errors: dict = {
        "encoder_max_abs": [],
        "encoder_mean_abs": [],
        "encoder_cos_sim": [],
        "actor_input_max_abs": [],
        "actor_output_max_abs": [],
        "actor_output_mean_abs": [],
        "actor_output_cos_sim": [],
    }

    # Get real observations
    observations, _ = vec_env.get_observations()
    num_envs = observations.shape[0]
    num_batches = min(num_samples, num_envs // batch_size)

    if num_batches == 0:
        # If not enough envs, pad observations
        print(f"[WARN] Not enough envs ({num_envs}) for batch_size={batch_size}. Padding.")
        repeat_factor = (batch_size // num_envs) + 1
        observations = observations.repeat(repeat_factor, 1)[:batch_size]
        num_batches = 1

    for b in range(num_batches):
        start_idx = b * batch_size
        obs = observations[start_idx : start_idx + batch_size]

        with torch.no_grad():
            # --- PyTorch path ---
            # Apply normalizer if present (same as act_inference)
            if normalizer is not None:
                obs_normalized = normalizer(obs)
            else:
                obs_normalized = obs

            # Encoder (ParallelLayer -> Conv2d)
            pt_encoded = model.encoders(obs_normalized)

            # Extract depth latent from encoded output
            pt_depth_latent = pt_encoded[:, -encoder_configs.depth_encoder.output_size:]

            # Actor (MoE)
            pt_action = model.actor(pt_encoded)

            # --- ONNX path ---
            obs_np = obs.numpy()

            # Apply normalizer manually for ONNX
            if onnx_normalizer_mean is not None:
                obs_for_onnx = (obs_np - onnx_normalizer_mean) / (onnx_normalizer_std + onnx_normalizer_eps)
            else:
                obs_for_onnx = obs_np

            # Extract depth_image
            onnx_depth = get_subobs_by_components(
                obs, depth_components, obs_segments, temporal=True
            ).cpu().numpy()
            onnx_depth = onnx_depth.reshape(-1, *depth_shape)

            # Extract proprio (everything before depth_image)
            onnx_proprio = obs_for_onnx[:, :proprio_size]

            # Encoder
            onnx_depth_latent = encoder_session.run(None, {enc_input_name: onnx_depth})[0]

            # Actor
            onnx_actor_input = np.concatenate([onnx_proprio, onnx_depth_latent], axis=1)
            onnx_action = actor_session.run(None, {act_input_name: onnx_actor_input})[0]

        # --- Compare ---
        pt_depth_latent_np = pt_depth_latent.cpu().numpy()
        pt_action_np = pt_action.cpu().numpy()
        pt_encoded_np = pt_encoded.cpu().numpy()

        # Encoder comparison
        enc_diff = np.abs(pt_depth_latent_np - onnx_depth_latent)
        errors["encoder_max_abs"].append(float(enc_diff.max()))
        errors["encoder_mean_abs"].append(float(enc_diff.mean()))
        enc_cos = np.sum(pt_depth_latent_np * onnx_depth_latent) / (
            np.linalg.norm(pt_depth_latent_np) * np.linalg.norm(onnx_depth_latent) + 1e-12
        )
        errors["encoder_cos_sim"].append(float(enc_cos))

        # Actor input comparison
        actor_in_diff = np.abs(pt_encoded_np - onnx_actor_input)
        errors["actor_input_max_abs"].append(float(actor_in_diff.max()))

        # Actor output comparison
        act_diff = np.abs(pt_action_np - onnx_action)
        errors["actor_output_max_abs"].append(float(act_diff.max()))
        errors["actor_output_mean_abs"].append(float(act_diff.mean()))
        act_cos = np.sum(pt_action_np * onnx_action) / (
            np.linalg.norm(pt_action_np) * np.linalg.norm(onnx_action) + 1e-12
        )
        errors["actor_output_cos_sim"].append(float(act_cos))

        if verbose and b == 0:
            print()
            print("=" * 60)
            print(f"Batch {b} (first)")
            print("=" * 60)
            print(f"  obs shape:             {obs.shape}")
            print(f"  obs_normalized range:  [{obs_normalized.min():.4f}, {obs_normalized.max():.4f}]")
            print(f"  onnx_proprio range:    [{onnx_proprio.min():.4f}, {onnx_proprio.max():.4f}]")
            print()
            print(f"  pt_depth_latent shape: {pt_depth_latent_np.shape}")
            print(f"  onnx_depth_latent:     {onnx_depth_latent.shape}")
            print(f"  encoder max abs err:   {enc_diff.max():.8f}")
            print(f"  encoder mean abs err:  {enc_diff.mean():.8f}")
            print(f"  encoder cosine sim:    {enc_cos:.8f}")
            print()
            print(f"  pt_actor_input shape:  {pt_encoded_np.shape}")
            print(f"  onnx_actor_input:      {onnx_actor_input.shape}")
            print(f"  actor input max err:   {actor_in_diff.max():.8f}")
            print()
            print(f"  pt_action shape:       {pt_action_np.shape}")
            print(f"  onnx_action shape:     {onnx_action.shape}")
            print(f"  action max abs err:    {act_diff.max():.8f}")
            print(f"  action mean abs err:   {act_diff.mean():.8f}")
            print(f"  action cosine sim:     {act_cos:.8f}")
            print()
            print(f"  pt_action[:5]:    {pt_action_np[0, :5]}")
            print(f"  onnx_action[:5]:  {onnx_action[0, :5]}")

    # Summary
    print()
    print("=" * 60)
    print(f"Summary ({num_batches} batches x {batch_size} envs)")
    print("=" * 60)
    metrics = [
        "encoder_max_abs",
        "encoder_mean_abs",
        "encoder_cos_sim",
        "actor_input_max_abs",
        "actor_output_max_abs",
        "actor_output_mean_abs",
        "actor_output_cos_sim",
    ]
    for key in metrics:
        vals = errors[key]
        label = key + ":"
        print(
            f"  {label:<31s}mean={np.mean(vals):.8f}, "
            f"max={np.max(vals):.8f}, min={np.min(vals):.8f}"
        )

    # Diagnosis
    print()
    print("=" * 60)
    print("Diagnosis")
    print("=" * 60)

    # Normalizer check
    if has_normalizer:
        print("  ℹ️  PyTorch uses EmpiricalNormalization for policy obs.")
        if onnx_normalizer_mean is not None:
            # Compare normalizer values
            pt_mean = normalizer._mean.cpu().numpy()
            pt_std = normalizer._std.cpu().numpy()
            mean_diff = np.abs(pt_mean - onnx_normalizer_mean).max()
            std_diff = np.abs(pt_std - onnx_normalizer_std).max()
            print(f"  ✅ Normalizer npz found. mean_diff={mean_diff:.8f}, std_diff={std_diff:.8f}")
            if mean_diff > 1e-6 or std_diff > 1e-6:
                print(f"  ⚠️  Normalizer values differ between checkpoint and npz file!")
        else:
            print("  ❌ Normalizer active in PyTorch but NO policy_normalizer.npz in deploy dir!")
            print("     → This means ONNX inference gets RAW obs while PyTorch gets NORMALIZED obs!")
            print("     → You MUST apply normalization before ONNX inference.")
    else:
        print("  ℹ️  No normalization in PyTorch path.")

    mean_enc_err = np.mean(errors["encoder_max_abs"])
    mean_act_err = np.mean(errors["actor_output_max_abs"])
    mean_cos = np.mean(errors["actor_output_cos_sim"])

    if mean_enc_err > 1e-3:
        print("  ⚠️  Encoder stage has noticeable numerical differences.")
        print("     → Try re-exporting encoder with higher opset or without dynamo.")
    else:
        print("  ✅ Encoder stage: numerically close.")

    if mean_act_err > 1e-2:
        print("  ⚠️  Actor output has significant differences!")
        if mean_enc_err < 1e-3:
            print("     → Encoder is fine but actor diverges — likely MoE precision issue.")
            print("     → Try: (1) export actor with opset 17, (2) use dynamo=True for actor,")
            print("            (3) or convert MoE to single-expert before export.")
        else:
            print("     → Error propagates from encoder stage.")
    elif mean_act_err > 1e-4:
        print("  ⚠️  Actor output has small differences — likely FP32 accumulation differences.")
    else:
        print("  ✅ Actor output: numerically close.")

    if mean_cos < 0.999:
        print(f"  ⚠️  Cosine similarity = {mean_cos:.6f} — outputs point in different directions!")
    else:
        print(f"  ✅ Cosine similarity = {mean_cos:.6f} — outputs are well-aligned.")

    return errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Verify v11-parkour ONNX models")
    parser.add_argument("model_dir", type=str, help="Path to the ONNX model directory")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to PyTorch checkpoint for full comparison",
    )
    parser.add_argument(
        "--task-id",
        type=str,
        default="Instinct-Parkour-Target-Amp-V11-Play-v0",
        help="Task ID for environment creation (default: V11 Play)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of batches for comparison (default: 10)",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Less verbose output")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"[ERROR] Model directory not found: {model_dir}")
        sys.exit(1)

    verbose = not args.quiet

    # Phase 1: Inspect ONNX shapes
    info = inspect_onnx(model_dir, verbose=verbose)

    # Phase 2: Dummy inference
    run_dummy_inference(model_dir, info, verbose=verbose)

    # Phase 3: Full comparison (optional)
    if args.checkpoint:
        if not Path(args.checkpoint).exists():
            print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
            sys.exit(1)
        compare_pytorch_onnx(
            model_dir=model_dir,
            checkpoint=args.checkpoint,
            task_id=args.task_id,
            num_samples=args.num_samples,
            verbose=verbose,
        )
    else:
        print()
        print("=" * 60)
        print("Tip: Add --checkpoint <path> for full PyTorch vs ONNX comparison")
        print("=" * 60)


if __name__ == "__main__":
    main()