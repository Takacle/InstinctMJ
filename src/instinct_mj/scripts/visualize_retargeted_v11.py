"""Visualize retargeted motion npz files on V11 robot using MuJoCo viewer.

Usage:
    # Single file
    python visualize_retargeted_v11.py --file /path/to/motion_retargeted.npz

    # Folder (plays all *retargeted.npz files sequentially)
    python visualize_retargeted_v11.py --dir /path/to/npz_folder/

    # Headless stats only (no viewer)
    python visualize_retargeted_v11.py --dir /path/to/npz_folder/ --headless
"""

from __future__ import annotations

import argparse
import os
import time

import mujoco
import mujoco.viewer
import numpy as np

V11_XML_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "assets", "resources", "v11", "xml", "v11.xml",
)


def load_retargeted_npz(filepath: str):
    """Load a retargeted npz and return (base_pos_w, base_quat_w, joint_pos, joint_names, fps)."""
    data = np.load(filepath, allow_pickle=True)
    return (
        data["base_pos_w"].astype(np.float64),
        data["base_quat_w"].astype(np.float64),
        data["joint_pos"].astype(np.float64),
        data["joint_names"].tolist() if hasattr(data["joint_names"], "tolist") else list(data["joint_names"]),
        float(data["framerate"]),
    )


def build_joint_mapping(model: mujoco.MjModel, npz_joint_names: list[str]) -> list[tuple[int, int]]:
    """Build mapping from npz joint index to MuJoCo qpos address.

    Returns list of (npz_col_idx, mj_qpos_addr) pairs.
    """
    mapping = []
    for npz_idx, name in enumerate(npz_joint_names):
        mj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if mj_id >= 0:
            mapping.append((npz_idx, model.jnt_qposadr[mj_id]))
    return mapping


def print_motion_stats(filepath: str, base_pos: np.ndarray, joint_pos: np.ndarray, fps: float):
    """Print basic motion statistics."""
    n = len(base_pos)
    duration = n / fps
    h = base_pos[:, 2]
    print(f"  File: {os.path.basename(filepath)}")
    print(f"  Frames: {n}, FPS: {fps:.0f}, Duration: {duration:.2f}s")
    print(f"  Root height: min={h.min():.3f}, max={h.max():.3f}, mean={h.mean():.3f}")
    if n > 1:
        vel = (base_pos[1:] - base_pos[:-1]) * fps
        max_speed = np.max(np.linalg.norm(vel, axis=-1))
        print(f"  Max linear velocity: {max_speed:.3f} m/s")
    jr = joint_pos.max(axis=0) - joint_pos.min(axis=0)
    print(f"  Joint range (mean): {jr.mean():.3f} rad")


def play_motion(model, data, base_pos, base_quat, joint_pos, fps, joint_mapping):
    """Play motion in MuJoCo passive viewer."""
    frame_dt = 1.0 / fps
    n = len(base_pos)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [base_pos[0, 0], base_pos[0, 1], 0.9]
        viewer.cam.distance = 3.5
        viewer.cam.elevation = -20.0

        frame_idx = 0
        loop = True
        while viewer.is_running():
            step_start = time.perf_counter()

            # Set floating base
            data.qpos[0:3] = base_pos[frame_idx]
            data.qpos[3:7] = base_quat[frame_idx]

            # Set joint positions
            for npz_idx, qpos_addr in joint_mapping:
                data.qpos[qpos_addr] = joint_pos[frame_idx, npz_idx]

            mujoco.mj_forward(model, data)
            viewer.sync()

            frame_idx += 1
            if frame_idx >= n:
                if not loop:
                    break
                frame_idx = 0  # loop playback

            elapsed = time.perf_counter() - step_start
            if elapsed < frame_dt:
                time.sleep(frame_dt - elapsed)


def collect_files(path: str) -> list[str]:
    """Collect retargeted npz files from a path."""
    if os.path.isfile(path):
        return [path]
    files = []
    for dirpath, _, filenames in os.walk(path):
        for f in sorted(filenames):
            if f.endswith("retargeted.npz") or f.endswith("retargetted.npz"):
                files.append(os.path.join(dirpath, f))
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description="Visualize V11 retargeted motion")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=str, help="Single npz file")
    group.add_argument("--dir", type=str, help="Directory of npz files")
    parser.add_argument("--xml", type=str, default=V11_XML_PATH, help="V11 MJCF XML path")
    parser.add_argument("--headless", action="store_true", help="Print stats only, no viewer")
    args = parser.parse_args()

    path = args.file or args.dir
    files = collect_files(path)
    if not files:
        print(f"[ERROR] No retargeted npz files found in {path}")
        return

    print(f"[INFO] Found {len(files)} motion file(s)")
    print(f"[INFO] Robot XML: {args.xml}")

    os.environ.setdefault("MUJOCO_GL", "glfw")
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    joint_mapping = None

    for i, filepath in enumerate(files):
        print(f"\n[{i+1}/{len(files)}]")
        base_pos, base_quat, joint_pos, joint_names, fps = load_retargeted_npz(filepath)
        print_motion_stats(filepath, base_pos, joint_pos, fps)

        if joint_mapping is None:
            joint_mapping = build_joint_mapping(model, joint_names)
            matched = len(joint_mapping)
            print(f"  Joint mapping: {matched}/{len(joint_names)} matched to MuJoCo model")

        if not args.headless:
            print("  [Playing... close viewer window to advance to next]")
            play_motion(model, data, base_pos, base_quat, joint_pos, fps, joint_mapping)

    print(f"\n[INFO] Done. Processed {len(files)} files.")


if __name__ == "__main__":
    main()
