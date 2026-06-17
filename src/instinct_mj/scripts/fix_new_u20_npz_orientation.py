"""Fix orientation issues in new_u20_npz motion files.

Two corrections are applied:
  1. Quaternion convention: xyzw -> wxyz (MuJoCo convention)
  2. Coordinate frame: rotate +90 deg around Z axis (position + quaternion)

These bring new_u20_npz into alignment with the convention used by
GMR_to_instinct_u20.py output (u20_npz), so the loading code in
amass_motion.py can consume the data directly.

Usage:
    python fix_new_u20_npz_orientation.py
    python fix_new_u20_npz_orientation.py --dry-run   # stats only, no write
    python fix_new_u20_npz_orientation.py --src <dir> --tgt <dir>  # custom paths
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

DEFAULT_SRC = os.path.expanduser("~/Instinct-mjlab/Datasets/new_u20_npz/")


def fix_file(filepath: str, dry_run: bool = False) -> dict:
    """Fix orientation of a single npz file.

    Returns a dict with before/after stats.
    """
    data = np.load(filepath, allow_pickle=True)

    base_pos = data["base_pos_w"].astype(np.float64).copy()
    base_quat = data["base_quat_w"].astype(np.float64).copy()

    # --- Correction 1: quaternion xyzw -> wxyz ---
    # The data is stored in [x, y, z, w] order; MuJoCo expects [w, x, y, z].
    quat_wxyz = base_quat[:, [3, 0, 1, 2]]

    # --- Correction 2: apply +90 deg Z-axis rotation to the world frame ---
    # This aligns the motion's forward direction from -Y to +X.
    R_z = Rotation.from_euler("z", 90, degrees=True)

    # Rotate root translation
    pos_corrected = R_z.apply(base_pos)

    # Rotate root orientation (left-multiply in world frame)
    # scipy uses xyzw internally
    R_motion = Rotation.from_quat(quat_wxyz[:, [1, 2, 3, 0]])
    R_corrected = R_z * R_motion
    quat_corrected = R_corrected.as_quat()[:, [3, 0, 1, 2]]  # back to wxyz

    stats = {
        "file": os.path.basename(filepath),
        "frames": base_pos.shape[0],
        "pos_before_z": float(base_pos[:, 2].mean()),
        "pos_after_z": float(pos_corrected[:, 2].mean()),
    }

    if not dry_run:
        np.savez(
            filepath,
            framerate=data["framerate"],
            joint_names=data["joint_names"],
            joint_pos=data["joint_pos"],
            base_pos_w=pos_corrected,
            base_quat_w=quat_corrected,
        )

    return stats


def main():
    parser = argparse.ArgumentParser(description="Fix orientation in new_u20_npz files")
    parser.add_argument("--src", type=str, default=DEFAULT_SRC, help="Source directory")
    parser.add_argument("--tgt", type=str, default=None, help="Target dir (default: overwrite src)")
    parser.add_argument("--dry-run", action="store_true", help="Print stats only, do not write")
    args = parser.parse_args()

    src_dir = args.src
    tgt_dir = args.tgt or args.src

    # Collect npz files
    files = []
    for dirpath, _, filenames in os.walk(src_dir):
        for f in sorted(filenames):
            if f.endswith("retargeted.npz") or f.endswith("retargetted.npz"):
                files.append(os.path.join(dirpath, f))
    files = sorted(files)

    if not files:
        print(f"[ERROR] No retargeted npz files found in {src_dir}")
        sys.exit(1)

    print(f"[INFO] Found {len(files)} files")
    print(f"[INFO] Source: {src_dir}")
    print(f"[INFO] Target: {tgt_dir}")
    print(f"[INFO] Dry run: {args.dry_run}")
    print()

    # If target differs from source, copy structure
    if tgt_dir != src_dir and not args.dry_run:
        os.makedirs(tgt_dir, exist_ok=True)

    for i, filepath in enumerate(files):
        if tgt_dir != src_dir:
            rel = os.path.relpath(filepath, src_dir)
            out_path = os.path.join(tgt_dir, rel)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            # Read from source, write to target
            data = np.load(filepath, allow_pickle=True)
            base_pos = data["base_pos_w"].astype(np.float64).copy()
            base_quat = data["base_quat_w"].astype(np.float64).copy()
            quat_wxyz = base_quat[:, [3, 0, 1, 2]]
            R_z = Rotation.from_euler("z", 90, degrees=True)
            pos_corrected = R_z.apply(base_pos)
            R_motion = Rotation.from_quat(quat_wxyz[:, [1, 2, 3, 0]])
            R_corrected = R_z * R_motion
            quat_corrected = R_corrected.as_quat()[:, [3, 0, 1, 2]]
            np.savez(
                out_path,
                framerate=data["framerate"],
                joint_names=data["joint_names"],
                joint_pos=data["joint_pos"],
                base_pos_w=pos_corrected,
                base_quat_w=quat_corrected,
            )
        else:
            stats = fix_file(filepath, dry_run=args.dry_run)

        if (i + 1) % 20 == 0 or i == 0:
            action = "Would fix" if args.dry_run else "Fixed"
            print(f"  [{i+1}/{len(files)}] {action} {os.path.basename(filepath)}")

    print(f"\n[INFO] Done. {'Processed (dry run)' if args.dry_run else 'Fixed'} {len(files)} files.")


if __name__ == "__main__":
    main()
