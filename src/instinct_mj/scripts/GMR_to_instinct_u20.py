"""Convert GMR (General Motion Retargeting) data to Instinct motion format for U20.

Applies two corrections to the raw GMR pkl output:
  1. Quaternion convention: xyzw -> wxyz (MuJoCo convention)
  2. Coordinate frame: rotate +90 deg around Z axis (position + quaternion)

The GMR pipeline outputs 23-DOF joint data using its own URDF naming convention
(lleg1_joint, etc.). These names are preserved as-is; the motion loading code
resolves them via joint_name_mapping in the parkour config.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import pickle as pkl

import numpy as np
import tqdm
from scipy.spatial.transform import Rotation

# U20 23-DOF joint names in GMR output order:
# legs(8) + waist(2) + head(1) + left arm(6) + right arm(6)
U20_JOINT_NAMES = [
    "lleg1_joint",
    "lleg2_joint",
    "lleg3_joint",
    "lleg4_joint",
    "rleg1_joint",
    "rleg2_joint",
    "rleg3_joint",
    "rleg4_joint",
    "waist1_joint",
    "waist2_joint",
    "head_joint",
    "larm1_joint",
    "larm2_joint",
    "larm3_joint",
    "larm4_joint",
    "larm5_joint",
    "larm6_joint",
    "rarm1_joint",
    "rarm2_joint",
    "rarm3_joint",
    "rarm4_joint",
    "rarm5_joint",
    "rarm6_joint",
]

# Coordinate frame rotation: GMR Y-forward -> MuJoCo X-forward
R_FRAME = Rotation.from_euler("z", 90, degrees=True)


def convert_file(src_tgt_pair):
    src_file, tgt_file = src_tgt_pair

    with open(src_file, "rb") as f:
        motion_data = pkl.load(f)

    joint_pos = motion_data["dof_pos"]  # (N, 23)
    base_pos_w = motion_data["root_pos"]  # (N, 3)
    base_quat_w_xyzw = motion_data["root_rot"]  # (N, 4), xyzw order
    base_quat_w = base_quat_w_xyzw[..., [3, 0, 1, 2]]  # convert to wxyz order
    framerate = motion_data["fps"]

    # Apply coordinate frame rotation (+90 deg around Z)
    base_pos_w = R_FRAME.apply(base_pos_w)
    R_motion = Rotation.from_quat(base_quat_w[:, [1, 2, 3, 0]])  # wxyz -> xyzw for scipy
    R_corrected = R_FRAME * R_motion
    base_quat_w = R_corrected.as_quat()[:, [3, 0, 1, 2]]  # xyzw -> wxyz

    np.savez(
        tgt_file,
        framerate=framerate,
        joint_names=U20_JOINT_NAMES,
        joint_pos=joint_pos,
        base_pos_w=base_pos_w,
        base_quat_w=base_quat_w,
    )


def main():
    parser = argparse.ArgumentParser(description="Convert GMR pkl to retargeted npz for U20")
    parser.add_argument("--src", type=str, required=True, help="Input file or folder")
    parser.add_argument("--tgt", type=str, required=True, help="Target file or folder")
    parser.add_argument("--num_cpus", type=int, default=10)
    args = parser.parse_args()

    src_tgt_pairs = []
    if os.path.isfile(args.src):
        src_tgt_pairs.append((args.src, args.tgt))
    else:
        os.makedirs(args.tgt, exist_ok=True)
        for root, _, filenames in os.walk(args.src):
            target_dirpath = os.path.join(args.tgt, os.path.relpath(root, args.src))
            os.makedirs(target_dirpath, exist_ok=True)
            for filename in filenames:
                if not filename.endswith(".pkl"):
                    continue
                src_tgt_pairs.append(
                    (
                        os.path.join(root, filename),
                        os.path.join(target_dirpath, filename.replace(".pkl", "_retargeted.npz")),
                    )
                )

    print(f"Converting {len(src_tgt_pairs)} files with {args.num_cpus} workers...")

    with mp.Pool(args.num_cpus) as pool:
        list(tqdm.tqdm(
            pool.imap_unordered(convert_file, src_tgt_pairs),
            total=len(src_tgt_pairs),
        ))

    print("Done.")


if __name__ == "__main__":
    main()
