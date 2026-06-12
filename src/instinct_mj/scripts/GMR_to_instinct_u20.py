"""Convert GMR (General Motion Retargeting) data to Instinct motion format for U20.

U20's root body is base_link, same as GMR output, so no frame transformation is needed.
This script repacks the pkl data into the retargeted npz format with U20 22-DOF joint names.
"""

from __future__ import annotations

import argparse
import functools
import multiprocessing as mp
import os
import pickle as pkl

import numpy as np
import tqdm

# U20 22-DOF joint names (same order as GMR output: legs(8) + waist(2) + arms(12))
U20_JOINT_NAMES = [
    "left_hip_yaw_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_knee_joint",
    "right_hip_yaw_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_knee_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def convert_file(src_tgt_pair):
    src_file, tgt_file = src_tgt_pair

    with open(src_file, "rb") as f:
        motion_data = pkl.load(f)

    joint_pos = motion_data["dof_pos"]  # (N, 22)
    base_pos_w = motion_data["root_pos"]  # (N, 3)
    base_quat_w_xyzw = motion_data["root_rot"]  # (N, 4), xyzw order
    base_quat_w = base_quat_w_xyzw[..., [3, 0, 1, 2]]  # convert to wxyz order
    framerate = motion_data["fps"]

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
