"""Visualize U20 foot sole collision capsules and volume-points grid.

Renders a side-by-side comparison of:
  - 5 foot collision capsules (blue, transparent)
  - 64 volume-point sample spheres per foot, color-coded by proximity:
      GREEN  = inside capsule collision surface
      YELLOW = within 10 mm of capsule surface
      RED    = > 10 mm from any capsule surface

Usage:
    python src/instinct_mj/scripts/visualize_u20_foot_contact.py
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np

XML_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets",
    "resources",
    "u20",
    "xml",
    "u20_popsicle.xml",
)

CAPSULE_RADIUS = 0.015
CAPSULE_HALF_LEN = 0.1
CAPSULE_XS = np.array([-0.04, -0.02, 0.0, 0.02, 0.04])
LEFT_CAPSULE_Y = 0.012763
RIGHT_CAPSULE_Y = -0.0127632
CAPSULE_Z = -0.08

VP_X = (-0.06, 0.06)
VP_XN = 8
VP_Y = (-0.08, 0.10)
VP_YN = 10
VP_Z = (-0.09, -0.07)
VP_ZN = 2

SPHERE_R = 0.004
NEAR_THRESH = 0.01

RGBA_INSIDE = "0.0 1.0 0.0 0.9"
RGBA_NEAR = "1.0 0.85 0.0 0.85"
RGBA_FAR = "1.0 0.0 0.0 0.8"
RGBA_CAPSULE = "0.2 0.6 1.0 0.55"


def generate_volume_points() -> np.ndarray:
    x = np.linspace(VP_X[0], VP_X[1], VP_XN)
    y = np.linspace(VP_Y[0], VP_Y[1], VP_YN)
    z = np.linspace(VP_Z[0], VP_Z[1], VP_ZN)
    gx, gy, gz = np.meshgrid(x, y, z, indexing="ij")
    return np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)


def _seg_dist(px, py, pz, cx, cy_c, cz, half_len):
    y_closest = np.clip(py, cy_c - half_len, cy_c + half_len)
    return np.sqrt((px - cx) ** 2 + (py - y_closest) ** 2 + (pz - cz) ** 2)


def surface_distance(px, py, pz, capsule_y_c):
    d_min = float("inf")
    for cx in CAPSULE_XS:
        d = _seg_dist(px, py, pz, cx, capsule_y_c, CAPSULE_Z, CAPSULE_HALF_LEN)
        d_min = min(d_min, d)
    return d_min - CAPSULE_RADIUS


def classify_points(points: np.ndarray, capsule_y_c: float) -> list[str]:
    rgbs: list[str] = []
    for px, py, pz in points:
        sd = surface_distance(px, py, pz, capsule_y_c)
        if sd <= 0:
            rgbs.append(RGBA_INSIDE)
        elif sd <= NEAR_THRESH:
            rgbs.append(RGBA_NEAR)
        else:
            rgbs.append(RGBA_FAR)
    return rgbs


def build_modified_xml(xml_path: str) -> tuple[str, np.ndarray]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    compiler = root.find("compiler")
    mesh_abs = os.path.abspath(os.path.join(os.path.dirname(xml_path), "..", "meshes"))
    compiler.set("meshdir", mesh_abs)

    worldbody = root.find("worldbody")
    ground = ET.Element("geom")
    ground.set("name", "ground")
    ground.set("type", "plane")
    ground.set("size", "100 100 0.1")
    ground.set("rgba", "0.88 0.88 0.88 1")
    ground.set("condim", "3")
    worldbody.insert(0, ground)

    points = generate_volume_points()

    for foot_name, cy in [
        ("left_foot_link", LEFT_CAPSULE_Y),
        ("right_foot_link", RIGHT_CAPSULE_Y),
    ]:
        foot_body = None
        for body in worldbody.iter("body"):
            if body.get("name") == foot_name:
                foot_body = body
                break
        if foot_body is None:
            print(f"[WARN] body '{foot_name}' not found")
            continue

        for geom in foot_body.findall("geom"):
            if "collision" in (geom.get("name") or ""):
                geom.set("group", "0")
                geom.set("rgba", RGBA_CAPSULE)
                geom.set("contype", "0")
                geom.set("conaffinity", "0")

        colors = classify_points(points, cy)
        for i, ((px, py, pz), rgba) in enumerate(zip(points, colors)):
            s = ET.Element("geom")
            s.set("name", f"{foot_name}_vp{i:03d}")
            s.set("type", "sphere")
            s.set("size", str(SPHERE_R))
            s.set("pos", f"{px:.6f} {py:.6f} {pz:.6f}")
            s.set("rgba", rgba)
            s.set("group", "0")
            s.set("contype", "0")
            s.set("conaffinity", "0")
            foot_body.append(s)

    tmp = os.path.join(os.path.dirname(xml_path), "_foot_viz_tmp.xml")
    tree.write(tmp, encoding="unicode", xml_declaration=True)
    return tmp, points


def print_analysis(points: np.ndarray, capsule_y_c: float, label: str):
    sds = np.array([surface_distance(p[0], p[1], p[2], capsule_y_c) for p in points])
    n_in = int((sds <= 0).sum())
    n_near = int(((sds > 0) & (sds <= NEAR_THRESH)).sum())
    n_far = int((sds > NEAR_THRESH).sum())
    n = len(points)

    print(f"\n{'=' * 62}")
    print(f"  {label}")
    print(f"{'=' * 62}")
    print(f"  Total volume points : {n}")
    print(f"  GREEN  (inside capsule)    : {n_in:3d}  ({100 * n_in / n:5.1f}%)")
    print(f"  YELLOW (0-10mm from surface): {n_near:3d}  ({100 * n_near / n:5.1f}%)")
    print(f"  RED    (>10mm from surface): {n_far:3d}  ({100 * n_far / n:5.1f}%)")

    cap_x_lo = CAPSULE_XS[0] - CAPSULE_RADIUS
    cap_x_hi = CAPSULE_XS[-1] + CAPSULE_RADIUS
    cap_y_lo = capsule_y_c - CAPSULE_HALF_LEN
    cap_y_hi = capsule_y_c + CAPSULE_HALF_LEN
    cap_z_lo = CAPSULE_Z - CAPSULE_RADIUS
    cap_z_hi = CAPSULE_Z + CAPSULE_RADIUS

    print(f"\n  Capsule bounding box (body frame):")
    print(f"    x : [{cap_x_lo:+.4f}, {cap_x_hi:+.4f}]")
    print(f"    y : [{cap_y_lo:+.4f}, {cap_y_hi:+.4f}]")
    print(f"    z : [{cap_z_lo:+.4f}, {cap_z_hi:+.4f}]")
    print(f"\n  Volume-points grid (body frame):")
    print(f"    x : [{VP_X[0]:+.4f}, {VP_X[1]:+.4f}]  ({VP_XN} pts)")
    print(f"    y : [{VP_Y[0]:+.4f}, {VP_Y[1]:+.4f}]  ({VP_YN} pts)")
    print(f"    z : [{VP_Z[0]:+.4f}, {VP_Z[1]:+.4f}]  ({VP_ZN} pts)")

    x_vals = np.linspace(VP_X[0], VP_X[1], VP_XN)
    y_vals = np.linspace(VP_Y[0], VP_Y[1], VP_YN)

    print(f"\n  Per-x-row detail (z={VP_Z[0]:.2f} layer):")
    print(f"    {'x':>7s}  {'in':>4s} {'near':>5s} {'far':>4s}  inside_capsule_x?")
    for xi in x_vals:
        mask_x = np.abs(points[:, 0] - xi) < 1e-9
        pts_x = points[mask_x]
        sds_x = np.array(
            [surface_distance(p[0], p[1], p[2], capsule_y_c) for p in pts_x]
        )
        ni = int((sds_x <= 0).sum())
        nn = int(((sds_x > 0) & (sds_x <= NEAR_THRESH)).sum())
        nf = int((sds_x > NEAR_THRESH).sum())
        in_cap = cap_x_lo <= xi <= cap_x_hi
        tag = " <-- outside" if not in_cap else ""
        print(f"    {xi:+7.4f}  {ni:4d} {nn:5d} {nf:4d}  {'YES' if in_cap else 'NO':>3s}{tag}")

    print(f"\n  Per-y-row detail (z={VP_Z[0]:.2f} layer):")
    print(f"    {'y':>7s}  {'in':>4s} {'near':>5s} {'far':>4s}  inside_capsule_y?")
    for yi in y_vals:
        mask_y = np.abs(points[:, 1] - yi) < 1e-9
        pts_y = points[mask_y]
        sds_y = np.array(
            [surface_distance(p[0], p[1], p[2], capsule_y_c) for p in pts_y]
        )
        ni = int((sds_y <= 0).sum())
        nn = int(((sds_y > 0) & (sds_y <= NEAR_THRESH)).sum())
        nf = int((sds_y > NEAR_THRESH).sum())
        in_cap = cap_y_lo <= yi <= cap_y_hi
        tag = " <-- outside" if not in_cap else ""
        print(f"    {yi:+7.4f}  {ni:4d} {nn:5d} {nf:4d}  {'YES' if in_cap else 'NO':>3s}{tag}")


def set_standing_pose(model: mujoco.MjModel, data: mujoco.MjData):
    data.qpos[2] = 1.02
    data.qpos[3] = 1.0
    data.qpos[6] = 0.0

    joints = {
        "left_hip_yaw_joint": 0.0,
        "left_hip_pitch_joint": 0.3,
        "left_hip_roll_joint": 0.0,
        "left_knee_joint": -0.5,
        "right_hip_yaw_joint": 0.0,
        "right_hip_pitch_joint": 0.3,
        "right_hip_roll_joint": 0.0,
        "right_knee_joint": -0.5,
    }
    for name, val in joints.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            data.qpos[model.jnt_qposadr[jid]] = val

    mujoco.mj_forward(model, data)


def main():
    os.environ.setdefault("MUJOCO_GL", "glfw")

    print(f"[INFO] XML: {XML_PATH}")

    tmp_path, points = build_modified_xml(XML_PATH)
    print(f"[INFO] Modified XML: {tmp_path}")

    print_analysis(points, LEFT_CAPSULE_Y, "Left Foot")
    print_analysis(points, RIGHT_CAPSULE_Y, "Right Foot")

    model = mujoco.MjModel.from_xml_path(tmp_path)
    data = mujoco.MjData(model)
    set_standing_pose(model, data)

    print(f"\n[VIEWER] Legend:")
    print(f"  Blue transparent  = 5 foot collision capsules")
    print(f"  Green  spheres = volume points INSIDE capsule surface")
    print(f"  Yellow spheres = volume points 0-10mm FROM capsule surface")
    print(f"  Red    spheres = volume points >10mm FROM capsule surface")
    print(f"  Close viewer window to exit.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0, 0, 0.35]
        viewer.cam.distance = 1.2
        viewer.cam.elevation = -20.0
        viewer.cam.azimuth = 135.0

        mujoco.mj_forward(model, data)
        viewer.sync()

        while viewer.is_running():
            pass

    os.unlink(tmp_path)
    print("[INFO] Cleaned up. Done.")


if __name__ == "__main__":
    main()
