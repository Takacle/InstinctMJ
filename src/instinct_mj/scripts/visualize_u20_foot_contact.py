"""Visualize U20 (ring foot) wheel collision capsules and arc volume-points.

Renders, per foot:
  - the foot's collision capsules (blue, transparent) loaded from u20_popsicle_ring.xml
  - arc half-ring volume-point spheres (from ``Arc3dPointsGeneratorCfg``, the same
    generator used by the parkour task config), color-coded by proximity to the
    nearest collision capsule:
      GREEN  = inside capsule collision surface
      YELLOW = within 10 mm of capsule surface
      RED    = > 10 mm from any capsule surface

Capsule proximity is computed from the actual collision geoms in the compiled
MuJoCo model, so it auto-tracks whatever capsules the XML defines.

Usage:
    python src/instinct_mj/scripts/visualize_u20_foot_contact.py
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np

from instinct_mj.sensors.volume_points import Arc3dPointsGeneratorCfg
from instinct_mj.sensors.volume_points.points_generator import arc3d_points_generator

XML_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets",
    "resources",
    "u20",
    "xml",
    "u20_popsicle_ring.xml",
)
MESHES_DIR = os.path.abspath(os.path.join(os.path.dirname(XML_PATH), "..", "meshes"))

FOOT_BODY_NAMES = ("left_foot_link", "right_foot_link")

# Same generator/defaults as the parkour task VolumePointsCfg.
# 4 wheel centers (union of L/R feet's staggered wheels).
POINTS_CFG = Arc3dPointsGeneratorCfg(
    centers=((0.0, 0.067), (0.0, -0.067), (0.20, 0.033), (0.20, -0.033)),
)

SPHERE_R = 0.004
NEAR_THRESH = 0.01

RGBA_INSIDE = "0.0 1.0 0.0 0.9"
RGBA_NEAR = "1.0 0.85 0.0 0.85"
RGBA_FAR = "1.0 0.0 0.0 0.8"
RGBA_CAPSULE = "0.2 0.6 1.0 0.55"


def generate_volume_points() -> np.ndarray:
    """Arc half-ring point pattern in the foot body-local frame (identical for both feet)."""
    return arc3d_points_generator(POINTS_CFG).numpy()


def compile_model(xml_path: str) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(xml_path)
    spec.meshdir = MESHES_DIR
    spec.assets = {}
    return spec.compile()


def _quat_rotate_vec(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    u = np.array([x, y, z])
    return v + 2.0 * w * np.cross(u, v) + 2.0 * np.cross(u, np.cross(u, v))


def load_foot_capsules(model: mujoco.MjModel) -> dict[str, np.ndarray]:
    """Per-foot collision-capsule arrays in the foot body-local frame.

    Returns dict[foot] = array shape (N, 8) with rows
    ``[cx, cy, cz, ax, ay, az, half_len, radius]`` (center + local axis + size).
    """
    out: dict[str, np.ndarray] = {}
    for foot in FOOT_BODY_NAMES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, foot)
        rows: list[list[float]] = []
        for gi in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gi)
            if model.geom_bodyid[gi] == bid and name and "collision" in name:
                center = np.asarray(model.geom_pos[gi])
                axis = _quat_rotate_vec(np.asarray(model.geom_quat[gi]), np.array([0.0, 0.0, 1.0]))
                half_len = float(model.geom_size[gi][1])
                radius = float(model.geom_size[gi][0])
                rows.append([center[0], center[1], center[2], axis[0], axis[1], axis[2], half_len, radius])
        out[foot] = np.array(rows)
    return out


def capsule_distance(point: np.ndarray, capsules: np.ndarray) -> float:
    """Signed surface distance from point to the nearest capsule (< 0 means inside)."""
    centers = capsules[:, 0:3]
    axes = capsules[:, 3:6]
    halfs = capsules[:, 6]
    radii = capsules[:, 7]
    d = point - centers
    proj = np.clip(np.einsum("nd,nd->n", d, axes), -halfs, halfs)
    closest = centers + proj[:, None] * axes
    dist = np.linalg.norm(point - closest, axis=1)
    return float((dist - radii).min())


def classify_points(points: np.ndarray, capsules: np.ndarray) -> list[str]:
    rgbs: list[str] = []
    for p in points:
        sd = capsule_distance(p, capsules)
        if sd <= 0.0:
            rgbs.append(RGBA_INSIDE)
        elif sd <= NEAR_THRESH:
            rgbs.append(RGBA_NEAR)
        else:
            rgbs.append(RGBA_FAR)
    return rgbs


def _find_body_element(worldbody: ET.Element, name: str) -> ET.Element | None:
    for body in worldbody.iter("body"):
        if body.get("name") == name:
            return body
    return None


def build_modified_xml(
    xml_path: str,
    points: np.ndarray,
    capsules_per_foot: dict[str, np.ndarray],
) -> str:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    compiler = root.find("compiler")
    compiler.set("meshdir", MESHES_DIR)

    worldbody = root.find("worldbody")
    ground = ET.Element("geom")
    ground.set("name", "ground")
    ground.set("type", "plane")
    ground.set("size", "100 100 0.1")
    ground.set("rgba", "0.88 0.88 0.88 1")
    ground.set("condim", "3")
    worldbody.insert(0, ground)

    for foot_name in FOOT_BODY_NAMES:
        foot_body = _find_body_element(worldbody, foot_name)
        if foot_body is None:
            print(f"[WARN] body '{foot_name}' not found")
            continue

        for geom in foot_body.findall("geom"):
            if "collision" in (geom.get("name") or ""):
                geom.set("group", "0")
                geom.set("rgba", RGBA_CAPSULE)
                geom.set("contype", "0")
                geom.set("conaffinity", "0")

        colors = classify_points(points, capsules_per_foot[foot_name])
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
    return tmp


def print_analysis(points: np.ndarray, capsules: np.ndarray, label: str):
    sds = np.array([capsule_distance(p, capsules) for p in points])
    n = len(points)
    n_in = int((sds <= 0.0).sum())
    n_near = int(((sds > 0.0) & (sds <= NEAR_THRESH)).sum())
    n_far = int((sds > NEAR_THRESH).sum())

    print(f"\n{'=' * 62}")
    print(f"  {label}")
    print(f"{'=' * 62}")
    print(f"  Collision capsules        : {capsules.shape[0]}")
    print(f"  Volume points (arc)       : {n}")
    print(f"  GREEN  (inside capsule)    : {n_in:3d}  ({100 * n_in / n:5.1f}%)")
    print(f"  YELLOW (0-10mm from surface): {n_near:3d}  ({100 * n_near / n:5.1f}%)")
    print(f"  RED    (>10mm from surface): {n_far:3d}  ({100 * n_far / n:5.1f}%)")
    print(f"  min / max signed distance : {sds.min():+.4f} / {sds.max():+.4f} m")

    radii = POINTS_CFG.radii
    # Exact per-radius grouping via the known meshgrid layout (centers, radii, angles).
    n_c = len(POINTS_CFG.centers)
    sds_grid = sds.reshape(n_c, len(radii), POINTS_CFG.angle_num)
    print("\n  Per-radius-shell breakdown:")
    print(f"    {'radius':>7s}  {'in':>4s} {'near':>5s} {'far':>4s}")
    for r_idx, r in enumerate(radii):
        sds_r = sds_grid[:, r_idx, :].flatten()
        ni = int((sds_r <= 0.0).sum())
        nn = int(((sds_r > 0.0) & (sds_r <= NEAR_THRESH)).sum())
        nf = int((sds_r > NEAR_THRESH).sum())
        print(f"    {r:7.4f}  {ni:4d} {nn:5d} {nf:4d}")

    print("\n  Arc pattern (body frame):")
    print(f"    centers     : {tuple((round(x, 4), round(y, 4)) for x, y in POINTS_CFG.centers)}")
    print(f"    radii       : {tuple(round(r, 4) for r in radii)}")
    print(f"    angle range : [{POINTS_CFG.angle_min:.4f}, {POINTS_CFG.angle_max:.4f}] rad "
          f"({POINTS_CFG.angle_num} pts)")


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

    capsules_per_foot = load_foot_capsules(compile_model(XML_PATH))
    for foot, caps in capsules_per_foot.items():
        print(f"[INFO] {foot}: {caps.shape[0]} collision capsules")

    points = generate_volume_points()
    print(f"[INFO] volume points per foot: {points.shape[0]}")

    tmp_path = build_modified_xml(XML_PATH, points, capsules_per_foot)
    print(f"[INFO] Modified XML: {tmp_path}")

    print_analysis(points, capsules_per_foot["left_foot_link"], "Left Foot")
    print_analysis(points, capsules_per_foot["right_foot_link"], "Right Foot")

    model = mujoco.MjModel.from_xml_path(tmp_path)
    data = mujoco.MjData(model)
    set_standing_pose(model, data)

    print("\n[VIEWER] Legend:")
    print("  Blue transparent = foot collision capsules")
    print("  Green  spheres = volume points INSIDE capsule surface")
    print("  Yellow spheres = volume points 0-10mm FROM capsule surface")
    print("  Red    spheres = volume points >10mm FROM capsule surface")
    print("  Close viewer window to exit.")

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
