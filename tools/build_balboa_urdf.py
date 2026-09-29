"""
Convert the Onshape export of the Pololu Balboa 32U4 kit into an Isaac Lab ready URDF.

Input : cad/balboa-balancing-robot-kit/balboa_balancing_robot_kit/  (Onshape URDF + glTF meshes)
Output: src/Balance_Car_RL/assets/data/balboa/{balboa.urdf, imu_mount.json, meshes/*.stl}

What this script fixes compared to the raw export:
  1. Pose: the CAD is modelled tilted, so the model is rotated about the axle until the centre of mass is
     straight above it (the equilibrium of the inverted pendulum). Frame: x forward, y left, z up, origin on the axle.
  2. Mass: the export has no usable mass (1e-5 kg parts, none on the base). Masses (BASE_MASS_KG, WHEEL_MASS_KG)
     come from src/Balance_Car_RL/car/car_cfg.py: nominal estimates spread over the CAD volume; replace them with the
     weighed values of the real robot and rerun.
  3. Wheel joints: both axes point along +y, so a positive velocity drives the car forward on both sides.
  4. Meshes: glTF (unsupported by the URDF importer, 100k+ triangles) -> decimated STL; collision uses primitives.
  5. The export double-counts a -19.86 mm offset on the board and buzzer meshes; it is removed here.
  6. imu_mount.json: pose of the IMU (control board axes) in the car frame, for the IMU sensor of the simulation.

Run inside the Isaac Lab conda env:  python tools/build_balboa_urdf.py
"""

from __future__ import annotations

import json
import pathlib
import xml.etree.ElementTree as ET

import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial.transform import Rotation as R

from Balance_Car_RL.car.car_cfg import BASE_MASS_KG, WHEEL_MASS_KG  # masses live in car_cfg.py

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "cad/balboa-balancing-robot-kit/balboa_balancing_robot_kit"
OUT = ROOT / "src/Balance_Car_RL/assets/data/balboa"

BASE_PARTS = ("chassis__NONE", "battery_cover__Boss_Extrude_logo", "bal01a01__bal01a01", "bal01a01__buzzer")
MESH_TRIANGLES = {
    "chassis__NONE": 6000,
    "battery_cover__Boss_Extrude_logo": 3000,
    "bal01a01__bal01a01": 6000,
    "bal01a01__buzzer": 400,
    "wheel_black__Body_Move_Copy1": 3000,
}
EXPORT_OFFSET_BUG_LINK = "bal01a01__1_"
COLLISION_THICKNESS_M = {"chassis__NONE": 0.03}
"""Hand-tuned thickness [m] (CAD z axis) of a collision box, instead of the 0.04741 m bounding box. It was edited
by hand in balboa.urdf; kept here so that a rebuild does not undo it."""

RX_M90 = R.from_euler("x", -np.pi / 2).as_matrix()  # wheel link frame: local z -> +y of the car


def _T(origin: ET.Element) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = R.from_euler("xyz", [float(v) for v in origin.get("rpy").split()]).as_matrix()
    m[:3, 3] = [float(v) for v in origin.get("xyz").split()]
    return m


def _load(name: str) -> trimesh.Trimesh:
    scene = trimesh.load(SRC / "meshes" / f"{name}.gltf", force="scene")
    return scene.to_geometry()


def _decimate(mesh: trimesh.Trimesh, triangles: int) -> trimesh.Trimesh:
    o3 = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices), o3d.utility.Vector3iVector(mesh.faces))
    o3 = o3.simplify_quadric_decimation(triangles)
    return trimesh.Trimesh(np.asarray(o3.vertices), np.asarray(o3.triangles))


def _hull_inertia(mesh: trimesh.Trimesh, mass: float) -> tuple[np.ndarray, np.ndarray]:
    """Inertia about the COM of the convex hull scaled to `mass` (uniform density), and the COM."""
    hull = mesh.convex_hull
    hull.density = 1.0
    return hull.moment_inertia * (mass / hull.volume), hull.center_mass


def _fmt(a) -> str:
    return " ".join(f"{v:.6g}" for v in a)


def main() -> None:
    root = ET.parse(SRC / "urdf/balboa_balancing_robot_kit.urdf").getroot()
    parent = {
        j.find("child").get("link"): (j.find("parent").get("link"), _T(j.find("origin")), j)
        for j in root.findall("joint")
    }
    colors = {}

    def world(link: str) -> np.ndarray:
        m = np.eye(4)
        while link in parent:
            p, t, _ = parent[link]
            m, link = t @ m, p
        return m

    # -- 1. every mesh in the CAD frame [m]
    cad: dict[str, trimesh.Trimesh] = {}
    for link in root.findall("link"):
        for vis in link.findall("visual"):
            f = pathlib.Path(vis.find("geometry/mesh").get("filename")).stem
            colors[f] = vis.find("material/color").get("rgba")
            m = _load(f)
            place = np.eye(4) if link.get("name") == EXPORT_OFFSET_BUG_LINK else _T(vis.find("origin"))
            m.apply_transform(world(link.get("name")) @ place)
            cad[f"{link.get('name')}|{f}"] = m
    wheel_meshes = {k: v for k, v in cad.items() if k.endswith("wheel_black__Body_Move_Copy1")}
    base_meshes = {k.split("|")[1]: v for k, v in cad.items() if k.split("|")[1] in BASE_PARTS}

    joints = {parent[c][2].get("name"): world(c)[:3, 3] for c in parent if "wheel" in c}
    left_cad, right_cad = joints["revolute_1"], joints["revolute_2"]  # x = -0.0535 (left), +0.0535 (right)
    axle = (left_cad + right_cad) / 2

    # -- 2. cad -> car frame (x forward = cad y, y left = -cad x, z up), COM above the axle
    swap = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], float)
    vols = {k: abs(m.volume) for k, m in base_meshes.items()}
    total_vol = sum(vols.values())
    mass = {k: BASE_MASS_KG * v / total_vol for k, v in vols.items()}
    com0 = sum(mass[k] * (base_meshes[k].convex_hull.center_mass - axle) for k in mass) / BASE_MASS_KG
    com0 = swap @ com0
    alpha = np.arctan2(-com0[0], com0[2])
    tilt = R.from_euler("y", alpha).as_matrix()
    M = tilt @ swap

    def to_car(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        m = mesh.copy()
        m.vertices = (M @ (m.vertices - axle).T).T
        return m

    car = {k: to_car(m) for k, m in base_meshes.items()}
    min_z = min(m.bounds[0, 2] for m in car.values())

    # -- 3. base inertial (parallel-axis sum of the parts)
    parts = {k: _hull_inertia(car[k], mass[k]) for k in car}
    com = sum(mass[k] * parts[k][1] for k in car) / BASE_MASS_KG
    inertia = np.zeros((3, 3))
    for k, (i_k, c_k) in parts.items():
        d = c_k - com
        inertia += i_k + mass[k] * (np.dot(d, d) * np.eye(3) - np.outer(d, d))

    # -- 4. write meshes + URDF
    (OUT / "meshes").mkdir(parents=True, exist_ok=True)
    for f in list(OUT.glob("meshes/*.stl")):
        f.unlink()
    robot = ET.Element("robot", name="balboa_balancing_robot_kit")
    robot.append(
        ET.Comment(
            " Generated by tools/build_balboa_urdf.py from the Onshape export of the Pololu Balboa 32U4 kit. "
            "Frame: x forward, y left, z up, origin on the wheel axle. Masses are nominal estimates. "
        )
    )
    base = ET.SubElement(robot, "link", name="base_link")
    inert = ET.SubElement(base, "inertial")
    ET.SubElement(inert, "origin", xyz=_fmt(com), rpy="0 0 0")
    ET.SubElement(inert, "mass", value=f"{BASE_MASS_KG:.6g}")
    ET.SubElement(
        inert,
        "inertia",
        ixx=f"{inertia[0, 0]:.6g}",
        ixy=f"{inertia[0, 1]:.6g}",
        ixz=f"{inertia[0, 2]:.6g}",
        iyy=f"{inertia[1, 1]:.6g}",
        iyz=f"{inertia[1, 2]:.6g}",
        izz=f"{inertia[2, 2]:.6g}",
    )
    for name, m in car.items():
        stl = f"base_{name.split('__')[0]}_{name.split('__')[-1]}.stl".replace("__", "_").lower()
        _decimate(m, MESH_TRIANGLES[name]).export(OUT / "meshes" / stl)
        vis = ET.SubElement(base, "visual")
        ET.SubElement(vis, "origin", xyz="0 0 0", rpy="0 0 0")
        ET.SubElement(ET.SubElement(vis, "geometry"), "mesh", filename=f"meshes/{stl}")
        mat = ET.SubElement(vis, "material", name=stl[:-4])
        ET.SubElement(mat, "color", rgba=colors[name])
    # collision: oriented boxes for the three big parts, expressed in the car frame
    for name in ("chassis__NONE", "battery_cover__Boss_Extrude_logo", "bal01a01__bal01a01"):
        lo, hi = base_meshes[name].bounds
        centre = M @ ((lo + hi) / 2 - axle)
        size = hi - lo
        if name in COLLISION_THICKNESS_M:
            size[2] = COLLISION_THICKNESS_M[name]
        col = ET.SubElement(base, "collision")
        ET.SubElement(col, "origin", xyz=_fmt(centre), rpy=_fmt(R.from_matrix(M).as_euler("xyz")))
        ET.SubElement(ET.SubElement(col, "geometry"), "box", size=_fmt(size))

    wheel_r = float(np.mean([m.extents[1:].max() / 2 for m in wheel_meshes.values()]))
    wheel_w = float(np.mean([m.extents[0] for m in wheel_meshes.values()]))
    for side, key_joint in (("left", left_cad), ("right", right_cad)):
        name = "left_wheel" if side == "left" else "right_wheel"
        pos = swap @ (key_joint - axle)  # wheel joint origin in the car frame (tilt is about y, axle is on it)
        link = ET.SubElement(robot, "link", name=name)
        mesh_cad = [v for k, v in wheel_meshes.items() if (v.centroid[0] < 0) == (side == "left")][0]
        m_car = mesh_cad.copy()
        m_car.vertices = (RX_M90.T @ ((swap @ (m_car.vertices - axle).T).T - pos).T).T  # into the wheel link frame
        wheel_i, wheel_c = _hull_inertia(m_car, WHEEL_MASS_KG)
        inert = ET.SubElement(link, "inertial")
        ET.SubElement(inert, "origin", xyz=_fmt(wheel_c), rpy="0 0 0")
        ET.SubElement(inert, "mass", value=f"{WHEEL_MASS_KG:.6g}")
        ET.SubElement(
            inert,
            "inertia",
            ixx=f"{wheel_i[0, 0]:.6g}",
            ixy="0",
            ixz="0",
            iyy=f"{wheel_i[1, 1]:.6g}",
            iyz="0",
            izz=f"{wheel_i[2, 2]:.6g}",
        )
        stl = f"{name}.stl"
        _decimate(m_car, MESH_TRIANGLES["wheel_black__Body_Move_Copy1"]).export(OUT / "meshes" / stl)
        vis = ET.SubElement(link, "visual")
        ET.SubElement(vis, "origin", xyz="0 0 0", rpy="0 0 0")
        ET.SubElement(ET.SubElement(vis, "geometry"), "mesh", filename=f"meshes/{stl}")
        mat = ET.SubElement(vis, "material", name=name)
        ET.SubElement(mat, "color", rgba=colors["wheel_black__Body_Move_Copy1"])
        col = ET.SubElement(link, "collision")
        ET.SubElement(col, "origin", xyz=_fmt(wheel_c), rpy="0 0 0")
        ET.SubElement(ET.SubElement(col, "geometry"), "cylinder", radius=f"{wheel_r:.6g}", length=f"{wheel_w:.6g}")
        joint = ET.SubElement(robot, "joint", name=f"{name}_joint", type="continuous")
        ET.SubElement(joint, "origin", xyz=_fmt(pos), rpy=_fmt([-np.pi / 2, 0, 0]))
        ET.SubElement(joint, "axis", xyz="0 0 1")
        ET.SubElement(joint, "parent", link="base_link")
        ET.SubElement(joint, "child", link=name)
    ET.indent(robot)
    ET.ElementTree(robot).write(OUT / "balboa.urdf", encoding="utf-8", xml_declaration=True)

    # IMU mount: the control board carries the IMU, so its axes are the board (CAD) axes, tilted with the chassis
    lo, hi = base_meshes["bal01a01__bal01a01"].bounds
    mount = {
        "comment": "IMU frame (control board axes) in the car frame; v_car = rotation @ v_imu. Generated by "
        "tools/build_balboa_urdf.py, read by car/car_cfg.py.",
        "position_m": [round(float(v), 6) for v in M @ ((lo + hi) / 2 - axle)],
        "rotation_car_from_imu": [[round(float(v), 9) for v in row] for row in M],
        "quat_xyzw": [round(float(v), 9) for v in R.from_matrix(M).as_quat()],
    }
    (OUT / "imu_mount.json").write_text(json.dumps(mount, indent=2) + "\n")

    print(f"tilt alpha        : {np.degrees(alpha):7.2f} deg (CAD pose -> upright)")
    print(f"base mass / COM   : {BASE_MASS_KG:.3f} kg, COM (x, z) = ({com[0] * 1e3:.1f}, {com[2] * 1e3:.1f}) mm")
    print(f"base inertia diag : {np.diag(inertia)}")
    print(f"wheel r / width   : {wheel_r * 1000:.1f} / {wheel_w * 1000:.1f} mm, wheel Izz = {wheel_i[2, 2]:.3e}")
    print(
        f"lowest base point : {min_z * 1000:.1f} mm vs wheel bottom {-wheel_r * 1000:.1f} mm (clearance "
        f"{(min_z + wheel_r) * 1000:.1f} mm)"
    )
    print(
        f"wheel joints at   : left {np.round(swap @ (left_cad - axle), 4)}, "
        f"right {np.round(swap @ (right_cad - axle), 4)}"
    )


if __name__ == "__main__":
    main()
