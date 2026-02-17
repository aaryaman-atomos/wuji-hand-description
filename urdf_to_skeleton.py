import csv
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def rpy_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF RPY: Rz(yaw) * Ry(pitch) * Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    Rx = np.array([[1, 0, 0],
                   [0, cr, -sr],
                   [0, sr, cr]], dtype=float)
    Ry = np.array([[cp, 0, sp],
                   [0, 1, 0],
                   [-sp, 0, cp]], dtype=float)
    Rz = np.array([[cy, -sy, 0],
                   [sy, cy, 0],
                   [0, 0, 1]], dtype=float)
    return Rz @ Ry @ Rx


def T_from_xyz_rpy(xyz, rpy) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = rpy_to_R(*rpy)
    T[:3, 3] = np.array(xyz, dtype=float)
    return T


def parse_vec(s: str, default) -> np.ndarray:
    if s is None:
        return np.array(default, dtype=float)
    vals = [float(v) for v in s.strip().split()]
    if len(vals) != 3:
        raise ValueError(f"Expected 3 floats, got: {s}")
    return np.array(vals, dtype=float)


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def make_orthonormal_frame(z_axis_world: np.ndarray, R_child_world: np.ndarray):
    """Return x,y,z as orthonormal basis with z aligned to joint axis."""
    z = normalize(z_axis_world)

    # pick a stable reference direction from child frame
    ref = R_child_world @ np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(ref, z))) > 0.95:
        ref = R_child_world @ np.array([0.0, 1.0, 0.0])

    x = ref - float(np.dot(ref, z)) * z
    x = normalize(x)
    y = np.cross(z, x)
    y = normalize(y)
    return x, y, z


def main(urdf_path: str, out_csv: str):
    urdf_path = str(Path(urdf_path).resolve())
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # Collect links
    links = {l.attrib["name"] for l in root.findall("link")}

    # Collect joints
    joints = {}
    child_links = set()
    parent_to_joints = {}

    for j in root.findall("joint"):
        name = j.attrib["name"]
        jtype = j.attrib.get("type", "fixed")

        parent_link = j.find("parent").attrib["link"]
        child_link = j.find("child").attrib["link"]
        child_links.add(child_link)

        origin_el = j.find("origin")
        if origin_el is None:
            xyz = np.array([0.0, 0.0, 0.0])
            rpy = np.array([0.0, 0.0, 0.0])
        else:
            xyz = parse_vec(origin_el.attrib.get("xyz"), [0.0, 0.0, 0.0])
            rpy = parse_vec(origin_el.attrib.get("rpy"), [0.0, 0.0, 0.0])

        axis_el = j.find("axis")
        # URDF default (when axis tag is omitted) is [1,0,0] for relevant joint types
        axis_local = parse_vec(axis_el.attrib.get("xyz") if axis_el is not None else None,
                               [1.0, 0.0, 0.0])

        joints[name] = {
            "type": jtype,
            "parent": parent_link,
            "child": child_link,
            "xyz": xyz,
            "rpy": rpy,
            "axis_local": axis_local,
        }
        parent_to_joints.setdefault(parent_link, []).append(name)

    # Root link = link that is never a child
    root_candidates = list(links - child_links)
    if not root_candidates:
        raise RuntimeError("Could not find a root link (tree may be cyclic or malformed).")
    base_link = root_candidates[0]

    # Forward-kinematics at zero pose: compute world transform for each link
    link_T_world = {base_link: np.eye(4, dtype=float)}
    stack = [base_link]

    while stack:
        parent = stack.pop()
        for jname in parent_to_joints.get(parent, []):
            jinfo = joints[jname]
            child = jinfo["child"]

            T_origin = T_from_xyz_rpy(jinfo["xyz"], jinfo["rpy"])
            link_T_world[child] = link_T_world[parent] @ T_origin
            stack.append(child)

    # Write CSV
    # Coordinates are in meters because URDF xyz is meters (typical URDF convention).
    # SolidWorks API also uses meters.
    out_csv = str(Path(out_csv).resolve())
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "joint", "type", "parent_link", "child_link",
            "parent_x", "parent_y", "parent_z",
            "child_x", "child_y", "child_z",
            "axis_x", "axis_y", "axis_z",
            "xaxis_x", "xaxis_y", "xaxis_z",
            "yaxis_x", "yaxis_y", "yaxis_z",
            "zaxis_x", "zaxis_y", "zaxis_z"
        ])

        for jname, jinfo in joints.items():
            parent = jinfo["parent"]
            child = jinfo["child"]

            if parent not in link_T_world or child not in link_T_world:
                # If disconnected, skip
                continue

            Tp = link_T_world[parent]
            Tc = link_T_world[child]

            p_parent = Tp[:3, 3]
            p_child = Tc[:3, 3]
            Rc = Tc[:3, :3]

            axis_world = Rc @ normalize(jinfo["axis_local"])
            xw, yw, zw = make_orthonormal_frame(axis_world, Rc)

            w.writerow([
                jname, jinfo["type"], parent, child,
                *p_parent.tolist(),
                *p_child.tolist(),
                *axis_world.tolist(),
                *xw.tolist(),
                *yw.tolist(),
                *zw.tolist()
            ])

    print("Base link:", base_link)
    print("Wrote:", out_csv)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python urdf_to_skeleton_csv.py path/to/model.urdf out.csv")
        raise SystemExit(2)
    main(sys.argv[1], sys.argv[2])
