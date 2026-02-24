#!/usr/bin/env python3
"""
Generate a self-contained interactive HTML hand simulator.
All kinematics run client-side in JavaScript — zero latency.

Usage:
    python build_html.py          # outputs index.html
    python build_html.py out.html # outputs to custom path
"""
import sys
import json
import struct
import os
import xml.etree.ElementTree as ET
import itertools
import numpy as np
from scipy.spatial import ConvexHull

# ── Config ───────────────────────────────────────────────────
URDF_FILE = "urdf/right.urdf"
HULL_SAMPLES = 7
OUTPUT = sys.argv[1] if len(sys.argv) > 1 else "index.html"

COLORS = {
    "Thumb":    "#e74c3c",
    "Thumb 2":  "#1abc9c",
    "Index":    "#27ae60",
    "Middle":   "#2980b9",
    "Ring":     "#f39c12",
    "Pinky":    "#8e44ad",
}

FINGER_TIPS = {
    "Thumb":  "right_finger1_tip_link",
    "Index":  "right_finger2_tip_link",
    "Middle": "right_finger3_tip_link",
    "Ring":   "right_finger4_tip_link",
    "Pinky":  "right_finger5_tip_link",
}

# Ordered list of all fingers (including Thumb 2 which is added programmatically)
FINGER_ORDER = ["Thumb", "Thumb 2", "Index", "Middle", "Ring", "Pinky"]

JOINT_ROLE_LABELS = {"default": ["MCP", "Abd", "PIP", "DIP"],
                     "Thumb":   ["CMC", "Abd", "MCP", "IP"],
                     "Thumb 2": ["CMC", "Abd", "MCP", "IP"]}
DISPLAY_ORDER = {"default": ["DIP", "PIP", "Abd", "MCP"],
                 "Thumb":   ["IP", "MCP", "Abd", "CMC"],
                 "Thumb 2": ["IP", "MCP", "Abd", "CMC"]}


# ── STL parser ──────────────────────────────────────────────
def parse_stl(filepath):
    """Read a binary STL file. Returns (unique_verts, face_indices)."""
    with open(filepath, "rb") as f:
        f.read(80)  # skip header
        num_tris = struct.unpack("<I", f.read(4))[0]
        raw_verts = []
        for _ in range(num_tris):
            f.read(12)  # skip normal (3 floats)
            v1 = struct.unpack("<3f", f.read(12))
            v2 = struct.unpack("<3f", f.read(12))
            v3 = struct.unpack("<3f", f.read(12))
            f.read(2)   # skip attribute
            raw_verts.extend([v1, v2, v3])

    # Deduplicate vertices
    vert_map = {}
    unique = []
    faces_i, faces_j, faces_k = [], [], []
    for tri in range(num_tris):
        ids = []
        for vi in range(3):
            v = raw_verts[tri * 3 + vi]
            # Round to avoid floating-point duplicates
            key = (round(v[0], 6), round(v[1], 6), round(v[2], 6))
            if key not in vert_map:
                vert_map[key] = len(unique)
                unique.append(list(key))
            ids.append(vert_map[key])
        faces_i.append(ids[0])
        faces_j.append(ids[1])
        faces_k.append(ids[2])

    return unique, faces_i, faces_j, faces_k


# ── Kinematics (same as sim_hand.py) ────────────────────────
def get_transform(xyz, rpy):
    roll, pitch, yaw = rpy
    cx, cy, cz = np.cos([roll, pitch, yaw])
    sx, sy, sz = np.sin([roll, pitch, yaw])
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    T = np.eye(4); T[:3,:3] = Rz@Ry@Rx; T[:3,3] = xyz
    return T

def get_revolute_transform(axis, angle):
    u = np.asarray(axis, dtype=float); u /= np.linalg.norm(u)
    x,y,z = u; c=np.cos(angle); s=np.sin(angle); C=1-c
    R = np.array([[x*x*C+c,x*y*C-z*s,x*z*C+y*s],
                  [y*x*C+z*s,y*y*C+c,y*z*C-x*s],
                  [z*x*C-y*s,z*y*C+x*s,z*z*C+c]])
    T = np.eye(4); T[:3,:3] = R
    return T


# ── Parse URDF ──────────────────────────────────────────────
print(f"Parsing {URDF_FILE} ...")
tree = ET.parse(URDF_FILE)
root = tree.getroot()

joints = {}
for joint in root.findall("joint"):
    name = joint.get("name")
    parent = joint.find("parent").get("link")
    child = joint.find("child").get("link")
    jtype = joint.get("type")
    origin = joint.find("origin")
    xyz = [float(v) for v in origin.get("xyz","0 0 0").split()]
    rpy = [float(v) for v in origin.get("rpy","0 0 0").split()]
    axis_elem = joint.find("axis")
    axis = [float(v) for v in axis_elem.get("xyz","1 0 0").split()] if axis_elem is not None else [1,0,0]
    limit = joint.find("limit")
    lower = float(limit.get("lower",0)) if limit is not None else 0.0
    upper = float(limit.get("upper",0)) if limit is not None else 0.0
    joints[child] = dict(name=name,type=jtype,parent=parent,child=child,
                         xyz=xyz,rpy=rpy,axis=axis,lower=lower,upper=upper)

chains = {}
for fname, tip in FINGER_TIPS.items():
    chain = []; curr = tip
    while curr in joints:
        chain.append(joints[curr]); curr = joints[curr]["parent"]
    chain.reverse(); chains[fname] = chain


# ── Precompute static transforms (4×4 flattened) ───────────
chains_json = {}
for fname, chain in chains.items():
    labels = JOINT_ROLE_LABELS.get(fname, JOINT_ROLE_LABELS["default"])
    jlist = []
    for j in chain:
        T = get_transform(j["xyz"], j["rpy"])
        entry = {
            "name": j["name"],
            "type": j["type"],
            "child": j["child"],
            "static_T": T.flatten().tolist(),
            "axis": j["axis"],
            "lower": j["lower"],
            "upper": j["upper"],
        }
        jlist.append(entry)
    ri = 0
    for entry in jlist:
        if entry["type"] == "revolute":
            entry["role"] = labels[ri] if ri < len(labels) else f"J{ri+1}"
            ri += 1
    chains_json[fname] = jlist


# ── Add Thumb 2 (relocated CMC joint) ───────────────────────
print("Building Thumb 2 (relocated CMC) ...")

# New CMC position in URDF coordinates (meters)
# Sketch (27.46, 18.54, 5.54) mm → URDF (sketch_z, sketch_x, sketch_y)
NEW_CMC_POS = np.array([0.00554, 0.02746, 0.01854])

# Axis direction: vector from new CMC to axis point, in URDF coords
# Axis point sketch (22.55, 9.91, 32.25) → URDF (32.25, 22.55, 9.91) mm
# Direction = axis_point - cmc = (26.71, -4.91, -8.63) mm
_axis_raw = np.array([26.71, -4.91, -8.63])
NEW_CMC_AXIS = _axis_raw / np.linalg.norm(_axis_raw)

# Get old CMC's rotation matrix from its static_T
T_old_first = np.array(chains_json["Thumb"][0]["static_T"]).reshape(4, 4)
R_old = T_old_first[:3, :3]
old_axis_global = R_old[:, 1]  # local Y in global = rotation axis

# Rotate old frame so its Y-axis aligns with new axis (Rodrigues' formula)
v = np.cross(old_axis_global, NEW_CMC_AXIS)
c_ang = float(np.dot(old_axis_global, NEW_CMC_AXIS))
s_ang = float(np.linalg.norm(v))
if s_ang > 1e-10:
    vn = v / s_ang
    K = np.array([[0, -vn[2], vn[1]], [vn[2], 0, -vn[0]], [-vn[1], vn[0], 0]])
    R_align = np.eye(3) + K * s_ang + (K @ K) * (1 - c_ang)
else:
    R_align = np.eye(3) if c_ang > 0 else -np.eye(3)
R_new = R_align @ R_old

# New first-joint static transform
T_new_first = np.eye(4)
T_new_first[:3, :3] = R_new
T_new_first[:3, 3] = NEW_CMC_POS

# Clone the Thumb chain, replace first joint transform, rename joints/links
thumb2_labels = JOINT_ROLE_LABELS["Thumb 2"]
thumb2_json = []
for i, entry in enumerate(chains_json["Thumb"]):
    ne = {
        "name":     entry["name"] + "_t2",
        "type":     entry["type"],
        "child":    entry["child"] + "_t2",
        "static_T": T_new_first.flatten().tolist() if i == 0 else list(entry["static_T"]),
        "axis":     list(entry["axis"]),
        "lower":    entry["lower"],
        "upper":    entry["upper"],
    }
    thumb2_json.append(ne)
ri = 0
for ne in thumb2_json:
    if ne["type"] == "revolute":
        ne["role"] = thumb2_labels[ri] if ri < len(thumb2_labels) else f"J{ri+1}"
        ri += 1
chains_json["Thumb 2"] = thumb2_json
print(f"  CMC pos  (URDF): [{NEW_CMC_POS[0]*1000:.2f}, {NEW_CMC_POS[1]*1000:.2f}, {NEW_CMC_POS[2]*1000:.2f}] mm")
print(f"  CMC axis (URDF): [{NEW_CMC_AXIS[0]:.4f}, {NEW_CMC_AXIS[1]:.4f}, {NEW_CMC_AXIS[2]:.4f}]")


# ── Load STL meshes ─────────────────────────────────────────
print("Loading STL meshes ...")
mesh_dir = "meshes/right"
meshes_json = {}

# Palm mesh (identity transform, no chain)
palm_stl = os.path.join(mesh_dir, "right_palm_link.STL")
if os.path.exists(palm_stl):
    verts, fi, fj, fk = parse_stl(palm_stl)
    meshes_json["right_palm_link"] = {
        "verts": verts, "i": fi, "j": fj, "k": fk,
        "finger": None, "chain_idx": -1,
    }
    print(f"  palm: {len(verts)} verts, {len(fi)} faces")

# Finger link meshes — map each child link to its chain position
for fname, chain in chains.items():
    for idx, j in enumerate(chain):
        link_name = j["child"]
        stl_file = os.path.join(mesh_dir, f"{link_name}.STL")
        if os.path.exists(stl_file):
            verts, fi, fj, fk = parse_stl(stl_file)
            meshes_json[link_name] = {
                "verts": verts, "i": fi, "j": fj, "k": fk,
                "finger": fname, "chain_idx": idx,
            }

total_verts = sum(len(m["verts"]) for m in meshes_json.values())
total_faces = sum(len(m["i"]) for m in meshes_json.values())
print(f"  Total: {len(meshes_json)} meshes, {total_verts} unique verts, {total_faces} faces")


# ── Precompute workspace hulls ──────────────────────────────
print("Computing workspace hulls ...")
hulls_json = {}
for fname, chain in chains.items():
    static_T = [get_transform(j["xyz"], j["rpy"]) for j in chain]
    rev_joints = [j for j in chain if j["type"] == "revolute"]
    ranges = [np.linspace(j["lower"], j["upper"], HULL_SAMPLES) for j in rev_joints]

    tips = []
    for combo in itertools.product(*ranges):
        T = np.eye(4); ri = 0
        for i, j in enumerate(chain):
            T = T @ static_T[i]
            if j["type"] == "revolute":
                T = T @ get_revolute_transform(j["axis"], combo[ri]); ri += 1
        tips.append(T[:3,3].tolist())
    tips_arr = np.array(tips)

    if len(tips_arr) >= 4:
        try:
            hull = ConvexHull(tips_arr)
            hulls_json[fname] = {
                "vertices": tips_arr.tolist(),
                "faces_i": hull.simplices[:,0].tolist(),
                "faces_j": hull.simplices[:,1].tolist(),
                "faces_k": hull.simplices[:,2].tolist(),
            }
            print(f"  {fname}: {len(tips)} pts → {len(hull.simplices)} faces")
        except Exception as e:
            print(f"  {fname}: hull failed ({e})")

# Hull for Thumb 2 (uses chains_json since its chain was built programmatically)
t2chain = chains_json["Thumb 2"]
t2rev = [j for j in t2chain if j["type"] == "revolute"]
t2ranges = [np.linspace(j["lower"], j["upper"], HULL_SAMPLES) for j in t2rev]
t2tips = []
for combo in itertools.product(*t2ranges):
    T = np.eye(4); ri = 0
    for j in t2chain:
        T_st = np.array(j["static_T"]).reshape(4, 4)
        T = T @ T_st
        if j["type"] == "revolute":
            T = T @ get_revolute_transform(j["axis"], combo[ri]); ri += 1
    t2tips.append(T[:3, 3].tolist())
t2arr = np.array(t2tips)
if len(t2arr) >= 4:
    try:
        h2 = ConvexHull(t2arr)
        hulls_json["Thumb 2"] = {
            "vertices": t2arr.tolist(),
            "faces_i": h2.simplices[:, 0].tolist(),
            "faces_j": h2.simplices[:, 1].tolist(),
            "faces_k": h2.simplices[:, 2].tolist(),
        }
        print(f"  Thumb 2: {len(t2tips)} pts → {len(h2.simplices)} faces")
    except Exception as e:
        print(f"  Thumb 2: hull failed ({e})")

# Compute global bounding box
all_pts = []
for h in hulls_json.values():
    all_pts.extend(h["vertices"])
all_pts = np.array(all_pts)
pad = 0.02
bbox = {
    "x": [float(all_pts[:,0].min()-pad), float(all_pts[:,0].max()+pad)],
    "y": [float(all_pts[:,1].min()-pad), float(all_pts[:,1].max()+pad)],
    "z": [float(all_pts[:,2].min()-pad), float(all_pts[:,2].max()+pad)],
}


# ── Round floats for smaller JSON ───────────────────────────
def round_nested(obj, decimals=5):
    if isinstance(obj, float):
        return round(obj, decimals)
    if isinstance(obj, list):
        return [round_nested(x, decimals) for x in obj]
    if isinstance(obj, dict):
        return {k: round_nested(v, decimals) for k, v in obj.items()}
    return obj

meshes_json = round_nested(meshes_json)
chains_json = round_nested(chains_json)
hulls_json = round_nested(hulls_json)


# ── Generate HTML ───────────────────────────────────────────
print("Generating HTML ...")

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Interactive Hand Simulator</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #f7f7f7; color: #333; }}
  .app {{ display: flex; height: 100vh; }}
  .sidebar {{ width: 320px; min-width: 280px; background: #fff; border-right: 1px solid #e0e0e0;
              overflow-y: auto; padding: 16px; flex-shrink: 0; }}
  .main {{ flex: 1; display: flex; flex-direction: column; }}
  .plot-title {{ text-align: center; font-size: 18px; font-weight: 700; color: #333;
                 padding: 12px 0 4px; background: #fff; flex-shrink: 0; }}
  #plot {{ flex: 1; min-height: 0; }}
  h1 {{ font-size: 18px; margin-bottom: 8px; }}
  h2 {{ font-size: 14px; margin: 16px 0 8px; color: #555; border-bottom: 1px solid #eee; padding-bottom: 4px; }}
  .finger-group {{ margin-bottom: 12px; padding: 10px; background: #fafafa; border-radius: 8px;
                   border: 1px solid #eee; }}
  .finger-title {{ font-weight: 700; font-size: 13px; margin-bottom: 6px; }}
  .slider-row {{ display: flex; align-items: center; margin-bottom: 4px; gap: 6px; }}
  .slider-row label {{ width: 36px; font-size: 11px; font-weight: 600; color: #666; text-align: right; flex-shrink: 0; }}
  .slider-row input[type=range] {{ flex: 1; height: 6px; cursor: pointer; }}
  .slider-row .val {{ width: 52px; font-size: 11px; color: #888; text-align: left; flex-shrink: 0; }}
  .toggle-section {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 4px; }}
  .toggle-btn {{ font-size: 11px; padding: 4px 10px; border-radius: 4px; cursor: pointer;
                 border: 2px solid; font-weight: 600; transition: opacity 0.15s; }}
  .toggle-btn.off {{ opacity: 0.35; }}
  .btn {{ display: block; width: 100%; padding: 8px; margin-top: 8px; border: none; border-radius: 6px;
          background: #555; color: #fff; font-size: 12px; font-weight: 600; cursor: pointer; }}
  .btn:hover {{ background: #333; }}
  .info {{ font-size: 11px; color: #999; margin-top: 12px; line-height: 1.5; }}
</style>
</head>
<body>
<div class="app">
  <div class="sidebar" id="sidebar">
    <h1>🤚 Hand Simulator</h1>
    <p style="font-size:12px;color:#888;margin-bottom:8px;">Drag sliders to move joints in real time.</p>

    <h2>👁️ Finger Visibility</h2>
    <div class="toggle-section" id="fingerToggles"></div>

    <h2>🔄 Workspace Visibility</h2>
    <div class="toggle-section" id="hullToggles"></div>

    <h2>🦴 Mesh Visibility</h2>
    <div class="toggle-section">
      <div class="toggle-btn" id="meshToggle"
           style="border-color:#888;color:#888;background:#8881">Hand Mesh</div>
    </div>

    <h2>➤ Tip Vectors</h2>
    <div class="toggle-section">
      <div class="toggle-btn" id="vectorToggle"
           style="border-color:#555;color:#555;background:#5551">Tip Direction</div>
    </div>

    <h2>🎛️ Joint Controls</h2>
    <div id="sliderContainer"></div>
    <button class="btn" onclick="resetAll()">↺ Reset All Joints</button>

    <div class="info">
      Tip: Click &amp; drag the 3D plot to rotate. Scroll to zoom.<br>
      Joint angles are in degrees.
    </div>
  </div>
  <div class="main">
    <div class="plot-title">Interactive Hand Simulator</div>
    <div id="plot"></div>
  </div>
</div>

<script>
// ── Embedded data (precomputed by build_html.py) ─────────
const CHAINS = {json.dumps(chains_json)};
const HULLS  = {json.dumps(hulls_json)};
const MESHES = {json.dumps(meshes_json)};
const BBOX   = {json.dumps(bbox)};
const COLORS = {json.dumps(COLORS)};
const FINGER_ORDER = {json.dumps(FINGER_ORDER)};
const DISPLAY_ORDER = {json.dumps(DISPLAY_ORDER)};
const RAD2DEG = 180 / Math.PI;
const DEG2RAD = Math.PI / 180;
const MESH_COLOR = '#c8cdd0';
const MESH_OPACITY = 0.55;

// ── Matrix helpers (row-major 4×4 as flat 16-element array) ──
function mat4() {{ return [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]; }}
function mat4Copy(m) {{ return m.slice(); }}
function mat4Mul(a, b) {{
  const o = new Array(16);
  for (let r=0;r<4;r++) for (let c=0;c<4;c++) {{
    let s=0; for (let k=0;k<4;k++) s += a[r*4+k]*b[k*4+c]; o[r*4+c]=s;
  }}
  return o;
}}
function revoluteT(axis, angle) {{
  let [x,y,z]=axis, n=Math.sqrt(x*x+y*y+z*z);
  x/=n;y/=n;z/=n;
  const c=Math.cos(angle),s=Math.sin(angle),C=1-c;
  return [
    x*x*C+c,   x*y*C-z*s, x*z*C+y*s, 0,
    y*x*C+z*s, y*y*C+c,   y*z*C-x*s, 0,
    z*x*C-y*s, z*y*C+x*s, z*z*C+c,   0,
    0,0,0,1
  ];
}}
function getPos(T) {{ return [T[3], T[7], T[11]]; }}

// Transform an array of [x,y,z] vertices by a 4×4 matrix, return {{x[], y[], z[]}}
function transformVerts(verts, T) {{
  const N = verts.length;
  const xs = new Float64Array(N), ys = new Float64Array(N), zs = new Float64Array(N);
  for (let i=0; i<N; i++) {{
    const [vx,vy,vz] = verts[i];
    xs[i] = T[0]*vx + T[1]*vy + T[2]*vz + T[3];
    ys[i] = T[4]*vx + T[5]*vy + T[6]*vz + T[7];
    zs[i] = T[8]*vx + T[9]*vy + T[10]*vz + T[11];
  }}
  return {{x: Array.from(xs), y: Array.from(ys), z: Array.from(zs)}};
}}

// ── State ────────────────────────────────────────────────
const angles = {{}};
const hullVisible = {{}};
const fingerVisible = {{}};
let meshVisible = true;
let vectorVisible = true;
const SHAFT_LEN = 0.013;   // 13 mm shaft
const HEAD_LEN  = 0.004;   // 4 mm arrowhead cone
const HEAD_RAD  = 0.0018;  // 1.8 mm arrowhead base radius
const HEAD_SIDES = 8;      // octagonal cone
FINGER_ORDER.forEach(f => hullVisible[f] = true);
FINGER_ORDER.forEach(f => fingerVisible[f] = true);
FINGER_ORDER.forEach(f => {{
  CHAINS[f].forEach(j => {{ if (j.type==='revolute') angles[j.name]=0; }});
}});

// ── FK — returns joint positions (for skeleton) ──────────
function fk(chain) {{
  let T = mat4();
  const pts = [[0,0,0]];
  chain.forEach(j => {{
    T = mat4Mul(T, j.static_T);
    pts.push(getPos(T));
    if (j.type === 'revolute') {{
      T = mat4Mul(T, revoluteT(j.axis, angles[j.name] || 0));
    }}
  }});
  return pts;
}}

// ── FK — returns tip position + palmar direction (X-axis of tip frame) ──
function fkTipFrame(chain) {{
  let T = mat4();
  chain.forEach(j => {{
    T = mat4Mul(T, j.static_T);
    if (j.type === 'revolute') {{
      T = mat4Mul(T, revoluteT(j.axis, angles[j.name] || 0));
    }}
  }});
  // X-axis of the tip frame = 1st column of rotation matrix (row-major)
  // This points toward the palmar side (front of the finger)
  return {{ pos: getPos(T), dir: [T[0], T[4], T[8]] }};
}}

// ── Build an arrow (shaft line + cone head) given base, direction, color ──
function buildArrowTraces(pos, dir, color, finger) {{
  // Normalize direction
  const len = Math.sqrt(dir[0]*dir[0]+dir[1]*dir[1]+dir[2]*dir[2]);
  const d = [dir[0]/len, dir[1]/len, dir[2]/len];
  // Shaft end
  const se = [pos[0]+d[0]*SHAFT_LEN, pos[1]+d[1]*SHAFT_LEN, pos[2]+d[2]*SHAFT_LEN];
  // Arrow tip (end of head)
  const at = [se[0]+d[0]*HEAD_LEN, se[1]+d[1]*HEAD_LEN, se[2]+d[2]*HEAD_LEN];

  // Find two perpendicular vectors to d
  let perp1, perp2;
  const absD = [Math.abs(d[0]), Math.abs(d[1]), Math.abs(d[2])];
  // Choose a non-parallel reference
  const ref = absD[0] < 0.9 ? [1,0,0] : [0,1,0];
  // perp1 = d × ref  (normalized)
  perp1 = [d[1]*ref[2]-d[2]*ref[1], d[2]*ref[0]-d[0]*ref[2], d[0]*ref[1]-d[1]*ref[0]];
  const plen1 = Math.sqrt(perp1[0]*perp1[0]+perp1[1]*perp1[1]+perp1[2]*perp1[2]);
  perp1 = [perp1[0]/plen1, perp1[1]/plen1, perp1[2]/plen1];
  // perp2 = d × perp1
  perp2 = [d[1]*perp1[2]-d[2]*perp1[1], d[2]*perp1[0]-d[0]*perp1[2], d[0]*perp1[1]-d[1]*perp1[0]];

  // Build cone mesh vertices: tip + base ring
  const vx=[at[0]], vy=[at[1]], vz=[at[2]]; // index 0 = tip
  for (let k=0; k<HEAD_SIDES; k++) {{
    const a = 2*Math.PI*k/HEAD_SIDES;
    const ca = Math.cos(a)*HEAD_RAD, sa = Math.sin(a)*HEAD_RAD;
    vx.push(se[0] + perp1[0]*ca + perp2[0]*sa);
    vy.push(se[1] + perp1[1]*ca + perp2[1]*sa);
    vz.push(se[2] + perp1[2]*ca + perp2[2]*sa);
  }}
  // Faces: tip (0) to each adjacent pair on the ring
  const fi=[], fj=[], fk_=[];
  for (let k=0; k<HEAD_SIDES; k++) {{
    fi.push(0);
    fj.push(1+k);
    fk_.push(1+(k+1)%HEAD_SIDES);
  }}

  const shaft = {{
    type:'scatter3d',
    x:[pos[0], se[0]], y:[pos[1], se[1]], z:[pos[2], se[2]],
    mode:'lines', line:{{color:color, width:5}},
    name:finger+' tip vector', hoverinfo:'name', showlegend:false,
    visible: vectorVisible,
    _finger:finger, _kind:'vector_shaft',
  }};
  const head = {{
    type:'mesh3d',
    x:vx, y:vy, z:vz,
    i:fi, j:fj, k:fk_,
    color:color, opacity:1,
    flatshading:true,
    lighting:{{ambient:0.8, diffuse:0.6, specular:0.3}},
    name:finger+' tip vector', hoverinfo:'name', showlegend:false,
    visible: vectorVisible,
    _finger:finger, _kind:'vector_head',
  }};
  return [shaft, head];
}}

// ── FK — returns per-link transforms (for meshes) ────────
function fkLinkTransforms(chain) {{
  let T = mat4();
  const transforms = {{}};
  chain.forEach(j => {{
    T = mat4Mul(T, j.static_T);
    if (j.type === 'revolute') {{
      T = mat4Mul(T, revoluteT(j.axis, angles[j.name] || 0));
    }}
    transforms[j.child] = mat4Copy(T);
  }});
  return transforms;
}}

// Mesh lookup: Thumb 2 links reuse original Thumb meshes (strip _t2 suffix)
function meshLookup(linkName) {{
  if (MESHES[linkName]) return MESHES[linkName];
  if (linkName.endsWith('_t2')) return MESHES[linkName.slice(0, -3)] || null;
  return null;
}}

// ── Build initial Plotly traces ──────────────────────────
function buildTraces() {{
  const traces = [];

  // 1) Mesh traces (render first, behind everything)
  // Palm mesh — identity transform
  const palmMesh = MESHES['right_palm_link'];
  if (palmMesh) {{
    traces.push({{
      type:'mesh3d',
      x: palmMesh.verts.map(v=>v[0]),
      y: palmMesh.verts.map(v=>v[1]),
      z: palmMesh.verts.map(v=>v[2]),
      i: palmMesh.i, j: palmMesh.j, k: palmMesh.k,
      color: MESH_COLOR, opacity: MESH_OPACITY,
      flatshading: true,
      lighting: {{ambient:0.7, diffuse:0.5, specular:0.2, roughness:0.8}},
      name: 'Palm mesh', hoverinfo:'skip', showlegend:false,
      visible: meshVisible,
      _kind: 'mesh', _link: 'right_palm_link',
    }});
  }}

  // Finger link meshes
  FINGER_ORDER.forEach(f => {{
    const chain = CHAINS[f];
    const linkTransforms = fkLinkTransforms(chain);
    chain.forEach(j => {{
      const m = meshLookup(j.child);
      if (!m) return;
      const T = linkTransforms[j.child];
      const tv = transformVerts(m.verts, T);
      traces.push({{
        type:'mesh3d',
        x: tv.x, y: tv.y, z: tv.z,
        i: m.i, j: m.j, k: m.k,
        color: MESH_COLOR, opacity: MESH_OPACITY,
        flatshading: true,
        lighting: {{ambient:0.7, diffuse:0.5, specular:0.2, roughness:0.8}},
        name: j.child, hoverinfo:'skip', showlegend:false,
        visible: meshVisible,
        _kind: 'mesh', _link: j.child, _finger: f,
      }});
    }});
  }});

  // 2) Workspace hulls
  FINGER_ORDER.forEach(f => {{
    const h = HULLS[f];
    if (!h) return;
    const v = h.vertices, xs=[], ys=[], zs=[];
    v.forEach(p => {{ xs.push(p[0]); ys.push(p[1]); zs.push(p[2]); }});
    traces.push({{
      type:'mesh3d', x:xs, y:ys, z:zs,
      i:h.faces_i, j:h.faces_j, k:h.faces_k,
      color: COLORS[f], opacity: 0.12,
      name: f+' workspace', hoverinfo:'name', showlegend:true,
      visible: hullVisible[f],
      _finger: f, _kind: 'hull',
    }});
  }});

  // 3) Skeleton lines + tips (on top)
  FINGER_ORDER.forEach(f => {{
    const pts = fk(CHAINS[f]);
    traces.push({{
      type:'scatter3d',
      x:pts.map(p=>p[0]), y:pts.map(p=>p[1]), z:pts.map(p=>p[2]),
      mode:'lines+markers',
      line:{{color:'black',width:5}}, marker:{{size:3,color:'black'}},
      name:f+' skeleton', hoverinfo:'name', showlegend:false,
      _finger:f, _kind:'skel',
    }});
    traces.push({{
      type:'scatter3d',
      x:[pts[pts.length-1][0]], y:[pts[pts.length-1][1]], z:[pts[pts.length-1][2]],
      mode:'markers', marker:{{size:6,color:COLORS[f]}},
      name:f+' tip', hoverinfo:'name', showlegend:false,
      _finger:f, _kind:'tip',
    }});
  }});

  // 4) Tip direction vectors (line + arrowhead — palmar direction)
  FINGER_ORDER.forEach(f => {{
    const tip = fkTipFrame(CHAINS[f]);
    const arrows = buildArrowTraces(tip.pos, tip.dir, COLORS[f], f);
    arrows.forEach(t => traces.push(t));
  }});

  return traces;
}}

const layout = {{
  template: 'plotly_white',
  paper_bgcolor:'white', plot_bgcolor:'white',
  scene: {{
    xaxis:{{range:BBOX.x, title:'X', backgroundcolor:'white', gridcolor:'#e0e0e0', showbackground:true}},
    yaxis:{{range:BBOX.y, title:'Y', backgroundcolor:'white', gridcolor:'#e0e0e0', showbackground:true}},
    zaxis:{{range:BBOX.z, title:'Z', backgroundcolor:'white', gridcolor:'#e0e0e0', showbackground:true}},
    aspectmode:'cube',
    camera:{{eye:{{x:1.4,y:1.4,z:0.8}}}},
    bgcolor:'white',
  }},
  margin:{{l:0,r:0,t:50,b:0}},
  legend:{{orientation:'h',yanchor:'bottom',y:1.02,xanchor:'center',x:0.5,font:{{size:11,color:'#333'}}}},
  showlegend:true,
}};

const initialTraces = buildTraces();
Plotly.newPlot('plot', initialTraces, layout, {{responsive:true}});

// ── Fast update — batch all restyle calls ────────────────
function updatePlot() {{
  const data = document.getElementById('plot').data;

  // Precompute all link transforms for all fingers
  const allLinkT = {{}};
  FINGER_ORDER.forEach(f => {{
    Object.assign(allLinkT, fkLinkTransforms(CHAINS[f]));
  }});

  // Batch updates for scatter3d / mesh3d (x, y, z only)
  const indices = [];
  const xBatch = [], yBatch = [], zBatch = [];

  // Precompute tip frames for arrow updates
  const tipFrames = {{}};
  FINGER_ORDER.forEach(f => {{ tipFrames[f] = fkTipFrame(CHAINS[f]); }});

  data.forEach((tr, idx) => {{
    if (tr._kind === 'skel') {{
      const pts = fk(CHAINS[tr._finger]);
      indices.push(idx);
      xBatch.push(pts.map(p=>p[0]));
      yBatch.push(pts.map(p=>p[1]));
      zBatch.push(pts.map(p=>p[2]));
    }} else if (tr._kind === 'tip') {{
      const pts = fk(CHAINS[tr._finger]);
      const tip = pts[pts.length-1];
      indices.push(idx);
      xBatch.push([tip[0]]);
      yBatch.push([tip[1]]);
      zBatch.push([tip[2]]);
    }} else if (tr._kind === 'mesh' && tr._finger) {{
      const m = meshLookup(tr._link);
      if (!m) return;
      const T = allLinkT[tr._link];
      if (!T) return;
      const tv = transformVerts(m.verts, T);
      indices.push(idx);
      xBatch.push(tv.x);
      yBatch.push(tv.y);
      zBatch.push(tv.z);
    }} else if (tr._kind === 'vector_shaft' || tr._kind === 'vector_head') {{
      // Rebuild arrow geometry from new tip frame
      const tf = tipFrames[tr._finger];
      const arrows = buildArrowTraces(tf.pos, tf.dir, COLORS[tr._finger], tr._finger);
      const arrowTr = tr._kind === 'vector_shaft' ? arrows[0] : arrows[1];
      indices.push(idx);
      xBatch.push(arrowTr.x);
      yBatch.push(arrowTr.y);
      zBatch.push(arrowTr.z);
    }}
  }});

  if (indices.length > 0) {{
    Plotly.restyle('plot', {{x: xBatch, y: yBatch, z: zBatch}}, indices);
  }}

  // Also update mesh3d face indices for arrowheads (i, j, k don't change
  // but Plotly needs them when x/y/z length changes — they stay the same here)
}}

// ── Build sidebar sliders ────────────────────────────────
const sliderContainer = document.getElementById('sliderContainer');
const sliderEls = {{}};

FINGER_ORDER.forEach(f => {{
  const group = document.createElement('div');
  group.className = 'finger-group';

  const title = document.createElement('div');
  title.className = 'finger-title';
  title.style.color = COLORS[f];
  title.textContent = '● ' + f;
  group.appendChild(title);

  const revJoints = [];
  CHAINS[f].forEach(j => {{ if (j.type==='revolute') revJoints.push(j); }});

  const order = DISPLAY_ORDER[f] || DISPLAY_ORDER['default'];
  order.forEach(role => {{
    const j = revJoints.find(jj => jj.role === role);
    if (!j) return;

    const row = document.createElement('div');
    row.className = 'slider-row';

    const lbl = document.createElement('label');
    lbl.textContent = role;

    const input = document.createElement('input');
    input.type = 'range';
    input.min = (j.lower * RAD2DEG).toFixed(1);
    input.max = (j.upper * RAD2DEG).toFixed(1);
    input.value = '0';
    input.step = '0.5';
    input.style.accentColor = COLORS[f];

    const val = document.createElement('span');
    val.className = 'val';
    val.textContent = '+0.0°';

    input.addEventListener('input', () => {{
      const deg = parseFloat(input.value);
      angles[j.name] = deg * DEG2RAD;
      val.textContent = (deg >= 0 ? '+' : '') + deg.toFixed(1) + '°';
      updatePlot();
    }});

    sliderEls[j.name] = {{input, val}};
    row.appendChild(lbl);
    row.appendChild(input);
    row.appendChild(val);
    group.appendChild(row);
  }});

  sliderContainer.appendChild(group);
}});

// ── Finger visibility toggle buttons ─────────────────────
const fingerToggleContainer = document.getElementById('fingerToggles');
['Thumb', 'Thumb 2'].forEach(f => {{
  const btn = document.createElement('div');
  btn.className = 'toggle-btn';
  btn.textContent = f;
  btn.style.borderColor = COLORS[f];
  btn.style.color = COLORS[f];
  btn.style.backgroundColor = COLORS[f] + '18';

  btn.addEventListener('click', () => {{
    fingerVisible[f] = !fingerVisible[f];
    btn.classList.toggle('off', !fingerVisible[f]);
    const vis = fingerVisible[f];
    const data = document.getElementById('plot').data;
    const indices = [];
    data.forEach((tr, idx) => {{
      if (tr._finger === f && (tr._kind === 'skel' || tr._kind === 'tip'
          || tr._kind === 'mesh' || tr._kind === 'vector_shaft'
          || tr._kind === 'vector_head')) {{
        indices.push(idx);
      }}
    }});
    if (indices.length > 0) {{
      Plotly.restyle('plot', {{visible: vis}}, indices);
    }}
  }});
  fingerToggleContainer.appendChild(btn);
}});

// ── Hull toggle buttons ──────────────────────────────────
const toggleContainer = document.getElementById('hullToggles');
FINGER_ORDER.forEach(f => {{
  const btn = document.createElement('div');
  btn.className = 'toggle-btn';
  btn.textContent = f;
  btn.style.borderColor = COLORS[f];
  btn.style.color = COLORS[f];
  btn.style.backgroundColor = COLORS[f] + '18';

  btn.addEventListener('click', () => {{
    hullVisible[f] = !hullVisible[f];
    btn.classList.toggle('off', !hullVisible[f]);
    const data = document.getElementById('plot').data;
    data.forEach((tr, idx) => {{
      if (tr._kind === 'hull' && tr._finger === f) {{
        Plotly.restyle('plot', {{visible: hullVisible[f]}}, [idx]);
      }}
    }});
  }});
  toggleContainer.appendChild(btn);
}});

// ── Mesh toggle button ───────────────────────────────────
const meshToggleBtn = document.getElementById('meshToggle');
meshToggleBtn.addEventListener('click', () => {{
  meshVisible = !meshVisible;
  meshToggleBtn.classList.toggle('off', !meshVisible);
  const data = document.getElementById('plot').data;
  const indices = [];
  data.forEach((tr, idx) => {{
    if (tr._kind === 'mesh') indices.push(idx);
  }});
  if (indices.length > 0) {{
    Plotly.restyle('plot', {{visible: meshVisible}}, indices);
  }}
}});

// ── Vector toggle button ─────────────────────────────────
const vectorToggleBtn = document.getElementById('vectorToggle');
vectorToggleBtn.addEventListener('click', () => {{
  vectorVisible = !vectorVisible;
  vectorToggleBtn.classList.toggle('off', !vectorVisible);
  const data = document.getElementById('plot').data;
  const indices = [];
  data.forEach((tr, idx) => {{
    if (tr._kind === 'vector_shaft' || tr._kind === 'vector_head') indices.push(idx);
  }});
  if (indices.length > 0) {{
    Plotly.restyle('plot', {{visible: vectorVisible}}, indices);
  }}
}});

// ── Reset ────────────────────────────────────────────────
function resetAll() {{
  Object.keys(angles).forEach(k => angles[k] = 0);
  Object.entries(sliderEls).forEach(([name, el]) => {{
    el.input.value = '0';
    el.val.textContent = '+0.0°';
  }});
  updatePlot();
}}
</script>
</body>
</html>
"""

with open(OUTPUT, "w") as f:
    f.write(html)
size_kb = len(html) // 1024
print(f"✅  Written to {OUTPUT}  ({size_kb} KB / {size_kb/1024:.1f} MB)")
print(f"   Open in any browser — or host on GitHub Pages / Notion embed.")
