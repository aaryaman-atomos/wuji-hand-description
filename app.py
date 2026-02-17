"""
Interactive Hand Simulator — Streamlit + Plotly Web App
Run locally:  streamlit run app.py
Deploy to Streamlit Cloud for free sharing & Notion embedding.
"""
import xml.etree.ElementTree as ET
import itertools
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.spatial import ConvexHull

# ──────────────────────── Page config ────────────────────────
st.set_page_config(
    page_title="Hand Simulator",
    page_icon="🤚",
    layout="wide",
)

# ──────────────────────── Constants ──────────────────────────
URDF_FILE = "urdf/right.urdf"
HULL_SAMPLES = 7  # 7^4 = 2401 tip positions per finger

COLORS = {
    "Thumb":  "#e74c3c",
    "Index":  "#27ae60",
    "Middle": "#2980b9",
    "Ring":   "#f39c12",
    "Pinky":  "#8e44ad",
}

FINGER_TIPS = {
    "Thumb":  "right_finger1_tip_link",
    "Index":  "right_finger2_tip_link",
    "Middle": "right_finger3_tip_link",
    "Ring":   "right_finger4_tip_link",
    "Pinky":  "right_finger5_tip_link",
}

# URDF joint order: joint1=MCP, joint2=Abd, joint3=PIP, joint4=DIP
JOINT_ROLE_LABELS = ["MCP", "Abd", "PIP", "DIP"]
DISPLAY_ORDER = ["DIP", "PIP", "Abd", "MCP"]


# ──────────────────────── Kinematics helpers ─────────────────
def get_transform(xyz, rpy):
    roll, pitch, yaw = rpy
    cx, cy, cz = np.cos([roll, pitch, yaw])
    sx, sy, sz = np.sin([roll, pitch, yaw])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    T = np.eye(4)
    T[:3, :3] = Rz @ Ry @ Rx
    T[:3, 3] = xyz
    return T


def get_revolute_transform(axis, angle):
    u = np.asarray(axis, dtype=float)
    u = u / np.linalg.norm(u)
    x, y, z = u
    c, s = np.cos(angle), np.sin(angle)
    C = 1 - c
    R = np.array([
        [x*x*C + c,   x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, y*y*C + c,   y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, z*z*C + c],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    return T


# ──────────────────────── Parse URDF (cached) ────────────────
@st.cache_data
def load_urdf(path):
    tree = ET.parse(path)
    root = tree.getroot()

    joints = {}
    for joint in root.findall("joint"):
        name = joint.get("name")
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        jtype = joint.get("type")
        origin = joint.find("origin")
        xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()]
        rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()]
        axis_elem = joint.find("axis")
        axis = (
            [float(v) for v in axis_elem.get("xyz", "1 0 0").split()]
            if axis_elem is not None
            else [1, 0, 0]
        )
        limit = joint.find("limit")
        lower = float(limit.get("lower", 0)) if limit is not None else 0.0
        upper = float(limit.get("upper", 0)) if limit is not None else 0.0

        joints[child] = dict(
            name=name, type=jtype, parent=parent, child=child,
            xyz=xyz, rpy=rpy, axis=axis, lower=lower, upper=upper,
        )

    chains = {}
    for fname, tip in FINGER_TIPS.items():
        chain = []
        curr = tip
        while curr in joints:
            chain.append(joints[curr])
            curr = joints[curr]["parent"]
        chain.reverse()
        chains[fname] = chain

    return chains


# ──────────────────────── FK for a single chain ──────────────
def forward_kinematics(chain, angles_dict):
    """Return list of joint positions given {joint_name: angle}."""
    points = [np.array([0, 0, 0])]
    T = np.eye(4)
    for j in chain:
        T = T @ get_transform(j["xyz"], j["rpy"])
        points.append(T[:3, 3].copy())
        if j["type"] == "revolute":
            angle = angles_dict.get(j["name"], 0.0)
            T = T @ get_revolute_transform(j["axis"], angle)
    return np.array(points)


# ──────────────────────── Workspace hull (cached) ────────────
@st.cache_data
def compute_hulls(_chains):
    """Compute convex-hull triangles for every finger's workspace."""
    hulls = {}
    for fname, chain in _chains.items():
        static_T = [get_transform(j["xyz"], j["rpy"]) for j in chain]
        rev_joints = [j for j in chain if j["type"] == "revolute"]
        ranges = [np.linspace(j["lower"], j["upper"], HULL_SAMPLES) for j in rev_joints]

        tips = []
        for combo in itertools.product(*ranges):
            T = np.eye(4)
            ri = 0
            for i, j in enumerate(chain):
                T = T @ static_T[i]
                if j["type"] == "revolute":
                    T = T @ get_revolute_transform(j["axis"], combo[ri])
                    ri += 1
            tips.append(T[:3, 3].copy())
        tips = np.array(tips)

        if len(tips) >= 4:
            try:
                hull = ConvexHull(tips)
                hulls[fname] = (tips, hull.simplices)
            except Exception:
                pass
    return hulls


# ──────────────────────── Build Plotly figure ────────────────
def build_figure(chains, angles, hulls, show_hulls):
    fig = go.Figure()

    # Workspace hulls
    for fname, (tips, simplices) in hulls.items():
        if not show_hulls.get(fname, True):
            continue
        fig.add_trace(go.Mesh3d(
            x=tips[:, 0], y=tips[:, 1], z=tips[:, 2],
            i=simplices[:, 0], j=simplices[:, 1], k=simplices[:, 2],
            color=COLORS[fname],
            opacity=0.12,
            name=f"{fname} workspace",
            hoverinfo="name",
            showlegend=True,
        ))

    # Skeleton
    for fname, chain in chains.items():
        pts = forward_kinematics(chain, angles)
        # Bone line
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="lines+markers",
            line=dict(color="black", width=5),
            marker=dict(size=3, color="black"),
            name=f"{fname} skeleton",
            hoverinfo="name",
            showlegend=False,
        ))
        # Fingertip
        fig.add_trace(go.Scatter3d(
            x=[pts[-1, 0]], y=[pts[-1, 1]], z=[pts[-1, 2]],
            mode="markers",
            marker=dict(size=6, color=COLORS[fname]),
            name=f"{fname} tip",
            hoverinfo="name",
            showlegend=False,
        ))

    limit = 0.15
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-0.05, limit], title="X"),
            yaxis=dict(range=[-limit / 2, limit / 2], title="Y"),
            zaxis=dict(range=[-0.05, limit], title="Z"),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.4, y=1.4, z=0.8)),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=650,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        title=dict(text="Interactive Hand Simulator", x=0.5),
    )
    return fig


# ═══════════════════════ MAIN APP ════════════════════════════
chains = load_urdf(URDF_FILE)
hulls = compute_hulls(chains)

# ── Sidebar controls ─────────────────────────────────────────
st.sidebar.title("🤚 Hand Simulator")
st.sidebar.markdown("Adjust joint angles below to explore the hand's range of motion.")

# View presets
st.sidebar.markdown("---")
st.sidebar.markdown("##### 🔄 Workspace Visibility")
show_hulls = {}
cols_toggle = st.sidebar.columns(3)
for idx, fname in enumerate(FINGER_TIPS.keys()):
    with cols_toggle[idx % 3]:
        show_hulls[fname] = st.checkbox(fname, value=True, key=f"hull_{fname}")

st.sidebar.markdown("---")
st.sidebar.markdown("##### 🎛️ Joint Controls")

# Reset button
if st.sidebar.button("↺  Reset All Joints", use_container_width=True):
    for fname in FINGER_TIPS.keys():
        for role in JOINT_ROLE_LABELS:
            key = f"{fname}_{role}"
            if key in st.session_state:
                st.session_state[key] = 0.0
    st.rerun()

# Joint angle sliders — one expander per finger
angles = {}
for fname in FINGER_TIPS.keys():
    chain = chains[fname]
    color_hex = COLORS[fname]

    with st.sidebar.expander(f"● {fname}", expanded=True):
        # Collect revolute joints with their role labels
        rev_joints = []
        ri = 0
        for j in chain:
            if j["type"] != "revolute":
                continue
            role = JOINT_ROLE_LABELS[ri] if ri < len(JOINT_ROLE_LABELS) else f"J{ri+1}"
            rev_joints.append((role, j))
            ri += 1

        # Display in the requested order: DIP, PIP, Abd, MCP
        for role_name in DISPLAY_ORDER:
            for role, j in rev_joints:
                if role != role_name:
                    continue
                key = f"{fname}_{role}"
                lo_deg = float(np.degrees(j["lower"]))
                hi_deg = float(np.degrees(j["upper"]))

                val_deg = st.slider(
                    role,
                    min_value=lo_deg,
                    max_value=hi_deg,
                    value=0.0,
                    step=0.5,
                    format="%.1f°",
                    key=key,
                )
                angles[j["name"]] = np.radians(val_deg)
                break

# ── Main area — 3D plot ──────────────────────────────────────
fig = build_figure(chains, angles, hulls, show_hulls)
st.plotly_chart(fig, use_container_width=True, key="hand_plot")

# ── Joint limits reference table ─────────────────────────────
with st.expander("📋 Joint Limits Reference"):
    import pandas as pd

    rows = []
    for fname in FINGER_TIPS.keys():
        chain = chains[fname]
        ri = 0
        for j in chain:
            if j["type"] != "revolute":
                continue
            role = JOINT_ROLE_LABELS[ri] if ri < len(JOINT_ROLE_LABELS) else f"J{ri+1}"
            rows.append({
                "Finger": fname,
                "Joint": j["name"],
                "Role": role,
                "Lower (deg)": f"{np.degrees(j['lower']):+.2f}°",
                "Upper (deg)": f"{np.degrees(j['upper']):+.2f}°",
            })
            ri += 1
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
