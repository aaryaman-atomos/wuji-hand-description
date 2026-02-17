import xml.etree.ElementTree as ET
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull

# --- CONFIGURATION ---
URDF_FILE = 'urdf/right.urdf'
MESH_RES = 10         # Lower resolution for the background mesh to keep it fast
SHOW_MESH = True      # Set to False if you only want the skeleton

# --- KINEMATICS ENGINE ---
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
    u = np.array(axis)
    u = u / np.linalg.norm(u)
    x, y, z = u
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c
    R = np.array([
        [x*x*C + c,    x*y*C - z*s,  x*z*C + y*s],
        [y*x*C + z*s,  y*y*C + c,    y*z*C - x*s],
        [z*x*C - y*s,  z*y*C + x*s,  z*z*C + c]
    ])
    T = np.eye(4)
    T[:3, :3] = R
    return T

# --- PARSE URDF ---
print(f"Loading {URDF_FILE}...")
tree = ET.parse(URDF_FILE)
root = tree.getroot()

joints = {}
links = {} # To store parent/child relationships easily

for joint in root.findall('joint'):
    name = joint.get('name')
    parent = joint.find('parent').get('link')
    child = joint.find('child').get('link')
    jtype = joint.get('type')
    origin = joint.find('origin')
    xyz = [float(x) for x in origin.get('xyz', '0 0 0').split()]
    rpy = [float(x) for x in origin.get('rpy', '0 0 0').split()]
    axis_elem = joint.find('axis')
    axis = [float(x) for x in axis_elem.get('xyz', '1 0 0').split()] if axis_elem is not None else [1,0,0]
    limit = joint.find('limit')
    lower = float(limit.get('lower', 0)) if limit is not None else 0.0
    upper = float(limit.get('upper', 0)) if limit is not None else 0.0
    
    joints[child] = {
        'name': name, 'type': jtype, 'parent': parent, 'child': child,
        'xyz': xyz, 'rpy': rpy, 'axis': axis, 'lower': lower, 'upper': upper,
        'current_val': 0.0 # Default position
    }

# Define Chains (Tip -> Base)
finger_tips = {
    'Thumb':  'right_finger1_tip_link',
    'Index':  'right_finger2_tip_link',
    'Middle': 'right_finger3_tip_link',
    'Ring':   'right_finger4_tip_link',
    'Pinky':  'right_finger5_tip_link'
}

# Build ordered chains for FK
chains = {}
for name, tip in finger_tips.items():
    chain = []
    curr = tip
    while curr in joints:
        chain.append(joints[curr])
        curr = joints[curr]['parent']
    chain.reverse()
    chains[name] = chain

# --- PRINT JOINT LIMITS TABLE ---
joint_labels = {
    'default': ['MCP', 'Abd', 'PIP', 'DIP'],
    'Thumb':   ['CMC', 'Abd', 'MCP', 'IP'],
}
print("\n" + "=" * 85)
print(f"  {'FINGER':<8} {'JOINT':<30} {'ROLE':<6} {'LOWER (rad)':<13} {'UPPER (rad)':<13} {'LOWER (deg)':<13} {'UPPER (deg)'}")
print("-" * 85)
for fname in finger_tips.keys():
    chain = chains[fname]
    labels = joint_labels.get(fname, joint_labels['default'])
    rev_idx = 0
    for j in chain:
        if j['type'] != 'revolute':
            continue
        role = labels[rev_idx] if rev_idx < len(labels) else f'J{rev_idx+1}'
        lo, hi = j['lower'], j['upper']
        print(f"  {fname:<8} {j['name']:<30} {role:<6} {lo:>+10.4f}   {hi:>+10.4f}   {np.degrees(lo):>+10.2f}°  {np.degrees(hi):>+10.2f}°")
        rev_idx += 1
    print()
print("=" * 85 + "\n")

# --- PLOTTING SETUP ---
fig = plt.figure(figsize=(16, 11))
fig.patch.set_facecolor('#f0f0f0')

# Layout regions (in figure-fraction coords: [left, bottom, width, height])
# 3D plot takes the top-left area
ax = fig.add_axes([0.02, 0.36, 0.68, 0.60], projection='3d')

colors = {'Thumb': '#e74c3c', 'Index': '#27ae60', 'Middle': '#2980b9', 'Ring': '#f39c12', 'Pinky': '#8e44ad'}
skeleton_lines = {}
tip_markers = {}
mesh_artists = {}   # Store convex hull artists for toggling
mesh_visible = {name: True for name in finger_tips.keys()}

# --- HELPER: COMPUTE WORKSPACE CONVEX HULL ---
HULL_SAMPLES = 7  # Samples per joint (7^4 = 2401 tip positions per finger)

def compute_tip_positions(chain, samples=HULL_SAMPLES):
    """Compute fingertip positions by sampling all revolute joints independently."""
    static_T = [get_transform(j['xyz'], j['rpy']) for j in chain]
    revolute_joints = [j for j in chain if j['type'] == 'revolute']

    # Generate sample values for each revolute joint within its URDF limits
    joint_ranges = [np.linspace(j['lower'], j['upper'], samples) for j in revolute_joints]

    tips = []
    for combo in itertools.product(*joint_ranges):
        T = np.eye(4)
        rev_idx = 0
        for i, j in enumerate(chain):
            T = T @ static_T[i]
            if j['type'] == 'revolute':
                T = T @ get_revolute_transform(j['axis'], combo[rev_idx])
                rev_idx += 1
        tips.append(T[:3, 3].copy())
    return np.array(tips)

def plot_workspace_hulls(ax):
    """Generate and plot the convex-hull workspace envelope for each finger."""
    if not SHOW_MESH:
        return
    print("Computing workspace convex hulls (this may take a moment)...")
    for name, chain in chains.items():
        tips = compute_tip_positions(chain)
        if len(tips) < 4:
            continue
        try:
            hull = ConvexHull(tips)
            triangles = [tips[simplex] for simplex in hull.simplices]
            poly = Poly3DCollection(triangles, alpha=0.12,
                                    facecolor=colors[name],
                                    edgecolor=colors[name],
                                    linewidth=0.15)
            ax.add_collection3d(poly)
            mesh_artists[name] = [poly]
            print(f"  {name:>7}: {len(tips)} samples → {len(hull.simplices)} hull faces")
        except Exception as e:
            print(f"  {name:>7}: hull failed ({e})")
    print("Done.\n")

def toggle_mesh(name):
    """Toggle visibility of a finger's workspace hull."""
    if name in mesh_artists:
        mesh_visible[name] = not mesh_visible[name]
        for artist in mesh_artists[name]:
            artist.set_visible(mesh_visible[name])
        fig.canvas.draw_idle()

if SHOW_MESH:
    plot_workspace_hulls(ax)

# --- HELPER: UPDATE SKELETON ---
def update_skeleton(val=None):
    for name, chain in chains.items():
        points = [[0,0,0]] # Start at palm origin (approx)
        T = np.eye(4)
        
        for j in chain:
            # 1. Apply Static Transform (Parent -> Joint)
            T = T @ get_transform(j['xyz'], j['rpy'])
            points.append(T[:3, 3]) # Joint location
            
            # 2. Apply Dynamic Rotation (Joint Axis * Slider Value)
            if j['type'] == 'revolute':
                angle = j['current_val']
                T = T @ get_revolute_transform(j['axis'], angle)
        
        # Extract X, Y, Z for plotting
        pts = np.array(points)
        
        # Update or Create Line
        if name in skeleton_lines:
            skeleton_lines[name].set_data(pts[:,0], pts[:,1])
            skeleton_lines[name].set_3d_properties(pts[:,2])
            
            # Update Tip Marker
            tip_markers[name].set_data([pts[-1,0]], [pts[-1,1]])
            tip_markers[name].set_3d_properties([pts[-1,2]])
        else:
            line, = ax.plot(pts[:,0], pts[:,1], pts[:,2], 'o-', color='black', linewidth=2, markersize=4)
            tip, = ax.plot([pts[-1,0]], [pts[-1,1]], [pts[-1,2]], 'o', color=colors[name], markersize=8)
            skeleton_lines[name] = line
            tip_markers[name] = tip
            
    fig.canvas.draw_idle()

# =====================================================================
#  LAYOUT — Right sidebar for buttons, bottom panel for sliders
# =====================================================================
finger_names = list(finger_tips.keys())

# ── Right Sidebar ────────────────────────────────────────────────────
sidebar_x = 0.73
sidebar_w = 0.24

# Section: Sweep Toggles
fig.text(sidebar_x + 0.01, 0.93, 'Sweep Visibility', fontsize=11, fontweight='bold',
         transform=fig.transFigure, color='#333333')

toggle_buttons = {}
if SHOW_MESH:
    btn_h = 0.032
    btn_gap = 0.005
    btn_y = 0.89
    for idx, name in enumerate(finger_names):
        y = btn_y - idx * (btn_h + btn_gap)
        ax_btn = fig.add_axes([sidebar_x, y, sidebar_w * 0.92, btn_h])
        ax_btn.set_facecolor('#fafafa')
        btn = Button(ax_btn, f'  {name}', color=colors[name], hovercolor=colors[name])
        btn.label.set_fontsize(9)
        btn.label.set_fontweight('bold')
        btn.label.set_color('white')

        def make_toggle(finger_name, button):
            def toggle(event):
                toggle_mesh(finger_name)
                # Visual feedback: dim the button when hidden
                if mesh_visible[finger_name]:
                    button.color = colors[finger_name]
                    button.hovercolor = colors[finger_name]
                else:
                    button.color = '#cccccc'
                    button.hovercolor = '#cccccc'
                button.ax.set_facecolor(button.color)
                fig.canvas.draw_idle()
            return toggle

        btn.on_clicked(make_toggle(name, btn))
        toggle_buttons[name] = btn

# Divider line
fig.text(sidebar_x + 0.01, 0.68, '─' * 30, transform=fig.transFigure, color='#cccccc', fontsize=8)

# Section: View Presets
fig.text(sidebar_x + 0.01, 0.65, 'Camera Presets', fontsize=11, fontweight='bold',
         transform=fig.transFigure, color='#333333')

view_presets = [
    ('Front',   (20,   0)),
    ('Side',    (20,  90)),
    ('Top',     (90,   0)),
    ('Iso',     (20,  45)),
    ('Back',    (20, 180)),
    ('Palm Up', ( 0,   0)),
]

view_buttons = {}
vbtn_h = 0.028
vbtn_gap = 0.004
# 2-column grid for view buttons
cols = 2
vbtn_w = (sidebar_w * 0.92 - 0.01) / cols
vbtn_y_start = 0.61

for idx, (label, (elev, azim)) in enumerate(view_presets):
    col = idx % cols
    row = idx // cols
    x = sidebar_x + col * (vbtn_w + 0.01)
    y = vbtn_y_start - row * (vbtn_h + vbtn_gap)
    ax_v = fig.add_axes([x, y, vbtn_w, vbtn_h])
    btn = Button(ax_v, label, color='#e8e8e8', hovercolor='#d0d0d0')
    btn.label.set_fontsize(8)

    def make_view_setter(e, a):
        def set_view(event):
            ax.view_init(elev=e, azim=a)
            fig.canvas.draw_idle()
        return set_view

    btn.on_clicked(make_view_setter(elev, azim))
    view_buttons[label] = btn

# ── Bottom Panel — Sliders (pyramid / staggered grid layout) ─────────
# Background
slider_bg = fig.add_axes([0.01, 0.01, 0.97, 0.33])
slider_bg.set_facecolor('#fafafa')
slider_bg.set_xticks([])
slider_bg.set_yticks([])
for spine in slider_bg.spines.values():
    spine.set_edgecolor('#dddddd')

fig.text(0.02, 0.335, 'Joint Controls', fontsize=12, fontweight='bold',
         transform=fig.transFigure, color='#333333')

sliders = []
ax_sliders = []
# URDF joint order: joint1=MCP, joint2=Abd, joint3=PIP, joint4=DIP
# Thumb uses different anatomical names: CMC, Abd, MCP, IP
joint_role_labels = {
    'default': ['MCP', 'Abd', 'PIP', 'DIP'],
    'Thumb':   ['CMC', 'Abd', 'MCP', 'IP'],
}
# Display order (top to bottom in each card)
display_order = {
    'default': ['DIP', 'PIP', 'Abd', 'MCP'],
    'Thumb':   ['IP', 'MCP', 'Abd', 'CMC'],
}

# --- Pyramid / staggered grid layout ---
# Row 0:  [  Index   Middle   Ring  ]      (3 fingers, centered)
# Row 1:  [    Thumb       Pinky    ]      (2 fingers, centered)
grid_rows = [
    ['Index', 'Middle', 'Ring'],
    ['Thumb', 'Pinky'],
]

card_w = 0.28             # width of one finger card
card_h = 0.14             # height of one finger card
card_pad_x = 0.03         # horizontal gap between cards
card_pad_y = 0.02         # vertical gap between grid rows
panel_top = 0.31          # top of the card area
slider_w = card_w - 0.04  # slider width inside card
slider_h = 0.016          # slider height
slider_row_gap = 0.024    # vertical gap between slider rows inside a card

def degree_format(val):
    """Format a radian value as degrees with ° symbol."""
    return f'{np.degrees(val):+.1f}°'

for grid_row_idx, grid_row in enumerate(grid_rows):
    n_in_row = len(grid_row)
    # Center the row
    total_row_w = n_in_row * card_w + (n_in_row - 1) * card_pad_x
    row_start_x = (1.0 - total_row_w) / 2.0
    row_top_y = panel_top - grid_row_idx * (card_h + card_pad_y)

    for col_idx, fname in enumerate(grid_row):
        chain = chains[fname]
        cx = row_start_x + col_idx * (card_w + card_pad_x)

        # Card background
        card_bg = fig.add_axes([cx, row_top_y - card_h, card_w, card_h])
        card_bg.set_facecolor('white')
        card_bg.set_xticks([])
        card_bg.set_yticks([])
        for sp in card_bg.spines.values():
            sp.set_edgecolor('#ddd')
            sp.set_linewidth(1.2)

        # Finger title inside card
        fig.text(cx + card_w / 2, row_top_y - 0.005, f'● {fname}',
                 transform=fig.transFigure, fontsize=10, fontweight='bold',
                 color=colors[fname], ha='center', va='top')

        # Collect revolute joints and assign correct role labels
        labels = joint_role_labels.get(fname, joint_role_labels['default'])
        order = display_order.get(fname, display_order['default'])
        rev_joints = []
        rev_idx = 0
        for j in chain:
            if j['type'] != 'revolute':
                continue
            role = labels[rev_idx] if rev_idx < len(labels) else f'J{rev_idx+1}'
            rev_joints.append((role, j))
            rev_idx += 1

        # Sort joints into the requested display order
        ordered_joints = []
        for role_name in order:
            for role, j in rev_joints:
                if role == role_name:
                    ordered_joints.append((role, j))
                    break

        # Sliders inside the card
        slider_start_y = row_top_y - 0.025  # below title
        slider_left = cx + 0.02

        for disp_row, (label, j) in enumerate(ordered_joints):
            sy = slider_start_y - disp_row * slider_row_gap
            ax_s = fig.add_axes([slider_left, sy, slider_w, slider_h])
            ax_s.set_facecolor('#f5f5f5')
            ax_sliders.append(ax_s)

            s = Slider(ax_s, label, j['lower'], j['upper'],
                       valinit=0.0, valstep=0.01,
                       color=colors[fname], initcolor='none')
            s.label.set_fontsize(8)
            s.valtext.set_fontsize(7)
            # Show initial value in degrees
            s.valtext.set_text(degree_format(s.val))
            s.joint_ref = j
            sliders.append(s)

            def make_update(joint_dict, slider_obj):
                def update(val):
                    joint_dict['current_val'] = val
                    # Update displayed value to degrees
                    slider_obj.valtext.set_text(degree_format(val))
                    update_skeleton()
                return update

            s.on_changed(make_update(j, s))


# ── Reset Button ─────────────────────────────────────────────────────
def reset_all(event):
    """Reset all joint angles to 0 (default pose)."""
    for s in sliders:
        s.set_val(0.0)
    update_skeleton()

ax_reset = fig.add_axes([sidebar_x, 0.50, sidebar_w * 0.92, 0.035])
btn_reset = Button(ax_reset, '↺  Reset All Joints', color='#555555', hovercolor='#333333')
btn_reset.label.set_fontsize(9)
btn_reset.label.set_fontweight('bold')
btn_reset.label.set_color('white')
btn_reset.on_clicked(reset_all)

# ── 3D Axes Settings ─────────────────────────────────────────────────
ax.set_title('Interactive Hand Simulator', fontsize=13, fontweight='bold', pad=10)
ax.set_xlabel('X', fontsize=9, labelpad=2)
ax.set_ylabel('Y', fontsize=9, labelpad=2)
ax.set_zlabel('Z', fontsize=9, labelpad=2)
ax.tick_params(axis='both', labelsize=7)
limit = 0.15
ax.set_xlim(-0.05, limit)
ax.set_ylim(-limit / 2, limit / 2)
ax.set_zlim(-0.05, limit)
ax.set_box_aspect([1, 1, 1])
ax.view_init(elev=20, azim=45)

# Initial Draw
update_skeleton()
plt.show()