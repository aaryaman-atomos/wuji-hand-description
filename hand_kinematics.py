import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- SETTINGS ---
URDF_FILE = 'urdf/right.urdf'
MESH_RESOLUTION = 25  # Higher = Smoother mesh, Lower = More "grid-like" look

# Toggle which fingers to show
SHOW_FINGERS = {
    'Thumb':  True,
    'Index':  True,
    'Middle': True,
    'Ring':   True,
    'Pinky':  True
}

# --- KINEMATICS HELPERS ---
def get_transform(xyz, rpy):
    """Returns 4x4 Homogeneous Transform from fixed XYZ/RPY."""
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
    """Returns 4x4 Transform for a joint rotation."""
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

# --- LOAD ROBOT ---
tree = ET.parse(URDF_FILE)
root = tree.getroot()

joints = {}
for joint in root.findall('joint'):
    name = joint.get('name')
    parent = joint.find('parent').get('link')
    child = joint.find('child').get('link')
    
    origin = joint.find('origin')
    xyz = [float(x) for x in origin.get('xyz', '0 0 0').split()]
    rpy = [float(x) for x in origin.get('rpy', '0 0 0').split()]
    
    axis_elem = joint.find('axis')
    axis = [float(x) for x in axis_elem.get('xyz', '1 0 0').split()] if axis_elem is not None else [1,0,0]
    
    limit = joint.find('limit')
    lower = float(limit.get('lower', 0)) if limit is not None else 0
    upper = float(limit.get('upper', 0)) if limit is not None else 0
        
    joints[child] = {
        'name': name, 'parent': parent, 'xyz': xyz, 'rpy': rpy, 
        'axis': axis, 'lower': lower, 'upper': upper
    }

finger_tips = {
    'Thumb':  'right_finger1_tip_link',
    'Index':  'right_finger2_tip_link',
    'Middle': 'right_finger3_tip_link',
    'Ring':   'right_finger4_tip_link',
    'Pinky':  'right_finger5_tip_link'
}

# --- GENERATE MESH ---
def generate_finger_mesh(tip_link_name):
    # 1. Build Chain (Tip -> Base)
    chain = []
    curr = tip_link_name
    while curr in joints:
        chain.append(joints[curr])
        curr = joints[curr]['parent']
    chain.reverse() # Base -> Tip
    
    # 2. Setup Parametric Grid
    # We map 'u' to Abduction (Side-to-Side) and 'v' to Flexion (Curling)
    
    # Joint 1: Abduction (The base pivot)
    j1 = chain[0]
    u_range = np.linspace(j1['lower'], j1['upper'], MESH_RESOLUTION)
    
    # Joint 2: Proximal Flexion (The main knuckle)
    j2 = chain[1]
    v_range = np.linspace(j2['lower'], j2['upper'], MESH_RESOLUTION)
    
    U, V = np.meshgrid(u_range, v_range)
    X, Y, Z = np.zeros_like(U), np.zeros_like(U), np.zeros_like(U)
    
    # Pre-calculate static offsets
    static_T = [get_transform(j['xyz'], j['rpy']) for j in chain]
    
    # 3. Calculate Surface
    for r in range(MESH_RESOLUTION):
        for c in range(MESH_RESOLUTION):
            theta_abd = U[r, c]   # Joint 1 Angle
            theta_flex = V[r, c]  # Joint 2 Angle
            
            # --- COUPLING LOGIC ---
            # To create a "shell", we usually extend the other joints fully or 
            # couple them. Here we couple J3 and J4 to J2 to show natural curling.
            # Change '0.8' to '0' if you want to see the stiff-finger sweep.
            theta_mid = theta_flex * 0.8  # J3 bends 80% as much as J2
            theta_dist = theta_flex * 0.6 # J4 bends 60% as much as J2
            
            # Forward Kinematics
            T = np.eye(4)
            
            # Link 1 (Abduction)
            T = T @ static_T[0] @ get_revolute_transform(chain[0]['axis'], theta_abd)
            
            # Link 2 (Proximal Flexion)
            T = T @ static_T[1] @ get_revolute_transform(chain[1]['axis'], theta_flex)
            
            # Link 3 (Medial Flexion)
            T = T @ static_T[2] @ get_revolute_transform(chain[2]['axis'], theta_mid)
            
            # Link 4 (Distal Flexion - if exists)
            if len(chain) > 3:
                 T = T @ static_T[3] @ get_revolute_transform(chain[3]['axis'], theta_dist)

            # Tip Fixed (if exists)
            if len(chain) > 4:
                 T = T @ static_T[4]
            
            pos = T[:3, 3]
            X[r,c] = pos[0]
            Y[r,c] = pos[1]
            Z[r,c] = pos[2]
            
    return X, Y, Z

# --- PLOTTING ---
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

colors = {'Thumb': 'red', 'Index': 'lime', 'Middle': 'blue', 'Ring': 'orange', 'Pinky': 'magenta'}

print("Generating kinematic meshes...")
for name, enabled in SHOW_FINGERS.items():
    if not enabled: continue
    
    tip = finger_tips[name]
    X, Y, Z = generate_finger_mesh(tip)
    
    # Plot as a Surface with "Mesh" lines (edgecolor='k')
    ax.plot_surface(X, Y, Z, color=colors[name], alpha=0.4, 
                    edgecolor='black', linewidth=0.5, rstride=1, cstride=1)

# Formatting
ax.set_title('Robot Hand Kinematic Mesh (Reachable Shell)')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')

# Force Aspect Ratio
limit = 0.15
ax.set_xlim(-0.05, limit)
ax.set_ylim(-limit/2, limit/2)
ax.set_zlim(-0.05, limit)
ax.set_box_aspect([1,1,1])

plt.show()