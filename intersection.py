import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay

# --- SETTINGS ---
URDF_FILE = 'urdf/right.urdf'
SAMPLES = 2000          # Points per finger
VOXEL_SIZE = 0.005      # 5mm grid for intersection calculation

# --- 1. ROBOT HAND KINEMATICS (FROM URDF) ---
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
    u = np.array(axis) / np.linalg.norm(axis)
    x, y, z = u
    c, s = np.cos(angle), np.sin(angle)
    C = 1 - c
    R = np.array([
        [x*x*C + c,    x*y*C - z*s,  x*z*C + y*s],
        [y*x*C + z*s,  y*y*C + c,    y*z*C - x*s],
        [z*x*C - y*s,  z*y*C + x*s,  z*z*C + c]
    ])
    T = np.eye(4)
    T[:3, :3] = R
    return T

def load_robot_model(urdf_file):
    tree = ET.parse(urdf_file)
    root = tree.getroot()
    joints = {}
    for joint in root.findall('joint'):
        name = joint.get('name')
        parent = joint.find('parent').get('link')
        child = joint.find('child').get('link')
        jtype = joint.get('type')
        origin = joint.find('origin')
        xyz = [float(x) for x in origin.get('xyz', '0 0 0').split()]
        rpy = [float(x) for x in origin.get('rpy', '0 0 0').split()]
        axis = [float(x) for x in joint.find('axis').get('xyz', '1 0 0').split()] if joint.find('axis') is not None else [1,0,0]
        limit = joint.find('limit')
        lower = float(limit.get('lower', 0)) if limit is not None else 0
        upper = float(limit.get('upper', 0)) if limit is not None else 0
        joints[child] = {'parent': parent, 'xyz': xyz, 'rpy': rpy, 'axis': axis, 'lower': lower, 'upper': upper, 'type': jtype}
    return joints

def sample_robot_workspace(joints, tips, n_samples):
    points = []
    chains = {}
    for name, tip in tips.items():
        chain = []
        curr = tip
        while curr in joints:
            chain.append(joints[curr])
            curr = joints[curr]['parent']
        chains[name] = chain[::-1] # Base to Tip

    for name, chain in chains.items():
        static_T = [get_transform(j['xyz'], j['rpy']) for j in chain]
        for _ in range(n_samples):
            T = np.eye(4)
            for i, j in enumerate(chain):
                T = T @ static_T[i]
                if j['type'] == 'revolute':
                    angle = np.random.uniform(j['lower'], j['upper'])
                    T = T @ get_revolute_transform(j['axis'], angle)
            points.append(T[:3, 3])
    return np.array(points)

# --- 2. HUMAN HAND MODEL (STANDARD ANTHROPOMETRY) ---
# Lengths in meters (Approx. 50th percentile male)
# L1=Meta, L2=Prox, L3=Med, L4=Dist
HUMAN_PARAMS = {
    'Thumb':  {'L': [0.046, 0.032, 0.022], 'lims': [(-20, 50), (-10, 55), (-15, 80)]}, # CMC(2DOF), MCP, IP
    'Index':  {'L': [0.068, 0.040, 0.025, 0.018], 'lims': [(-30, 90), (0, 100), (0, 80)]}, # MCP(2DOF), PIP, DIP
    'Middle': {'L': [0.065, 0.045, 0.028, 0.019], 'lims': [(-30, 90), (0, 100), (0, 80)]},
    'Ring':   {'L': [0.060, 0.040, 0.026, 0.018], 'lims': [(-30, 90), (0, 100), (0, 80)]},
    'Pinky':  {'L': [0.055, 0.033, 0.018, 0.016], 'lims': [(-30, 90), (0, 100), (0, 80)]}
}
# Base positions of human fingers relative to wrist (approx)
HUMAN_BASES = {
    'Thumb':  [0.020, -0.010, 0.000],
    'Index':  [0.025, 0.030, 0.000],
    'Middle': [0.005, 0.032, 0.000],
    'Ring':   [-0.015, 0.028, 0.000],
    'Pinky':  [-0.030, 0.020, 0.000]
}

def sample_human_workspace(n_samples):
    points = []
    for name, params in HUMAN_PARAMS.items():
        L = params['L']
        deg2rad = np.pi / 180.0
        base = HUMAN_BASES[name]
        
        for _ in range(n_samples):
            T = np.eye(4)
            T[:3, 3] = base
            
            # Simple kinematics for human fingers
            if name == 'Thumb':
                # CMC (2 DOF: Flex/Ext + Abd/Add)
                flex = np.random.uniform(*params['lims'][0]) * deg2rad
                abd  = np.random.uniform(0, 60) * deg2rad
                T = T @ get_revolute_transform([0,0,1], abd) @ get_revolute_transform([1,0,0], flex)
                T = T @ get_transform([L[0], 0, 0], [0,0,0]) # Metacarpal
                
                # MCP
                flex = np.random.uniform(*params['lims'][1]) * deg2rad
                T = T @ get_revolute_transform([1,0,0], flex)
                T = T @ get_transform([L[1], 0, 0], [0,0,0]) # Proximal
                
                # IP
                flex = np.random.uniform(*params['lims'][2]) * deg2rad
                T = T @ get_revolute_transform([1,0,0], flex)
                T = T @ get_transform([L[2], 0, 0], [0,0,0]) # Distal
            
            else: # Fingers
                # MCP (2 DOF)
                flex = np.random.uniform(*params['lims'][0]) * deg2rad
                abd  = np.random.uniform(-15, 15) * deg2rad
                T = T @ get_revolute_transform([0,0,1], abd) @ get_revolute_transform([1,0,0], flex)
                T = T @ get_transform([L[1], 0, 0], [0,0,0]) # Proximal
                
                # PIP
                flex = np.random.uniform(*params['lims'][1]) * deg2rad
                T = T @ get_revolute_transform([1,0,0], flex)
                T = T @ get_transform([L[2], 0, 0], [0,0,0]) # Medial

                # DIP
                flex = np.random.uniform(*params['lims'][2]) * deg2rad
                T = T @ get_revolute_transform([1,0,0], flex)
                T = T @ get_transform([L[3], 0, 0], [0,0,0]) # Distal

            points.append(T[:3, 3])
            
    return np.array(points)

# --- 3. INTERSECTION ANALYSIS ---
print("Sampling Robot Workspace...")
robot_joints = load_robot_model(URDF_FILE)
robot_tips = {
    'Thumb': 'right_finger1_tip_link', 'Index': 'right_finger2_tip_link',
    'Middle': 'right_finger3_tip_link', 'Ring': 'right_finger4_tip_link',
    'Pinky': 'right_finger5_tip_link'
}
robot_pts = sample_robot_workspace(robot_joints, robot_tips, SAMPLES)

print("Sampling Human Workspace...")
human_pts = sample_human_workspace(SAMPLES)

# Voxel Grid Intersection
def get_voxel_set(points, size):
    return set([tuple(np.floor(p / size).astype(int)) for p in points])

print("Calculating Intersection...")
robot_voxels = get_voxel_set(robot_pts, VOXEL_SIZE)
human_voxels = get_voxel_set(human_pts, VOXEL_SIZE)

# Intersection: Voxels reachable by BOTH
intersect_voxels = robot_voxels.intersection(human_voxels)
# Missing: Voxels reachable by Human but NOT Robot
missing_voxels = human_voxels - robot_voxels

# Convert back to points for plotting
def voxels_to_points(voxels, size):
    if not voxels: return np.array([])
    return np.array(list(voxels)) * size

p_intersect = voxels_to_points(intersect_voxels, VOXEL_SIZE)
p_missing = voxels_to_points(missing_voxels, VOXEL_SIZE)

# --- PLOTTING ---
fig = plt.figure(figsize=(12, 6))

# Plot 1: Overlap
ax1 = fig.add_subplot(121, projection='3d')
if len(p_intersect) > 0:
    ax1.scatter(p_intersect[:,0], p_intersect[:,1], p_intersect[:,2], c='green', s=1, alpha=0.1, label='Shared Space')
if len(p_missing) > 0:
    ax1.scatter(p_missing[:,0], p_missing[:,1], p_missing[:,2], c='red', s=1, alpha=0.1, label='Human Only')

ax1.set_title("Human vs Robot Reachability")
ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
ax1.legend()
ax1.set_box_aspect([1,1,1])

# Plot 2: Coverage Ratio
ax2 = fig.add_subplot(122)
categories = ['Shared', 'Robot Only', 'Human Only']
counts = [len(intersect_voxels), len(robot_voxels - human_voxels), len(missing_voxels)]
ax2.bar(categories, counts, color=['green', 'blue', 'red'])
ax2.set_title(f"Workspace Overlap (Voxel Count)\nCoverage: {len(intersect_voxels)/len(human_voxels)*100:.1f}% of Human Range")

plt.tight_layout()
plt.show()