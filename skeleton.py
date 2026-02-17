import numpy as np
from urdfpy import URDF

# Load your robot
robot = URDF.load('urdf/right.urdf')

print(f"{'JOINT NAME':<25} | {'GLOBAL XYZ (m)':<25} | {'AXIS'}")
print("-" * 65)

# Get absolute transforms for all links (Forward Kinematics at rest)
fk = robot.link_fk()

for joint in robot.joints:
    # Find the child link (the link this joint moves)
    child_link_name = joint.child
    # Find the link object
    child_link = next((l for l in robot.links if l.name == child_link_name), None)
    
    if child_link and child_link in fk:
        # The joint is located at the origin of its child link
        transform = fk[child_link]
        pos = transform[:3, 3] # Extract XYZ position
        
        # Calculate the Global Axis vector
        # (Rotate the local axis by the link's orientation)
        rot_matrix = transform[:3, :3]
        global_axis = np.dot(rot_matrix, joint.axis)
        
        print(f"{joint.name:<25} | {pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}  | {global_axis}")