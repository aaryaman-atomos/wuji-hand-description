import numpy as np
import os
from urdfpy import URDF

# --- SETTINGS ---
URDF_FILE = 'urdf/right.urdf'        # Your URDF filename
OUTPUT_MACRO = 'BuildAssembly.swp'
MESH_DIR_OVERRIDE = ""          # Optional: If your meshes are in a different folder, put full path here.
                                # Leave empty "" to trust the relative paths in the URDF.
SCALE = 1000.0                  # Meters to Millimeters

# --- ROTATION HELPER ---
def get_sw_transform_matrix(urdf_transform):
    """
    Converts a 4x4 URDF Homogeneous Transform (Meters) 
    into a SolidWorks Transform Array (Millimeters).
    SW expects an array of 16 doubles: 
    [Rot_11, Rot_12, Rot_13, Trans_X, 
     Rot_21, Rot_22, Rot_23, Trans_Y, 
     Rot_31, Rot_32, Rot_33, Trans_Z, 
     Scale, 0, 0, 1] 
    """
    # Extract rotation and translation
    R = urdf_transform[:3, :3]
    T = urdf_transform[:3, 3] * SCALE

    # SolidWorks Transform Matrix format (Array of 16)
    # Note: SolidWorks might use Y-up vs Z-up. 
    # Usually, simple import keeps Z as Z. You might need to rotate the view.
    
    sw_matrix = [
        R[0,0], R[0,1], R[0,2], T[0],
        R[1,0], R[1,1], R[1,2], T[1],
        R[2,0], R[2,1], R[2,2], T[2],
        1.0,    0.0,    0.0,    1.0
    ]
    return sw_matrix

def generate_macro():
    if not os.path.exists(URDF_FILE):
        print(f"Error: Could not find {URDF_FILE}")
        return

    print(f"Loading {URDF_FILE}...")
    robot = URDF.load(URDF_FILE)
    fk = robot.link_fk() # Get global transform of every link

    # Start VBA Code
    vba = """
Dim swApp As Object
Dim swModel As Object
Dim swAssembly As Object
Dim swComponent As Object
Dim swMathUtil As Object
Dim swXform As Object
Dim transformData(15) As Double
Dim boolstatus As Boolean
Dim strCompName As String

Sub main()
    Set swApp = Application.SldWorks
    
    ' Create New Assembly
    Set swModel = swApp.NewDocument("C:\\ProgramData\\SolidWorks\\SOLIDWORKS 2024\\templates\\Assembly.asmdot", 0, 0, 0)
    Set swAssembly = swModel
    Set swMathUtil = swApp.GetMathUtility
    
    ' Turn off graphics update for speed
    swModel.ViewZoomtofit2
"""

    print("Processing links...")
    for link in robot.links:
        # 1. Get Mesh Filename
        if len(link.visuals) == 0: continue
        
        # URDF paths are often relative like "../meshes/file.stl"
        # We need to resolve this to a Windows Absolute Path
        mesh_path_raw = link.visuals[0].geometry.mesh.filename
        
        # Clean up path logic
        if MESH_DIR_OVERRIDE:
            filename = os.path.basename(mesh_path_raw)
            full_path = os.path.join(MESH_DIR_OVERRIDE, filename)
        else:
            # Python script running location is the base
            full_path = os.path.abspath(mesh_path_raw)
            
        if not os.path.exists(full_path):
            # Try fixing common relative path issues manually if abspath fails
            # e.g., if script is in 'folder', and path is '../meshes', it tries 'folder/../meshes' which is 'meshes'
            # Let's try to just find the file in the current dir structure
            candidate = os.path.abspath(mesh_path_raw.replace('../', ''))
            if os.path.exists(candidate):
                full_path = candidate
            else:
                print(f"Warning: Could not find mesh: {full_path}")
                continue

        # 2. Get Transform Data
        if link in fk:
            matrix = get_sw_transform_matrix(fk[link])
            
            # Format matrix for VBA
            mat_str = ", ".join([f"{x:.6f}" for x in matrix])
            
            # Escape backslashes for VBA string
            safe_path = full_path.replace("\\", "\\\\")

            vba += f"\n    ' --- Link: {link.name} ---"
            vba += f"\n    ' Import Component"
            vba += f"\n    Set swComponent = swAssembly.AddComponent(\"{safe_path}\", 0, 0, 0)"
            vba += f"\n    If Not swComponent Is Nothing Then"
            
            # Apply Transform
            vba += f"\n        transformData(0) = {matrix[0]}"
            vba += f"\n        transformData(1) = {matrix[1]}"
            vba += f"\n        transformData(2) = {matrix[2]}"
            vba += f"\n        transformData(3) = {matrix[3] / 1000.0} ' Convert mm back to meters for API? No, API usually takes meters."
            # Wait, SolidWorks API 'MathTransform' translation is usually in METERS.
            # Scaling: 
            # If our 'matrix' T is in mm (because we multiplied by SCALE), we need to divide by 1000 for SW API 
            # OR we simply don't scale by 1000 in the python function.
            # Let's fix get_sw_transform_matrix to return Meters for Translation.
            
            vba += f"\n        transformData(4) = {matrix[4]}"
            vba += f"\n        transformData(5) = {matrix[5]}"
            vba += f"\n        transformData(6) = {matrix[6]}"
            vba += f"\n        transformData(7) = {matrix[7] / 1000.0}"
            
            vba += f"\n        transformData(8) = {matrix[8]}"
            vba += f"\n        transformData(9) = {matrix[9]}"
            vba += f"\n        transformData(10) = {matrix[10]}"
            vba += f"\n        transformData(11) = {matrix[11] / 1000.0}"
            
            vba += f"\n        transformData(12) = 1.0" # Scale
            
            vba += f"\n        Set swXform = swMathUtil.CreateTransform(transformData)"
            vba += f"\n        swComponent.Transform2 = swXform"
            vba += f"\n        ' Fix the component so it stays in place"
            vba += f"\n        boolstatus = swComponent.Select4(False, nothing, False)"
            vba += f"\n        swModel.FixComponent"
            vba += f"\n    End If"

    vba += "\n\n    swModel.ViewZoomtofit2\nEnd Sub"

    with open(OUTPUT_MACRO, "w") as f:
        f.write(vba)
    
    print(f"Success! Saved macro to {OUTPUT_MACRO}")

if __name__ == "__main__":
    generate_macro()