import pybullet as p
import pybullet_data
import numpy as np
import h5py
import time

# Initialize PyBullet
physicsClient = p.connect(p.GUI)  # Use p.DIRECT for headless mode
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, -9.8,0)
p.setRealTimeSimulation(0)

# Load Panda hand (replace with your URDF path)
panda_path = "franka_description/robots/common/hand.urdf"  # Ensure this file exists in your folder
panda_id = p.loadURDF(panda_path, useFixedBase=True)

# Load rectangle.obj as a collision shape
obj_path = "object/rectangle/rectangle.obj"
obj_visual = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=obj_path)
obj_collision = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=obj_path)
obj_id = p.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=obj_collision, baseVisualShapeIndex=obj_visual)

# Function to sample grasp poses
def sample_grasp_poses(obj_id, num_poses=10):
    aabb = p.getAABB(obj_id)
    grasp_poses = []
    for _ in range(num_poses):
        # Sample position on top of the object
        px = np.random.uniform(aabb[0][0], aabb[1][0])
        py = np.random.uniform(aabb[0][1], aabb[1][1])
        pz = aabb[1][2] + 0.05  # Slightly above
        
        # Top-down grasp (quaternion: [w, x, y, z])
        quat = p.getQuaternionFromEuler([np.pi, 0, 0])  # Rotate 180° around X-axis
        grasp_poses.append([px, py, pz, quat[3], quat[0], quat[1], quat[2]])
    return np.array(grasp_poses)

# Generate and validate grasps
grasp_poses = sample_grasp_poses(obj_id, num_poses=10)
valid_poses = []
for pose in grasp_poses:
    px, py, pz, w, x, y, z = pose
    p.resetBasePositionAndOrientation(panda_id, [px, py, pz], [x, y, z, w])
    # Simulate for 50 steps to check stability
    for _ in range(50):
        p.stepSimulation()
        time.sleep(0.01)  # Slow down for visualization
    # Check if object is still near the hand
    obj_pos, _ = p.getBasePositionAndOrientation(obj_id)
    distance = np.linalg.norm(np.array(obj_pos) - np.array([px, py, pz]))
    if distance < 0.1:
        valid_poses.append(pose)

# Save to .h5 file
with h5py.File("rectangle_grasps.h5", "w") as f:
    f.create_dataset("poses", data=np.array(valid_poses))

print(f"Saved {len(valid_poses)} valid grasps to rectangle_grasps.h5")
p.disconnect()