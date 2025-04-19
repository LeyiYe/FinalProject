from core.physics.panda_physics import PandaPhysics
from core.utils.soft_object import create_soft_cube
from pysph.solver.utils import dump
import os
import numpy as np

def main():
    # 1. Initialize soft cube with proper properties
    cube = create_soft_cube(
        size=0.1,
        resolution=15,
        density=1200,  # Should match rho0 in PandaPhysics
        h=0.005,       # Smoothing length
        E=1e6,         # Young's modulus (1 MPa)
        nu=0.45        # Poisson's ratio
    )
    
    # 2. Initialize physics
    physics = PandaPhysics(
        urdf_file="franka_description/robots/common/hand.urdf",
        particle_array=cube
    )
    
    # 3. Create output directory
    output_dir = "output/panda_grasp"
    os.makedirs(output_dir, exist_ok=True)

    # 4. Run simulation
    for step in range(100):
        # Update finger positions (0.04m max opening)
        finger_pos = min(0.04, 0.01 * step)
        physics.update_boundaries({
            "panda_finger_joint1": -finger_pos,
            "panda_finger_joint2": finger_pos
        }, hand_origin=np.eye(4))  # Assuming hand at origin
        
        physics.step()
        
        # Monitor contacts
        contacts = physics.get_contacts()
        print(f"Step {step}: {len(contacts)} contact points")
        
        # Save output every 5 steps
        if step % 5 == 0:
            filename = os.path.join(output_dir, f"panda_grasp_{step:05d}.npz")
            # Save all particle arrays including boundaries
            dump(filename, physics.solver.particles, 
                 solver_data={
                     "t": physics.solver.t,
                     "contacts": contacts  # Optional: save contact info
                 })

    # 5. Visualize final state
    physics.visualize()

if __name__ == "__main__":
    main()