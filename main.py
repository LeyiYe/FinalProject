from core.physics.panda_physics import PandaPhysics
from core.utils.soft_object import create_soft_cube
from pysph.tools import pysph_viewer

def main():
    # 1. Initialize objects
    cube = create_soft_cube(size=0.1, resolution=15, density=800)
    physics = PandaPhysics("core/robot/hand.urdf", cube)

    # 2. Run simulation
    for step in range(100):
        physics.update_boundaries({
            "panda_finger_joint1": -0.01 * step,
            "panda_finger_joint2": 0.01 * step
        })
        physics.step()

    # 3. Visualize
    pysph_viewer.show(physics.solver.particles)

if __name__ == "__main__":
    main()