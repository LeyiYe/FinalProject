from core.physics.panda_physics import PandaPhysics
from core.utils.soft_object import create_soft_cube
from pysph.tools.interactive import write_output

def main():
    # 1. Initialize objects
    cube = create_soft_cube(size=0.1, resolution=15, density=800)
    physics = PandaPhysics("core/robot/hand.urdf", cube)

    output_dir = "output/panda_grasp"
    os.makedirs(output_dir, exist_ok=True)

    # 2. Run simulation
    for step in range(100):
        physics.update_boundaries({
            "panda_finger_joint1": -0.01 * step,
            "panda_finger_joint2": 0.01 * step
        })
        physics.step()

        # 3. Write output every 5 steps
        if step % 5 == 0:
            write_output(physics.solver.particles, output_dir, step)

if __name__ == "__main__":
    main()