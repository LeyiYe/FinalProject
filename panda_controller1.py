import pybullet as p
import pybullet_data
import numpy as np
import time
import math
from deformable_object import DeformableObjectSim
from enum import Enum, auto
from scipy.spatial import KDTree 
from panda_fsm1 import PandaFSM

class PandaController:
    def __init__(self, mode="pickup"):
        # Physics client setup
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Load plane
        self.plane_id = p.loadURDF("plane.urdf")

        # Create platform for the object
        self._create_platform()

        # SPH Integration Additionsbv
        self.sph_app = DeformableObjectSim()
        self.sph_solver = self.sph_app.create_solver()
        self.sph_particles = self.sph_app.create_particles()
        
        # Position object on platform
       # self._position_object_on_platform()

        # Load full Panda arm
        self.panda = self._load_panda_arm()
        
        # Get joint information
        self.movable_joints = self._validate_joints()
        print(f"Controlling joints: {self.movable_joints}")
        self.num_joints = p.getNumJoints(self.panda)
        self.joint_info = self._get_joint_info()

        # Initialize arm position above object with open gripper
        self._initialize_arm_position()
        
        # Configuration parameters
        self.config = {
            'franka': {
                'num_joints': self.num_joints,
                'gripper_tip_z_offset': 0.1,
                'gripper_tip_y_offset': 0.05,
                'joint_damping': 0.1
            },
            'force_control': {
                'Kp': 0.01,
                'min_torque': -0.5
            },
        }

        self.force_feedback = []
        self.lift_height = 0.3  # Target lift height in meters
        self.lift_complete = False

        # Coupling parameters
        self.coupling_stiffness = 1e8  # N/m (tune based on material)
        self.gripper_influence_radius = 0.02  # meters
        self.last_gripper_pos = np.zeros(3)
        self.particle_radius = 0.005  # Visual radius of particles
        
        self._create_sph_visualization()
        self._update_sph_kdtree()

        # Initialize variables
        self._init_variables()
        self.fsm = PandaFSM(self)

        # Setup arm control
        self._setup_hand_control()

        # Camera setup to view the scene
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0.5]
        )

    def _load_panda_arm(self):
        """Load the complete Franka Panda arm"""
        flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        panda_orn = p.getQuaternionFromEuler([0, 0, 0])
        
        # Position the arm base next to the platform
        panda_pos = [-0.5, 0, 0]
        panda = p.loadURDF("franka_panda/panda.urdf", panda_pos, panda_orn, 
                          useFixedBase=True, flags=flags)
        
        # Set joint damping for better stability
        for i in range(p.getNumJoints(panda)):
            p.changeDynamics(panda, i, linearDamping=0.04, angularDamping=0.04)
        
        return panda

    def _create_platform(self):
        """Create a platform for the object to rest on"""
        platform_height = 0.02  # 2cm thick
        platform_size = 0.5      # 50cm square
        
        # Visual shape
        platform_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[platform_size/2, platform_size/2, platform_height/2],
            rgbaColor=[0.8, 0.8, 0.8, 1]  # Light gray
        )
        
        # Collision shape
        collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[platform_size/2, platform_size/2, platform_height/2]
        )
        
        self.platform = p.createMultiBody(
            baseMass=0,  # Static platform
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=platform_shape,
            basePosition=[0, 0, platform_height/2]  # Center of the scene
        )

    def _initialize_arm_position(self):
        """Position the arm with open gripper above the object"""
        # Open gripper immediately
        self._open_gripper()
        
        # Calculate object center
        obj_center = [
            np.mean(self.sph_particles.x),
            np.mean(self.sph_particles.y),
            np.max(self.sph_particles.z)
        ]
        
        # Target position 5cm above object center
        target_pos = [obj_center[0], obj_center[1], obj_center[2] + 0.05]
        target_orn = p.getQuaternionFromEuler([0, -math.pi, 0])  # Standard gripper orientation
        
        # Get end effector link index (typically the last link)
        self.end_effector_index = p.getNumJoints(self.panda) - 1
        
        # Calculate IK solution
        joint_positions = p.calculateInverseKinematics(
            self.panda,
            self.end_effector_index,
            target_pos,
            targetOrientation=target_orn,
            lowerLimits=[-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
            upperLimits=[2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
            jointRanges=[5.8, 3.5, 5.8, 3.1, 5.8, 3.7, 5.8],
            restPoses=[0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32],
            maxNumIterations=100,
            residualThreshold=1e-5
        )
        
        # Set arm joints to IK solution
        for i in range(7):  # First 7 joints are the arm
            p.resetJointState(self.panda, i, joint_positions[i])
            p.setJointMotorControl2(
                self.panda,
                i,
                p.POSITION_CONTROL,
                targetPosition=joint_positions[i],
                force=500,
                positionGain=0.3
            )
        
        print("Arm initialized above object")

    def _open_gripper(self):
        """Set gripper fingers to open position"""
        # Set target positions for both fingers
        open_position = 0.04  # Open position
        
        for joint_name in ['panda_finger_joint1', 'panda_finger_joint2']:
            if joint_name in self.joint_info:
                joint_idx = self.joint_info[joint_name]['index']
                p.resetJointState(self.panda, joint_idx, targetValue=open_position)
                p.setJointMotorControl2(
                    self.panda,
                    joint_idx,
                    p.POSITION_CONTROL,
                    targetPosition=open_position,
                    force=20,
                    positionGain=0.1
                )
        print("Gripper set to open position")

    def _close_gripper(self):
        """Close gripper until contact is detected"""
        # Get current finger positions
        finger_pos = self.get_gripper_positions()
        
        # Gradually close fingers
        target_pos = [0.0, 0.0]  # Fully closed position
        contact_force_threshold = 5.0  # Force threshold to detect contact
        
        # Check if we've made contact with the object
        contact_force = self.get_gripper_force()
        if np.any(contact_force > contact_force_threshold):
            print("Contact detected with force:", contact_force)
            return True  # Contact made
        
        # If no contact, continue closing
        for i, joint_name in enumerate(['panda_finger_joint1', 'panda_finger_joint2']):
            if joint_name in self.joint_info:
                joint_idx = self.joint_info[joint_name]['index']
                p.setJointMotorControl2(
                    self.panda,
                    joint_idx,
                    p.POSITION_CONTROL,
                    targetPosition=target_pos[i],
                    force=10,
                    positionGain=0.1
                )
        return False  # No contact yet

    def _lift_object(self):
        """Lift the object to target height"""
        # Get current end effector position
        current_pos = p.getLinkState(self.panda, self.end_effector_index)[0]
        
        # Target position is same XY but higher Z
        target_pos = [current_pos[0], current_pos[1], self.lift_height]
        target_orn = p.getQuaternionFromEuler([0, -math.pi, 0])
        
        # Calculate new joint positions
        joint_positions = p.calculateInverseKinematics(
            self.panda,
            self.end_effector_index,
            target_pos,
            targetOrientation=target_orn,
            lowerLimits=[-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
            upperLimits=[2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
            maxNumIterations=100,
            residualThreshold=1e-5
        )
        
        # Move arm to new position
        for i in range(7):  # First 7 joints are the arm
            p.setJointMotorControl2(
                self.panda,
                i,
                p.POSITION_CONTROL,
                targetPosition=joint_positions[i],
                force=500,
                positionGain=0.3
            )
        
        # Check if we've reached the target height
        new_pos = p.getLinkState(self.panda, self.end_effector_index)[0]
        if abs(new_pos[2] - self.lift_height) < 0.01:
            self.lift_complete = True
            print("Object lifted to target height")
            return True
        return False

    # [Keep all your existing methods like _position_object_on_platform, 
    #  _get_joint_info, get_gripper_force, etc. unchanged]

    def run(self):
        """Main simulation loop"""
        while True:
            # SPH coupling
            gripper_pos = self.get_gripper_center()
            self._apply_gripper_to_sph(gripper_pos)
            reaction_force = self._compute_sph_reaction_force(gripper_pos)
            
            p.applyExternalForce(
                self.panda,
                -1,
                forceObj=reaction_force,
                posObj=gripper_pos,
                flags=p.WORLD_FRAME
            )
            
            # Update visualization
            self._update_sph_visualization()
            
            # State machine update
            if not self.fsm.update():
                break
                    
            # Step simulation
            p.stepSimulation()
            time.sleep(1./240.)

if __name__ == "__main__":
    controller = PandaController(mode="pickup")
    try:
        controller.run()
    finally:
        controller.visualize_force_feedback()
        p.disconnect()