import pybullet as p
import pybullet_data
import numpy as np
import time
from deformable_object import DeformableObjectSim
from enum import Enum, auto
from scipy.spatial import KDTree 
from panda_fsm import PandaFSM

class PandaController:
    def __init__(self, mode="pickup"):
        # Physics client setup
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Load plane and Panda hand
        self.plane_id = p.loadURDF("plane.urdf")
        self.panda = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

        # Hide all links except hand and fingers
        self.finger_joint_indices = []
        self.hand_link_index = None
        for i in range(p.getNumJoints(self.panda)):
            joint_info = p.getJointInfo(self.panda, i)
            joint_name = joint_info[1].decode("utf-8")

            # Store hand and finger indices
            if "hand" in joint_name:
                self.hand_link_index = i
            if "finger" in joint_name:
                self.finger_joint_indices.append(i)


            if "hand" not in joint_name and "finger" not in joint_name:
                p.setCollisionFilterGroupMask(self.panda, i, 0, 0)  # Disable collisions
                p.changeVisualShape(self.panda, i, rgbaColor=[0, 0, 0, 0])  # Make invisible
                
        # Create platform for the object
        self._create_platform()

        self._position_hand_above_object()


        # Get joint information
        self.num_joints = p.getNumJoints(self.panda)
        self.joint_info = self._get_joint_info()
        
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
        # SPH Integration Additions
        self.sph_app = DeformableObjectSim()
        self.sph_solver = self.sph_app.create_solver()
        self.sph_particles = self.sph_app.create_particles()
        self.force_feedback= []

        
        # Coupling parameters
        self.coupling_stiffness = 1e8  # N/m (tune based on material)
        self.gripper_influence_radius = 0.02  # meters
        self.last_gripper_pos = np.zeros(3)
        self.particle_radius = 0.005  # Visual radius of particles
        
        # Initialize SPH simulation
        self._position_object_on_platform()
        self._create_sph_visualization()
        self._update_sph_kdtree()

        # Initialize variables
        self._init_variables()
        self.fsm = PandaFSM(self)

        # Setup arm control
        self._setup_hand_control()

        # Camera setup to view the platform
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0.5, -0.5, 0.5]
        )

    def _position_hand_above_object(self):
        """Position the hand directly above the object using IK"""
        obj_center = [
            np.mean(self.sph_particles.x),
            np.mean(self.sph_particles.y),
            np.max(self.sph_particles.z)  # Use max Z since particles may vary in height
        ]
        
        # Target position 5cm above object center
        target_pos = [obj_center[0], obj_center[1], obj_center[2] + 0.05]
        target_orn = p.getQuaternionFromEuler([0, -np.pi, 0])  # Standard gripper orientation
        
        # Calculate IK solution
        joint_positions = p.calculateInverseKinematics(
            self.panda,
            self.hand_link_index,
            target_pos,
            targetOrientation=target_orn,
            lowerLimits=[-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
            upperLimits=[2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
            jointDamping=[0.1]*7,
            maxNumIterations=200,
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

    def _setup_hand_control(self):
        """Initialize hand joint control parameters (for hand-only setup)"""
        # Get all joint information
        num_joints = p.getNumJoints(self.panda)
        print(f"Total joints in loaded model: {num_joints}")
        
        # Find finger joints (they should be the only movable joints in hand-only mode)
        finger_joints = []
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.panda, i)
            joint_name = joint_info[1].decode("utf-8")
            joint_type = joint_info[2]
            
            print(f"Joint {i}: {joint_name} (type: {joint_type})")
            
            # Look for finger joints (typically 'panda_finger_joint1' and 'panda_finger_joint2')
            if "finger" in joint_name.lower() and joint_type != p.JOINT_FIXED:
                finger_joints.append(i)
        
        # Initialize control for finger joints only
        for joint_idx in finger_joints:
            p.resetJointState(self.panda, joint_idx, targetValue=0)
            p.setJointMotorControl2(
                self.panda,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=0,
                force=10,  # Reduced force for fingers
                positionGain=0.1
            )
            p.changeDynamics(self.panda, joint_idx, 
                            linearDamping=0.1, 
                            angularDamping=0.1)
        
        # Store finger joint indices for later use
        self.finger_joint_indices = finger_joints
        print(f"Initialized control for finger joints: {finger_joints}")

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
            basePosition=[0.5, -0.5, 0.5]  # Position of the platform
        )


    def _position_object_on_platform(self):
        """Center SPH object on platform with random distribution"""
        if not hasattr(self, 'sph_particles') or self.sph_particles is None:
            return
            
        platform_center = np.array([0.5, -0.5, 0.51])  # Slightly above platform

        # Calculate particle bounds
        min_x, max_x = np.min(self.sph_particles.x), np.max(self.sph_particles.x)
        min_y, max_y = np.min(self.sph_particles.y), np.max(self.sph_particles.y)
        min_z = np.min(self.sph_particles.z)
        
        # Calculate required offsets
        x_offset = platform_center[0] - (min_x + max_x)/2
        y_offset = platform_center[1] - (min_y + max_y)/2
        z_offset = platform_center[2] - min_z  # Align bottom with platform
        
        # Apply offsets
        self.sph_particles.x += x_offset
        self.sph_particles.y += y_offset
        self.sph_particles.z += z_offset   
        
        print(f"Particle positions initialized around {platform_center}")

        
    def _init_variables(self):
        """Initialize all state variables"""
        # Contact and force tracking
        self.contacts = []
        self.F_history = []
        self.filtered_forces = []
        self.f_moving_average = []
        self.f_errs = np.ones(10)
        
        # Gripper state
        self.left_has_contacted = False
        self.right_has_contacted = False
        self.franka_positions_at_contact = np.zeros(self.config['franka']['num_joints'])
        self.grippers_pre_squeeze = [-1, -1]
        
        # Counters
        self.close_fails = 0
        self.squeeze_counter = 0
        self.squeeze_holding_counter = 0
        self.squeeze_no_gravity_counter = 0
        self.open_counter = 0
        self.hang_counter = 0
        
        # Control outputs
        self.vel_des = np.zeros(self.config['franka']['num_joints'])
        self.pos_des = np.zeros(self.config['franka']['num_joints'])
        self.torque_des = np.zeros(self.config['franka']['num_joints'])
        self.running_torque = [-0.1, -0.1]
        
        # Success flags
        self.pickup_success = False
        
    def _get_joint_info(self):
        joint_info = {}
        for i in range(self.num_joints):
            info = p.getJointInfo(self.panda, i)
            joint_name = info[1].decode("utf-8")
            joint_info[joint_name] = {
                'index': i,
                'type': info[2],
                'limit_lower': info[8],
                'limit_upper': info[9]
            }
        return joint_info
    
    
    def get_gripper_force(self):
        """Estimate gripper force based on contact points"""
        left_force = 0
        right_force = 0
        
        # Get contact points between fingers and objects
        contacts = p.getContactPoints(bodyA=self.panda)
        
        for contact in contacts:
            # Check if contact is with fingers
            if contact[3] in [self.joint_info['panda_finger_joint1']['index'], 
                             self.joint_info['panda_finger_joint2']['index']]:
                normal_force = contact[9]  # Normal force magnitude
                
                if contact[3] == self.joint_info['panda_finger_joint1']['index']:
                    left_force += normal_force
                else:
                    right_force += normal_force
        
        return np.array([left_force, right_force])
    
    def set_gripper_velocity(self, left_vel, right_vel):
        """Set gripper finger velocities"""
        p.setJointMotorControlArray(
            self.panda,
            [self.joint_info['panda_finger_joint1']['index'], 
             self.joint_info['panda_finger_joint2']['index']],
            p.VELOCITY_CONTROL,
            targetVelocities=[left_vel, right_vel]
        )
    
    def set_gripper_torque(self, left_torque, right_torque):
        """Set gripper finger torques"""
        p.setJointMotorControlArray(
            self.panda,
            [self.joint_info['panda_finger_joint1']['index'], 
             self.joint_info['panda_finger_joint2']['index']],
            p.TORQUE_CONTROL,
            forces=[left_torque, right_torque]
        )
    
    def get_gripper_positions(self): 
        """Get current gripper finger positions"""
        states = p.getJointStates(self.panda, [
            self.joint_info['panda_finger_joint1']['index'],
            self.joint_info['panda_finger_joint2']['index']
        ])
        return [states[0][0], states[1][0]]
    
    def get_gripper_center(self):
        """Calculate the center point between the gripper fingers"""
        # Get finger joint positions
        left_finger_pos = p.getLinkState(self.panda, 
                                    self.joint_info['panda_finger_joint1']['index'])[0]
        right_finger_pos = p.getLinkState(self.panda, 
                                        self.joint_info['panda_finger_joint2']['index'])[0]
        
        # Calculate midpoint between fingers
        gripper_center = [
            (left_finger_pos[0] + right_finger_pos[0]) / 2,
            (left_finger_pos[1] + right_finger_pos[1]) / 2,
            (left_finger_pos[2] + right_finger_pos[2]) / 2
        ]
        return gripper_center
    
    def _create_sph_visualization(self):
        """Create efficient particle visualization"""
        if not hasattr(self, 'sph_particles') or self.sph_particles is None:
            raise RuntimeError("SPH particles not initialized before visualization")
        
        # Clear any existing visual bodies
        if hasattr(self, 'sph_visual_bodies'):
            for body in self.sph_visual_bodies:
                p.removeBody(body)
        
        # Create visual shape with bright color
        self.particle_shape = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=self.particle_radius,
            rgbaColor=[1, 0, 0, 1.0]  # Bright red, fully opaque
        )
        
        self.sph_visual_bodies = []
        for i in range(len(self.sph_particles.x)):
            body = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=self.particle_shape,
                basePosition=[
                    self.sph_particles.x[i],
                    self.sph_particles.y[i],
                    self.sph_particles.z[i]
                ]
            )
            self.sph_visual_bodies.append(body)
        
        print(f"Created {len(self.sph_visual_bodies)} visual bodies at positions:")
        print(f"X: {min(self.sph_particles.x):.3f} to {max(self.sph_particles.x):.3f}")
        print(f"Y: {min(self.sph_particles.y):.3f} to {max(self.sph_particles.y):.3f}")
        print(f"Z: {min(self.sph_particles.z):.3f} to {max(self.sph_particles.z):.3f}")
    
    def _update_sph_visualization(self):
        """Update particle positions every simulation step"""
        if not hasattr(self, 'sph_visual_bodies'):
            return
            
        for i, body in enumerate(self.sph_visual_bodies):
            p.resetBasePositionAndOrientation(
                body,
                posObj=[
                    self.sph_particles.x[i],
                    self.sph_particles.y[i],
                    self.sph_particles.z[i]
                ],
                ornObj=[0, 0, 0, 1]
            )

    def _update_sph_kdtree(self):
        """Update KDTree for efficient neighbor searches"""
        particle_positions = np.column_stack([
            self.sph_particles.x,
            self.sph_particles.y,
            self.sph_particles.z
        ])
        self.sph_kdtree = KDTree(particle_positions)

    def _apply_gripper_to_sph(self, gripper_pos):
        """Apply gripper motion to SPH particles"""
        # Get current gripper velocity (finite difference)
        gripper_vel = np.array(gripper_pos) - self.last_gripper_pos
        self.last_gripper_pos = np.array(gripper_pos)
        
        # Find particles near gripper
        neighbor_indices = self.sph_kdtree.query_ball_point(
            gripper_pos, 
            self.gripper_influence_radius
        )
        
        # Apply motion (rigid attachment approximation)
        for i in neighbor_indices:
            self.sph_particles.u[i] = gripper_vel[0] * 240.0  # Scale by timestep
            self.sph_particles.v[i] = gripper_vel[1] * 240.0
            self.sph_particles.w[i] = gripper_vel[2] * 240.0

    def _compute_sph_reaction_force(self, gripper_pos):
        """Calculate reaction force from SPH particles"""
        print(f"Gripper pos: {gripper_pos}")
        print(f"Particle range: x[{min(self.sph_particles.x):.3f}-{max(self.sph_particles.x):.3f}]")
        neighbor_indices = self.sph_kdtree.query_ball_point(
            gripper_pos,
            self.gripper_influence_radius
        )
        
        total_force = np.zeros(3)
    
        for i in neighbor_indices:
            # Compute displacement vector
            r = np.array([
                self.sph_particles.x[i] - gripper_pos[0],
                self.sph_particles.y[i] - gripper_pos[1],
                self.sph_particles.z[i] - gripper_pos[2]
            ])
            dist = np.linalg.norm(r)
            
            if dist < 1e-6:
                continue
                
            # Normalized direction vector
            n = r / dist
            
            # Simplified force model (stress + spring)
            stress_force = np.array([
                self.sph_particles.s00[i] + self.sph_particles.s01[i] + self.sph_particles.s02[i],
                self.sph_particles.s10[i] + self.sph_particles.s11[i] + self.sph_particles.s12[i],
                self.sph_particles.s20[i] + self.sph_particles.s21[i] + self.sph_particles.s22[i]
            ])
            
            spring_force = -self.coupling_stiffness * r
            total_force += (stress_force + spring_force) * self.sph_particles.m[i]
        
        print(f"Computed force: {total_force}")
        return total_force

    def run(self):
        """Main simulation loop with forced visualization updates"""
        while True:
            # SPH coupling
            # gripper_pos = p.getLinkState(self.panda, self.joint_info['panda_hand_joint']['index'])[0]
            gripper_pos = self.get_gripper_center()
            self._apply_gripper_to_sph(gripper_pos)
            reaction_force = self._compute_sph_reaction_force(gripper_pos)
            
            p.applyExternalForce(
                self.panda,
                -1, # self.joint_info['panda_hand_joint']['index'],
                forceObj=reaction_force,
                posObj=gripper_pos,
                flags=p.WORLD_FRAME
            )
            
            # Update visualization every step
            self._update_sph_visualization()
            
            # Update FSM
            if not self.fsm.update():
                break
                    
            # Step simulation
            p.stepSimulation()
            time.sleep(1./240.)
    

    def visualize_force_feedback(self):
        """Plot force magnitude over time"""
        if not self.force_feedback:
            print("No force feedback data recorded!")
            return

        import matplotlib.pyplot as plt
        forces = np.array(self.force_feedback)  # Convert to NumPy array
        
        # If forces is 1D (single component), plot directly
        if forces.ndim == 1:
            plt.plot(forces, label="Force (1D)")
        else:
            # Compute norm only if forces are 2D (N x 3)
            forces = np.linalg.norm(forces, axis=1)
            plt.plot(forces, label="Force Magnitude")
        
        plt.title("Gripper Reaction Forces")
        plt.xlabel("Time step")
        plt.ylabel("Force (N)")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    controller = PandaController(mode="pickup")
    try:
        controller.run()
    finally:
        controller.visualize_force_feedback()
        p.disconnect()