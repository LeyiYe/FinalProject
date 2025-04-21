import pybullet as p
import pybullet_data
import numpy as np
import time
from deformable_object import DeformableObjectSim
from enum import Enum, auto
from scipy.spatial import KDTree 

class PandaState(Enum):
    OPEN = auto()
    CLOSE = auto()
    START_CLOSER = auto()
    CLOSE_SOFT = auto()
    SQUEEZE = auto()
    SQUEEZE_HOLDING = auto()
    SQUEEZE_NO_GRAVITY = auto()
    HANG = auto()
    REORIENT = auto()
    LIN_ACC = auto()
    ANG_ACC = auto()
    DONE = auto()

class PandaFSM:
    def __init__(self, mode="pickup"):
        # Physics client setup
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Load plane and Panda hand
        self.plane_id = p.loadURDF("plane.urdf")
        self.panda = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
        """self.panda = p.loadURDF("franka_description/robots/common/hand.urdf", useFixedBase=True)"""
        
        # Get joint information
        self.num_joints = p.getNumJoints(self.panda)
        self.joint_info = self._get_joint_info()
        
        # FSM state
        self.state = PandaState.OPEN
        self.mode = mode
        self.timer = 0
        
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
            'lp_filter': {
                'running_window_size': 10,
                'averaging_window': 5
            },
            'squeeze_no_gravity': {
                'num_dp': 10,
                'torque_step_period': 10,
                'soft_object_torque_step': 0.01,
                'near_rigid_object_torque_step': 0.05,
                'soft_object_F_des': 10.0,
                'near_rigid_object_F_des': 20.0
            }
        }
        # SPH Integration Additions
        self.sph_app = DeformableObjectSim()
        self.sph_solver = None
        self.sph_particles = None
        self.sph_kdtree = None
        self.force_feedback = []
        self.gripper_poses = []
        
        # Coupling parameters
        self.coupling_stiffness = 1e6  # N/m (tune based on material)
        self.gripper_influence_radius = 0.15  # meters
        self.last_gripper_pos = np.zeros(3)
        
        # Initialize variables
        self._init_variables()
        
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
            joint_type = info[2]
            joint_limit_lower = info[8]
            joint_limit_upper = info[9]
            
            joint_info[joint_name] = {
                'index': i,
                'type': joint_type,
                'limit_lower': joint_limit_lower,
                'limit_upper': joint_limit_upper
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
    
    def _init_sph_simulation(self):
        """Initialize SPH simulation"""
        self.sph_solver = self.sph_app.create_solver()
        self.sph_particles = self.sph_app.create_particles()
        self._update_sph_kdtree()

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
            
        return total_force

    def run(self):
        """Main simulation loop with SPH integration"""
        self._init_sph_simulation()
        
        while True:
            self.timer += 1
            
            # --- PyBullet Step ---
            gripper_pos = p.getLinkState(self.panda, self.joint_info['panda_hand_joint']['index'])[0]
            self.gripper_poses.append(gripper_pos)
            
            # --- SPH Coupling ---
            self._apply_gripper_to_sph(gripper_pos)

            print(dir(self.sph_solver))
            
            # Step SPH (substepping for stability)
            """for _ in range(5):
                self.sph_solver.solve(1.0/1200.0)"""  # Smaller timesteps
            
            # Compute and apply reaction forces
            reaction_force = self._compute_sph_reaction_force(gripper_pos)
            self.force_feedback.append(reaction_force)
            
            p.applyExternalForce(
                self.panda,
                self.joint_info['panda_hand_joint']['index'],
                forceObj=reaction_force,
                posObj=gripper_pos,
                flags=p.WORLD_FRAME
            )
            
            # --- State Machine ---
            F_curr = self.get_gripper_force()
            gripper_pos = self.get_gripper_positions()
            
            # State machine logic
            if self.state == PandaState.OPEN:
                self._open_state()
            elif self.state == PandaState.CLOSE:
                self._close_state(F_curr, gripper_pos)
            elif self.state == PandaState.START_CLOSER:
                self._start_closer_state(gripper_pos)
            elif self.state == PandaState.CLOSE_SOFT:
                self._close_soft_state(F_curr)
            elif self.state == PandaState.SQUEEZE:
                self._squeeze_state(F_curr, gripper_pos)
            elif self.state == PandaState.SQUEEZE_HOLDING:
                self._squeeze_holding_state()
            elif self.state == PandaState.SQUEEZE_NO_GRAVITY:
                self._squeeze_no_gravity_state(F_curr, gripper_pos)
            elif self.state == PandaState.HANG:
                self._hang_state(F_curr)
            elif self.state == PandaState.DONE:
                # Stop all movement
                self.set_gripper_velocity(0, 0)
                self.set_gripper_torque(0, 0)
                break
                
            # Step simulation
            p.stepSimulation()
            time.sleep(1./240.)
    
    def _open_state(self):
        """Open gripper state"""
        self.open_counter += 1
        
        # Open gripper
        self.set_gripper_velocity(0.5, 0.5)
        
        # Transition after delay
        if self.open_counter > 100:
            self.state = PandaState.CLOSE
            self.open_counter = 0
    
    def _close_state(self, F_curr, gripper_pos):
        """Close gripper until contact"""
        closing_speed = -0.7
        
        # Close fingers until contact
        if F_curr[0] > 0:
            self.left_has_contacted = True
        if F_curr[1] > 0:
            self.right_has_contacted = True
            
        left_vel = closing_speed if not self.left_has_contacted else 0
        right_vel = closing_speed if not self.right_has_contacted else 0
        
        self.set_gripper_velocity(left_vel, right_vel)
        
        # Check if closed too far without contact
        if (not (self.left_has_contacted or self.right_has_contacted) and 
            all(p < 0.001 for p in gripper_pos)):
            print("Failed: Grippers closed without contacting object.")
            self.state = PandaState.DONE
            
        # Transition when both fingers have contacted
        if self.left_has_contacted and self.right_has_contacted:
            self.franka_positions_at_contact = gripper_pos
            self.state = PandaState.START_CLOSER
    
    def _start_closer_state(self, gripper_pos):
        """Reset and close to near-contact position"""
        closing_speed = -0.7
        
        # Close until near contact position
        left_vel = closing_speed if gripper_pos[0] < self.franka_positions_at_contact[0] + 0.003 else 0
        right_vel = closing_speed if gripper_pos[1] < self.franka_positions_at_contact[1] + 0.003 else 0
        
        self.set_gripper_velocity(left_vel, right_vel)
        
        # Transition when close to contact position
        if all(p < pos + 0.004 for p, pos in zip(gripper_pos, self.franka_positions_at_contact)):
            self.state = PandaState.CLOSE_SOFT
    
    def _close_soft_state(self, F_curr):
        """Soft closing with force checking"""
        # Adjust closing speed based on failures
        first_speed = 0.25
        closing_speed = -first_speed / (self.close_fails + 1)
        
        # Check if forces are too high
        if sum(F_curr) > 300:
            self.close_fails += 1
            self.left_has_contacted = False
            self.right_has_contacted = False
            print("Forces too high during close_soft, resetting state")
            self.state = PandaState.CLOSE
        
        # Check for contact
        force_threshold = 0.005
        left_in_contact = F_curr[0] > force_threshold
        right_in_contact = F_curr[1] > force_threshold
        
        left_vel = closing_speed if not left_in_contact else 0
        right_vel = closing_speed if not right_in_contact else 0
        
        self.set_gripper_velocity(left_vel, right_vel)
        
        # Transition when both fingers have contacted
        if left_in_contact and right_in_contact:
            self.grippers_pre_squeeze = self.get_gripper_positions()
            self.state = PandaState.SQUEEZE
    
    def _squeeze_state(self, F_curr, gripper_pos):
        """Squeeze to desired force"""
        self.squeeze_counter += 1
        
        # Torque control to achieve desired force
        desired_force = 10.0  # Example value
        F_des = np.array([desired_force / 2.0, desired_force / 2.0])
        
        # Simple P controller for torque
        total_F_curr = sum(F_curr)
        total_F_err = sum(F_des) - total_F_curr
        
        Kp = self.config['force_control']['Kp']
        min_torque = self.config['force_control']['min_torque']
        
        self.running_torque[0] -= min(total_F_err * Kp, 3 * Kp)
        self.running_torque[1] -= min(total_F_err * Kp, 3 * Kp)
        self.running_torque[0] = max(min_torque, self.running_torque[0])
        self.running_torque[1] = max(min_torque, self.running_torque[1])
        
        self.set_gripper_torque(self.running_torque[0], self.running_torque[1])
        
        # Check for failure conditions
        if any(p > 0.04 for p in gripper_pos):  # Exceed joint limits
            print("Grippers exceeded joint limits")
            self.state = PandaState.DONE
        
        # Check if desired force is achieved
        if abs(total_F_err) < 0.05 * desired_force:
            print("Desired squeezing force achieved")
            self.state = PandaState.HANG
    
    def _hang_state(self, F_curr):
        """Hang object to test grasp"""
        self.hang_counter += 1
        
        # Maintain force with torque control
        desired_force = 10.0  # Example value
        F_des = np.array([desired_force / 2.0, desired_force / 2.0])
        
        total_F_curr = sum(F_curr)
        total_F_err = sum(F_des) - total_F_curr
        
        Kp = self.config['force_control']['Kp']
        self.running_torque[0] -= total_F_err * Kp
        self.running_torque[1] -= total_F_err * Kp
        
        self.set_gripper_torque(self.running_torque[0], self.running_torque[1])
        
        # Check if object is still held
        if total_F_curr < 1.0:  # Object dropped
            self.pickup_success = False
            print("Object dropped during hang")
            self.state = PandaState.DONE
        elif self.hang_counter > 200:  # Successfully held
            self.pickup_success = True
            print("Pickup successful")
            self.state = PandaState.DONE

    def _squeeze_holding_state(self):
        """Holding squeeze with gradually increasing torque"""
        self.squeeze_holding_counter += 1
        
        # Gradually increase torque
        if self.squeeze_holding_counter % 10 == 0:
            self.running_torque[0] -= 0.01
            self.running_torque[1] -= 0.01
        
        self.set_gripper_torque(self.running_torque[0], self.running_torque[1])
        
        # Check for contact
        F_curr = self.get_gripper_force()
        if all(f > 0 for f in F_curr):
            self.state = PandaState.SQUEEZE_NO_GRAVITY
    
    def _squeeze_no_gravity_state(self, F_curr, gripper_pos):
        """Squeeze without gravity effects"""
        self.squeeze_no_gravity_counter += 1
        
        # Periodically increase torque
        if self.squeeze_no_gravity_counter % self.config['squeeze_no_gravity']['torque_step_period'] == 0:
            torque_step = self.config['squeeze_no_gravity']['soft_object_torque_step']
            self.running_torque[0] -= torque_step
            self.running_torque[1] -= torque_step
        
        self.set_gripper_torque(self.running_torque[0], self.running_torque[1])
        
        # Check termination conditions
        desired_force = self.config['squeeze_no_gravity']['soft_object_F_des']
        if sum(F_curr) > desired_force or self.squeeze_no_gravity_counter > 500:
            print("Squeeze no gravity complete")
            self.state = PandaState.DONE


    def visualize_force_feedback(self):
        """Plot force magnitude over time"""
        import matplotlib.pyplot as plt
        forces = np.linalg.norm(self.force_feedback, axis=1)
        plt.plot(forces)
        plt.title("Gripper Reaction Forces")
        plt.xlabel("Time step")
        plt.ylabel("Force (N)")
        plt.show()

if __name__ == "__main__":
    fsm = PandaFSM(mode="pickup")
    try:
        fsm.run()
    finally:
        fsm.visualize_force_feedback()
        p.disconnect()