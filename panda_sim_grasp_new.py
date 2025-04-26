import time
import numpy as np
import math
from deformable_object import DeformableObjectSim

# Constants for Panda robot
useNullSpace = 1
ikSolver = 0
pandaEndEffectorIndex = 11
pandaNumDofs = 7

# Joint limits
ll = [-7]*pandaNumDofs
ul = [7]*pandaNumDofs
jr = [7]*pandaNumDofs

# Default joint positions
jointPositions = [0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32, 0.02, 0.02]
rp = jointPositions

class PandaSim(object):
    def __init__(self, bullet_client, offset):
        self.bullet_client = bullet_client
        self.bullet_client.setPhysicsEngineParameter(solverResidualThreshold=0)
        self.offset = np.array(offset)
        
        # Load tray/workspace
        flags = self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        self.bullet_client.loadURDF("tray/traybox.urdf", 
                                   [0+offset[0], 0+offset[1], -0.6+offset[2]], 
                                   [-0.5, -0.5, -0.5, 0.5], 
                                   flags=flags)
        
        # Initialize SPH deformable object
        self.sph_system = DeformableObjectSim(particle_radius=0.005)
        self.sph_particles = self.sph_system.create_particles()[0]
        self.sph_solver = self.sph_system.create_solver()
        
        # Position SPH object
        self._position_sph_object()
        
        # Create visualization
        self._create_sph_visualization()
        
        # Load Panda robot
        orn = [-0.707107, 0.0, 0.0, 0.707107]
        self.panda = self.bullet_client.loadURDF(
            "franka_panda/panda.urdf", 
            np.array([0,0,0])+self.offset, 
            orn, 
            useFixedBase=True, 
            flags=flags
        )
        
        # Initialize robot state
        index = 0
        self.state = 0  # Initial state
        self.control_dt = 1./240.
        self.finger_target = 0.04  # Start with open gripper
        self.gripper_height = 0.2
        
        # Create finger constraint
        c = self.bullet_client.createConstraint(
            self.panda, 9, self.panda, 10,
            jointType=self.bullet_client.JOINT_GEAR,
            jointAxis=[1, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0]
        )
        self.bullet_client.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)
        
        # Initialize joints
        for j in range(self.bullet_client.getNumJoints(self.panda)):
            self.bullet_client.changeDynamics(self.panda, j, linearDamping=0, angularDamping=0)
            info = self.bullet_client.getJointInfo(self.panda, j)
            jointType = info[2]
            
            if jointType == self.bullet_client.JOINT_PRISMATIC:
                self.bullet_client.resetJointState(self.panda, j, jointPositions[index]) 
                index += 1
            elif jointType == self.bullet_client.JOINT_REVOLUTE:
                self.bullet_client.resetJointState(self.panda, j, jointPositions[index]) 
                index += 1
                
        self.t = 0.
        self.prev_pos = None

    def _position_sph_object(self):
        """Center SPH object on platform"""
        platform_center = np.array([0, 0.05, -0.6])  # Above platform center
        
        # Calculate particle bounds
        min_x, max_x = np.min(self.sph_particles.x), np.max(self.sph_particles.x)
        min_y, max_y = np.min(self.sph_particles.y), np.max(self.sph_particles.y)
        min_z = np.min(self.sph_particles.z)
        
        # Calculate required offsets
        x_offset = platform_center[0] - (min_x + max_x)/2
        y_offset = platform_center[1] - (min_y + max_y)/2
        z_offset = platform_center[2] - min_z
        
        # Apply offsets
        self.sph_particles.x += x_offset
        self.sph_particles.y += y_offset
        self.sph_particles.z += z_offset

    def _create_sph_visualization(self):
        """Create visual representation of SPH particles"""
        self.sph_visuals = []
        particle_shape = self.bullet_client.createVisualShape(
            self.bullet_client.GEOM_SPHERE,
            radius=self.sph_system.particle_radius,
            rgbaColor=[1, 0, 0, 0.7]
        )

        for i in range(len(self.sph_particles.x)):
            visual = self.bullet_client.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=particle_shape,
                basePosition=[
                    self.sph_particles.x[i],
                    self.sph_particles.y[i],
                    self.sph_particles.z[i]
                ],
                baseCollisionShapeIndex=-1
            )
            self.sph_visuals.append(visual)

    def _update_sph_visualization(self):
        """Update particle positions in visualization"""
        for i, visual in enumerate(self.sph_visuals):
            self.bullet_client.resetBasePositionAndOrientation(
                visual,
                posObj=[self.sph_particles.x[i], 
                       self.sph_particles.y[i], 
                       self.sph_particles.z[i]],
                ornObj=[0, 0, 0, 1]
            )

    def get_gripper_center(self):
        """Get current gripper position"""
        link_state = self.bullet_client.getLinkState(
            self.panda, 
            pandaEndEffectorIndex
        )
        return list(link_state[0])  # World position

    def _handle_sph_coupling(self):
        """Handle interaction between gripper and SPH particles"""
        gripper_pos = self.get_gripper_center()
        particles = self.sph_particles
        stiffness = 1e4
        gripper_force = np.zeros(3)
        
        # Vectorized interaction calculation
        dx = particles.x - gripper_pos[0]
        dy = particles.y - gripper_pos[1]
        dz = particles.z - gripper_pos[2]
        dist_sq = dx**2 + dy**2 + dz**2
        mask = dist_sq < 0.02**2  # Interaction radius squared
        
        if np.any(mask):
            dist = np.sqrt(dist_sq[mask])
            nx = dx[mask] / dist
            ny = dy[mask] / dist
            nz = dz[mask] / dist
            
            # Spring force (displacement from equilibrium)
            displacement = 0.02 - dist
            force_magnitude = stiffness * displacement
            
            # Apply forces to particles
            particles.u[mask] += force_magnitude * nx * self.sph_system.dt
            particles.v[mask] += force_magnitude * ny * self.sph_system.dt
            particles.w[mask] += force_magnitude * nz * self.sph_system.dt
            
            # Calculate reaction force on gripper
            gripper_force = -np.array([
                np.sum(force_magnitude * nx),
                np.sum(force_magnitude * ny),
                np.sum(force_magnitude * nz)
            ])
            
        # Apply reaction force to gripper
        self.bullet_client.applyExternalForce(
            self.panda,
            -1,  # Apply to base
            forceObj=gripper_force,
            posObj=gripper_pos,
            flags=self.bullet_client.WORLD_FRAME
        )

    def update_state(self):
        """Finite State Machine for grasping sequence"""
        # Update SPH simulation
        for _ in range(5):
            self.sph_system.step()
            
        # Handle coupling
        self._handle_sph_coupling()
            
        # Update visualization
        self._update_sph_visualization()

        # State transitions
        if self.state == 0:  # Initial state
            self.state = 1  # Move to approach position
            
        elif self.state == 1:  # Approach object
            target_pos = self._get_pre_grasp_position()
            if self._reached_position(target_pos):
                self.state = 3  # Move to pre-grasp position
                
        elif self.state == 3:  # Pre-grasp position (above object)
            target_pos = self._get_pre_grasp_position()
            if self._reached_position(target_pos):
                self.state = 4  # Move to grasp position
                
        elif self.state == 4:  # Grasp position
            target_pos = self._get_grasp_position()
            if self._reached_position(target_pos):
                self.state = 6  # Close gripper
                
        elif self.state == 6:  # Gripper closed
            if self._grasp_is_secure():
                self.state = 7  # Lift object
                
        elif self.state == 7:  # Lift object
            target_pos = self._get_lift_position()
            if self._reached_position(target_pos, threshold=0.02):
                self.state = 8  # Done
                

    def _get_pre_grasp_position(self):
        """Get position 5cm above object center"""
        sph_center = [
            np.mean(self.sph_particles.x),
            np.mean(self.sph_particles.y) + 0.05,  # 5cm above
            np.mean(self.sph_particles.z)
        ]
        return sph_center

    def _get_grasp_position(self):
        """Get position at object center"""
        sph_center = [
            np.mean(self.sph_particles.x),
            np.mean(self.sph_particles.y)+0.005,  # 2cm into object
            np.mean(self.sph_particles.z)
        ]
        return sph_center

    def _get_lift_position(self):
        """Get lifted position"""
        if self.prev_pos is None:
            self.prev_pos = self._get_grasp_position()
        self.prev_pos[1] += 0.001  # Move up slowly
        return self.prev_pos

    def _reached_position(self, target_pos, threshold=0.01):
        """Check if gripper has reached target position"""
        gripper_pos = self.get_gripper_center()
        distance = np.linalg.norm(np.array(gripper_pos) - np.array(target_pos))
        return distance < threshold

    def _grasp_is_secure(self):
        """Check if grasp is secure by measuring particle contacts"""
        gripper_pos = self.get_gripper_center()
        particles = self.sph_particles
        count = 0
        
        for i in range(len(particles.x)):
            dist = np.linalg.norm([
                particles.x[i] - gripper_pos[0],
                particles.y[i] - gripper_pos[1],
                particles.z[i] - gripper_pos[2]
            ])
            if dist < 0.03:  # 3cm interaction radius
                count += 1
                
        return count > 10  # At least 10 particles in contact

    def step(self, graspWidth):
        """Main simulation step"""
        self.update_state()
        
        # Set gripper width based on state
        if self.state == 6:  # Close gripper
            self.finger_target = 0.01
        elif self.state == 5:  # Open gripper
            self.finger_target = 0.06
            
        # Calculate target position based on state
        if self.state == 1:  # Approach
            pos = [0, 0.2, -0.6]  # Default approach position
        elif self.state == 3:  # Pre-grasp
            pos = self._get_pre_grasp_position()
            self.prev_pos = pos
        elif self.state == 4:  # Grasp
            pos = self._get_grasp_position()
            self.prev_pos = pos
        elif self.state == 7:  # Lift
            pos = self._get_lift_position()
        else:  # Default circular motion
            t = self.t
            self.t += self.control_dt
            pos = [
                self.offset[0] + 0.2 * math.sin(1.5 * t),
                0.2,
                self.offset[2] - 0.6 + 0.1 * math.cos(1.5 * t)
            ]

        # Calculate IK and control joints
        orn = self.bullet_client.getQuaternionFromEuler([math.pi/2., 0., 0.])
        jointPoses = self.bullet_client.calculateInverseKinematics(
            self.panda, pandaEndEffectorIndex, pos, orn, 
            ll, ul, jr, rp, maxNumIterations=200
        )

        # Apply joint controls
        for i in range(pandaNumDofs):
            self.bullet_client.setJointMotorControl2(
                self.panda, i, 
                self.bullet_client.POSITION_CONTROL, 
                jointPoses[i],
                force=500,
                positionGain=0.3,
                velocityGain=1.0
            )

        # Control fingers
        for i in [9, 10]:
            self.bullet_client.setJointMotorControl2(
                self.panda, i,
                self.bullet_client.POSITION_CONTROL,
                self.finger_target,
                force=20
            )

        # Update SPH simulation
        for _ in range(5):
            self.sph_system.step()
            


class PandaSimAuto(PandaSim):
    def __init__(self, bullet_client, offset):
        PandaSim.__init__(self, bullet_client, offset)
        self.state_t = 0
        self.cur_state = 0
        self.states = [0, 1, 3, 4, 6, 7]  # State sequence
        self.state_durations = [2, 3, 2, 2, 3, 5]  # Duration for each state
    
    def update_state(self):
        self.state_t += self.control_dt
        
        # Check if current state duration has elapsed
        if self.state_t > self.state_durations[self.cur_state]:
            self.cur_state += 1
            if self.cur_state >= len(self.states):
                self.cur_state = 0
            self.state_t = 0
            self.state = self.states[self.cur_state]