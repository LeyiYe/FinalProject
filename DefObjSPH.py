import os
# Disable Cython JIT compilation to avoid build errors
os.environ['PYSPH_DISABLE_CYTHON'] = '1'

import numpy as np
from pysph.base.utils import get_particle_array
from pysph.solver.application import Application
from pysph.sph.scheme import SchemeChooser
from pysph.sph.solid_mech.basic import ElasticSolidsScheme
from pysph.sph.equation import Group
from pysph.sph.basic_equations import BodyForce

class GraspDeformableBlock(Application):
    def initialize(self):
        # Simulation parameters
        self.dim = 3
        self.dx = 0.02                 # particle spacing
        self.hdx = 1.2                # smoothing length factor
        self.rho0 = 1000.0            # reference density (kg/m^3)
        self.E = 1e5                  # Young's modulus (Pa) — softer for rubber-like deformation
        self.nu = 0.49                # Poisson ratio — near incompressible for rubber
        self.c0 = 50.0                # speed of sound for EOS

        # Geometry dimensions (m)
        self.block_size = (0.3, 0.2, 0.1)
        self.platform_size = (1.0, 0.6, 0.02)
        self.gripper_size = (0.05, 0.2, 0.1)

    def create_block(self, center, size):
        # Generate a regular grid of particles for a rectangular prism
        nx = 30
        ny = 10
        nz = 5
        xs = np.linspace(center[0] - size[0] / 2 + self.dx/2,
                         center[0] + size[0] / 2 - self.dx/2, nx)
        ys = np.linspace(center[1] - size[1] / 2 + self.dx/2,
                         center[1] + size[1] / 2 - self.dx/2, ny)
        zs = np.linspace(center[2] - size[2] / 2 + self.dx/2,
                         center[2] + size[2] / 2 - self.dx/2, nz)
        x, y, z = np.meshgrid(xs, ys, zs, indexing='ij')
        return x.ravel(), y.ravel(), z.ravel()

    def create_particles(self):
        # Deformable block particles
        bx, by, bz = self.create_block(
            center=(0.0, 0.0, self.platform_size[2] + self.block_size[2]/2),
            size=self.block_size
        )
        block = get_particle_array(name='block', x=bx, y=by, z=bz,
                                   h=self.hdx * self.dx,
                                   m=self.dx**3 * self.rho0,
                                   rho=self.rho0)
        # set material and EOS properties on block
        block.add_property('E')
        block.add_property('nu')
        block.add_property('rho_ref')
        block.add_property('c0_ref')
        block.add_property('G')
        block.add_property('ax')
        block.add_property('ay')
        block.add_property('az')
        block.add_property('e')
        block.add_property('e0')
        block.add_property('rho0')
        block.add_property('u0')
        block.add_property('v0')
        block.add_property('w0')
        block.add_property('x0')
        block.add_property('y0')
        block.add_property('z0')
        block.add_property('ae')
        block.E[:] = self.E
        block.nu[:] = self.nu
        block.rho_ref[:] = self.rho0
        block.c0_ref[:] = self.c0
        # Allocate velocity gradient arrays (v_ij) for scheme
        for i in range(self.dim):
            for j in range(self.dim):
                block.add_property(f'v{i}{j}')
                block.add_property(f'as{i}{j}')
                block.add_property(f's{i}{j}0')
        # Allocate stress and rotation arrays for MonaghanArtificialStress
        for i in range(self.dim):
            for j in range(i, self.dim):
                block.add_property(f'r{i}{j}')
                block.add_property(f's{i}{j}')


        # Rigid platform as boundary
        px, py, pz = self.create_block(
            center=(0.0, 0.0, self.platform_size[2] / 2),
            size=self.platform_size
        )
        platform = get_particle_array(name='platform', x=px, y=py, z=pz,
                                      h=self.hdx * self.dx,
                                      m=1e12, rho=self.rho0,
                                      is_boundary=1, is_rigid=1)

        # Left gripper (rigid)
        g1x, g1y, g1z = self.create_block(
            center=(-0.4, 0.0, self.platform_size[2] + self.gripper_size[2]/2),
            size=self.gripper_size
        )
        gripper1 = get_particle_array(name='gripper1', x=g1x, y=g1y, z=g1z,
                                      h=self.hdx * self.dx,
                                      m=1e12, rho=self.rho0,
                                      is_boundary=1, is_rigid=1)

        # Right gripper (rigid)
        g2x, g2y, g2z = self.create_block(
            center=(0.4, 0.0, self.platform_size[2] + self.gripper_size[2]/2),
            size=self.gripper_size
        )
        gripper2 = get_particle_array(name='gripper2', x=g2x, y=g2y, z=g2z,
                                      h=self.hdx * self.dx,
                                      m=1e12, rho=self.rho0,
                                      is_boundary=1, is_rigid=1)

        # Allocate additional fields required by stress-based equations
        for arr in [block, platform, gripper1, gripper2]:
            # Reciprocal density tracer needed for continuity equation
            arr.add_property('cs')
            if 'arho' not in arr.properties:
                arr.add_property('arho')
                arr.arho[:] = 1.0 / self.rho0
            # Nyquist field for particle splitting
            if 'n' not in arr.properties:
                arr.add_property('n')
            # Pressure rate change for Monaghan & momentum update
            if 'wdeltap' not in arr.properties:
                arr.add_property('wdeltap')
            # Stress and artificial stress fields
            for i in range(self.dim):
                for j in range(i, self.dim):
                    prop_r = f'r{i}{j}'
                    prop_s = f's{i}{j}'
                    if prop_r not in arr.properties:
                        arr.add_property(prop_r)
                    if prop_s not in arr.properties:
                        arr.add_property(prop_s)

        return [block, platform, gripper1, gripper2] 

    def create_scheme(self):
        # ElasticSolidsScheme expects (elastic_solids, solids, dim)
        elastic = ElasticSolidsScheme(
            elastic_solids=['block'],
            solids=['platform', 'gripper1', 'gripper2'],
            dim=self.dim
        )
        return SchemeChooser(default='elastic', elastic=elastic)

    def configure_scheme(self):
        self.scheme.configure_solver(dt=1e-4, tf=2.0, pfreq=100)

    def create_equations(self):
        # Use the scheme's predefined solid mechanics + contact equations
        eqns = self.scheme.get_equations()
        # Add gravity as a body force on the deformable block
        eqns.append(
            Group(
                equations=[BodyForce(dest='block', sources=None, fx=0.0, fy=0.0, fz=-9.81)],
                real=False
            )
        )
        return eqns

    def post_step(self, solver):
        """
        Called after each time-step: update gripper motion via position control, enforce floor, and integrate rigid bodies.
        """
        block = self.particles[0]
        g1, g2 = self.particles[2], self.particles[3]
        dt = solver.dt

        # Target approach position so gripper inner faces meet block edges
        half_block = self.block_size[0] * 0.5
        half_grip = self.gripper_size[0] * 0.5
        target_pos = -half_block - half_grip  # -0.3 - 0.025 = -0.325

        # 1) Position-based gripper approach and lift
        v_in = 0.2
        v_lift = 0.3
        # Check left gripper: moving right if still outside target
        if g1.x[0] > target_pos:
            g1.u[:] =  v_in
            g2.u[:] = -v_in
            g1.v[:] = g2.v[:] = 0.0
            g1.w[:] = g2.w[:] = 0.0
        else:
            # Stop horizontal, start lifting
            g1.u[:] = g2.u[:] = 0.0
            g1.v[:] = g2.v[:] = 0.0
            g1.w[:] = g2.w[:] = v_lift

        # 2) Integrate rigid-body motion manually
        for gr in (g1, g2):
            gr.x += gr.u * dt
            gr.y += gr.v * dt
            gr.z += gr.w * dt

        # 3) Enforce floor: clamp block particles from going below platform
        floor_z = self.platform_size[2] + 0.5 * self.dx
        mask = block.z < floor_z
        if mask.any():
            block.z[mask] = floor_z
            block.u[mask] = 0.0
            block.v[mask] = 0.0
            block.w[mask] = 0.0 

if __name__ == '__main__':
    app = GraspDeformableBlock()
    app.run()
