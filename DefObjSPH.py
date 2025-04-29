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
        self.dx = 0.05                 # particle spacing
        self.hdx = 1.2                # smoothing length factor
        self.rho0 = 1000.0            # reference density (kg/m^3)
        self.E = 1e6                  # Young's modulus (Pa)
        self.nu = 0.3                 # Poisson ratio
        self.c0 = 50.0                # speed of sound for EOS

        # Geometry dimensions (m)
        self.block_size = (0.6, 0.2, 0.1)
        self.platform_size = (1.0, 0.6, 0.02)
        self.gripper_size = (0.05, 0.2, 0.1)

    def create_block(self, center, size):
        # Generate a regular grid of particles for a rectangular prism
        nx = max(2, int(size[0] / self.dx))
        ny = max(2, int(size[1] / self.dx))
        nz = max(2, int(size[2] / self.dx))
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
        block.add_property('arho')
        block.E[:] = self.E
        block.nu[:] = self.nu
        block.rho_ref[:] = self.rho0
        block.c0_ref[:] = self.c0
        block.arho[:] = 1.0 / self.rho0
        # Allocate velocity gradient arrays (v_ij) for scheme
        for i in range(self.dim):
            for j in range(self.dim):
                block.add_property(f'v{i}{j}')
        # Allocate stress and rotation arrays for MonaghanArtificialStress
        for i in range(self.dim):
            for j in range(i, self.dim):
                block.add_property(f'r{i}{j}')
                block.add_property(f's{i}{j}')
        # Allocate velocity gradient arrays
        for i in range(self.dim):
            for j in range(self.dim):
                block.add_property(f'v{i}{j}')

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
            # Nyquist field for particle splitting
            arr.add_property('n')
            # Pressure rate change for Monaghan & momentum update
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
        t = solver.t
        v_in, v_lift = 0.2, 0.3
        g1, g2 = self.particles[2], self.particles[3]
        if t < 0.5:
            g1.u[:] = v_in;    g2.u[:] = -v_in
            g1.v[:] = g2.v[:] = 0; g1.w[:] = g2.w[:] = 0
        else:
            g1.u[:] = g2.u[:] = 0
            g1.v[:] = g2.v[:] = 0
            g1.w[:] = g2.w[:] = v_lift

if __name__ == '__main__':
    app = GraspDeformableBlock()
    app.run()
