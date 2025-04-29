import os
# Disable Cython JIT compilation to avoid build errors
os.environ['PYSPH_DISABLE_CYTHON'] = '1'
os.environ['PYSPH_USE_PUREPYTHON'] = '1'

import numpy as np
from pysph.base.utils import get_particle_array
from pysph.solver.application import Application
from pysph.sph.scheme import SchemeChooser
from pysph.sph.solid_mech.basic import ElasticSolidsScheme
from pysph.sph.equation import Group
from pysph.sph.basic_equations import BodyForce
from pysph.sph.wc.transport_velocity import MomentumEquationPressureGradient

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
        block.add_property('E');       block.E[:] = self.E
        block.add_property('nu');      block.nu[:] = self.nu
        block.add_property('rho_ref'); block.rho_ref[:] = self.rho0
        block.add_property('c0_ref');  block.c0_ref[:] = self.c0
        block.add_property('G')
        block.add_property('ax'); block.add_property('ay'); block.add_property('az')
        block.add_property('e'); block.add_property('e0')
        block.add_property('rho0'); block.add_property('u0'); block.add_property('v0'); block.add_property('w0')
        block.add_property('x0'); block.add_property('y0'); block.add_property('z0')
        block.add_property('ae')
        block.add_property('d_idx'); block.add_property('d_au')
        block.add_property('d_av'); block.add_property('d_aw'); block.add_property('d_auhat')
        block.add_property('d_avhat'); block.add_property('d_awhat')
        block.add_property('V'); block.add_property('auhat'); block.add_property('awhat')
        block.add_property('avhat')
        # Allocate velocity gradient, artificial stress, and stress arrays
        for i in range(self.dim):
            for j in range(self.dim):
                block.add_property(f'v{i}{j}')
                block.add_property(f'as{i}{j}')
                block.add_property(f's{i}{j}0')
        for i in range(self.dim):
            for j in range(i, self.dim):
                block.add_property(f'r{i}{j}')
                block.add_property(f's{i}{j}')

        # Rigid platform as boundary
        px, py, pz = self.create_block(
            center=(0.0, 0.0, self.platform_size[2]/2), size=self.platform_size
        )
        platform = get_particle_array(name='platform', x=px, y=py, z=pz,
                                      h=self.hdx * self.dx, m=1e12, rho=self.rho0,
                                      is_boundary=1, is_rigid=1)
        # Left gripper (rigid)
        g1x, g1y, g1z = self.create_block(
            center=(-0.4, 0.0, self.platform_size[2] + self.gripper_size[2]/2),
            size=self.gripper_size
        )
        gripper1 = get_particle_array(name='gripper1', x=g1x, y=g1y, z=g1z,
                                      h=self.hdx * self.dx, m=1e12, rho=self.rho0,
                                      is_boundary=1, is_rigid=1)
        # Right gripper (rigid)
        g2x, g2y, g2z = self.create_block(
            center=(0.4, 0.0, self.platform_size[2] + self.gripper_size[2]/2),
            size=self.gripper_size
        )
        gripper2 = get_particle_array(name='gripper2', x=g2x, y=g2y, z=g2z,
                                      h=self.hdx * self.dx, m=1e12, rho=self.rho0,
                                      is_boundary=1, is_rigid=1)
        # Additional fields for continuity and artificial stress
        for arr in (block, platform, gripper1, gripper2):
            arr.add_property('cs')
            if 'arho' not in arr.properties:
                arr.add_property('arho'); arr.arho[:] = 1.0/self.rho0
            arr.add_property('n'); arr.add_property('wdeltap')
            for i in range(self.dim):
                for j in range(i, self.dim):
                    arr.add_property(f'r{i}{j}'); arr.add_property(f's{i}{j}')
        return [block, platform, gripper1, gripper2]

    def create_scheme(self):
        elastic = ElasticSolidsScheme(
            elastic_solids=['block'],
            solids=['platform','gripper1','gripper2'],
            dim=self.dim
        )
        return SchemeChooser(default='elastic', elastic=elastic)

    def configure_scheme(self):
        self.scheme.configure_solver(dt=1e-4, tf=2.0, pfreq=100)

    def create_equations(self):
        eqns = self.scheme.get_equations()
        eqns.append(
            Group(equations=[
                BodyForce(dest='block', sources=None, fx=0, fy=0, fz=-9.81)
            ], real=False)
        )
        return eqns
    
    def post_step(self, solver):
        """
        After each step: move grippers by position-control and clamp block bottom but allow SPH compression.
        """
        block = self.particles[0]
        g1, g2 = self.particles[2], self.particles[3]
        dt = solver.dt
        # compute jaw target
        half_block = 0.5*self.block_size[0]
        half_grip  = 0.5*self.gripper_size[0]
        target = -half_block - half_grip
        # approach until contact then lift
        if g1.x[0] < target:
            g1.u[:] =  0.2; g2.u[:] = -0.2
            g1.v[:] = g2.v[:] = 0; g1.w[:] = g2.w[:] = 0
        else:
            g1.u[:] = g2.u[:] = 0; g1.v[:] = g2.v[:] = 0; g1.w[:] = g2.w[:] = 0.3
        # integrate rigid bodies
        for gr in (g1, g2):
            gr.x += gr.u*dt; gr.y += gr.v*dt; gr.z += gr.w*dt
        # clamp block at floor but retain SPH deformation
        zmin = self.platform_size[2] + 0.5*self.dx
        block.z[:] = np.maximum(block.z, zmin)
        block.w[:] = np.where(block.z<=zmin, np.maximum(block.w,0), block.w)

if __name__ == '__main__':
    app = GraspDeformableBlock()
    app.run()
