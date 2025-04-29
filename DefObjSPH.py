import os
# Disable Cython and force pure Python kernels to avoid segfaults
os.environ['PYSPH_DISABLE_CYTHON'] = '1'
os.environ['PYSPH_USE_PUREPYTHON'] = '1'

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
        self.dx = 0.01                 # particle spacing
        self.hdx = 1.3              # smoothing length factor
        self.rho0 = 1000.0            # reference density
        self.E = 1e7                  # Young's modulus (Pa)
        self.nu = 0.49                # Poisson ratio
        self.c0 = 50.0                # speed of sound

        # Geometry dimensions (m)
        self.block_size = (0.3, 0.2, 0.1)
        self.platform_size = (1.0, 0.6, 0.02)
        self.gripper_size = (0.05, 0.1, 0.2)

    def create_block(self, center, size):
        # Dynamic resolution
        nx = max(3, int(round(size[0]/self.dx)))
        ny = max(3, int(round(size[1]/self.dx)))
        nz = max(3, int(round(size[2]/self.dx)))
        xs = np.linspace(center[0] - size[0]/2 + self.dx/2,
                         center[0] + size[0]/2 - self.dx/2, nx)
        ys = np.linspace(center[1] - size[1]/2 + self.dx/2,
                         center[1] + size[1]/2 - self.dx/2, ny)
        zs = np.linspace(center[2] - size[2]/2 + self.dx/2,
                         center[2] + size[2]/2 - self.dx/2, nz)
        x, y, z = np.meshgrid(xs, ys, zs, indexing='ij')
        return x.ravel(), y.ravel(), z.ravel()

    def create_particles(self):
        # Block
        bx, by, bz = self.create_block(
            center=(0,0,self.platform_size[2] + self.block_size[2]/2),
            size=self.block_size)
        block = get_particle_array(name='block', x=bx, y=by, z=bz,
                                   h=self.hdx*self.dx,
                                   m=self.dx**3*self.rho0,
                                   rho=self.rho0)
        # Elastic and EOS properties
        block.add_property('E');       block.E[:] = self.E
        block.add_property('nu');      block.nu[:] = self.nu
        block.add_property('rho_ref'); block.rho_ref[:] = self.rho0
        block.add_property('c0_ref');  block.c0_ref[:] = self.c0
        # Shear modulus
        block.add_property('G');       block.G[:] = self.E/(2*(1+self.nu))
                # Continuity tracer and viscosity properties
        block.add_property('arho'); block.arho[:] = 1.0/self.rho0
        block.add_property('cs');   block.cs[:]   = self.c0

        for arr in ['as22', 'as22', 'as12','as00', 'as11', 'as01', 'as02',
                    'ay', 'ax', 'az', 'e', 'e0', 'eho0', 's000', 's010', 's020',
                    's110', 's120', 's220', 'u0', 'v0', 'w0', 'x0', 'y0', 'z0' ]:
            block.add_property(arr)
            

        # Stress (rij) and deviatoric stress (sij)
        for i in range(self.dim):
            for j in range(i, self.dim):
                block.add_property(f'r{i}{j}'); block.add_property(f's{i}{j}')
        # Velocity gradients vij
        for i in range(self.dim):
            for j in range(self.dim):
                block.add_property(f'v{i}{j}')
        # Pressure-rate and splitting tracer
        block.add_property('wdeltap')
        block.add_property('n'); block.n[:] = 1.0/self.rho0

        particles = [block]

        # Platform and grippers
        def make_rigid(name, center, size):
            x,y,z = self.create_block(center, size)
            pa = get_particle_array(
                name=name, x=x, y=y, z=z,
                h=0.02*2.0, m=1e12, rho=self.rho0,
                is_boundary=1, is_rigid=1)
            pa.add_property('arho', type='double', default=1.0/self.rho0)
            pa.add_property('cs', type='double', default=self.c0)
            pa.add_property('z0', type='double', default=pa.z.copy())
            for i in range(self.dim):
                for j in range(i, self.dim):
                    pa.add_property(f'r{i}{j}')
                    pa.add_property(f's{i}{j}')
                for j in range(self.dim):
                    pa.add_property(f'v{i}{j}')
            pa.add_property('wdeltap'); pa.add_property('n')
            particles.append(pa)
            return pa

        platform = make_rigid('platform', (0,0,self.platform_size[2]/2), self.platform_size)
        gripper1 = make_rigid('gripper1', (-0.4, 0, self.platform_size[2] + self.gripper_size[2]/2), self.gripper_size)
        gripper2 = make_rigid('gripper2', (0.4, 0, self.platform_size[2] + self.gripper_size[2]/2), self.gripper_size)

        return particles

    def create_scheme(self):
        elastic = ElasticSolidsScheme(
            elastic_solids=['block'],
            solids=['platform','gripper1','gripper2'],
            dim=self.dim,
            artificial_stress_eps=0.5,
            xsph_eps=0.5)
        return SchemeChooser(default='elastic', elastic=elastic)

    def configure_scheme(self):
        self.scheme.configure_solver(dt=5e-5, tf=2.0, pfreq=50)

    def create_equations(self):
        eqns = self.scheme.get_equations()
        eqns.append(Group(equations=[BodyForce(dest='block', sources=None,
                                                fx=0, fy=0, fz=-9.81)],
                           real=False))
        return eqns

    def post_step(self, solver):
        block = self.particles[0]
        g1, g2 = self.particles[2], self.particles[3]
        dt = solver.dt
        
        half_block = 0.5*self.block_size[0]
        half_grip  = 0.5*self.gripper_size[0]
        target = -half_block - half_grip + 0.02

        if g1.x[0] < target:
            g1.u[:] =  0.2; g2.u[:] = -0.2
            g1.v[:] = g2.v[:] = 0; g1.w[:] = g2.w[:] = 0
        else:
            g1.u[:] = g2.u[:] = 0
            g1.v[:] = g2.v[:] = 0
            g1.w[:] = g2.w[:] = 0.3

        for gr in (g1, g2):
            gr.x += gr.u * dt; gr.y += gr.v * dt; gr.z += gr.w * dt
            gr.z0 += gr.w * dt

if __name__=='__main__':
    app = GraspDeformableBlock()
    app.run()
