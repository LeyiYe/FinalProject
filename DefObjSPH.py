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
        self.gripper_size = (0.05, 0.2, 0.1)

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
        # Additional solid fields (required by scheme)
        for prop in ['eo', 'ae', 'arho','as12', 'as22', 'as02', 'as11', 'as01', 'as00', 'ay', 'ax', 'az',
                    'e', 'e0', 'rho0', 's000', 's010', 's020', 's110', 's120', 's220', 'u0', 'v0',
                    'w0', 'x0', 'y0', 'z0']:
            block.add_property(prop)
        block.arho[:] = 1.0/self.rho0

        # Initialize scheme properties (will add continuity, stress, viscosity fields)
        particles = [block]

        # Platform and grippers
        def make_rigid(name, center, size):
            x,y,z = self.create_block(center, size)
            pa = get_particle_array(
                name=name, x=x, y=y, z=z,
                h=self.hdx*self.dx, m=1e12, rho=self.rho0,
                is_boundary=1, is_rigid=1)
            # Mirror same extra fields
            for arr in (pa,):
                arr.add_property('arho'); arr.arho[:] = 1.0/self.rho0
                arr.add_property('cs');   arr.cs[:]   = self.c0
                for i in range(self.dim):
                    for j in range(i, self.dim):
                        arr.add_property(f'r{i}{j}'); arr.add_property(f's{i}{j}')
                    for j in range(self.dim):
                        arr.add_property(f'v{i}{j}')
                arr.add_property('wdeltap'); arr.add_property('n')
            particles.append(pa)
            return pa
        platform = make_rigid('platform', (0,0,self.platform_size[2]/2), self.platform_size)
        gripper1 = make_rigid('gripper1', (
            -0.4, 0, self.platform_size[2] + self.gripper_size[2]/2), self.gripper_size)
        gripper2 = make_rigid('gripper2', (
             0.4, 0, self.platform_size[2] + self.gripper_size[2]/2), self.gripper_size)

                # Manually add any missing fields required by the solid scheme
        for arr in particles:
            # speed of sound tracer for artificial viscosity
            if 'cs' not in arr.properties:
                arr.add_property('cs'); arr.cs[:] = self.c0
            # reciprocal density tracer for continuity
            if 'arho' not in arr.properties:
                arr.add_property('arho'); arr.arho[:] = 1.0/self.rho0
        for arr in particles:
            # reciprocal density tracer for continuity
            if 'arho' not in arr.properties:
                arr.add_property('arho'); arr.arho[:] = 1.0/self.rho0
            # artificial stress components
            for i in range(self.dim):
                for j in range(i, self.dim):
                    if f'r{i}{j}' not in arr.properties:
                        arr.add_property(f'r{i}{j}')
                    if f's{i}{j}' not in arr.properties:
                        arr.add_property(f's{i}{j}')
            # velocity gradients
            for i in range(self.dim):
                for j in range(self.dim):
                    if f'v{i}{j}' not in arr.properties:
                        arr.add_property(f'v{i}{j}')
            # pressure-rate and splitting tracer
            if 'wdeltap' not in arr.properties:
                arr.add_property('wdeltap')
            if 'n' not in arr.properties:
                arr.add_property('n')

        print("G-range", block.G.min(), block.G.max())
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
        # Gravity on block
        eqns.append(Group(equations=[BodyForce(dest='block', sources=None,
                                                fx=0, fy=0, fz=-9.81)],
                           real=False))
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
        target = -half_block - half_grip + 0.02
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
        mask = block.z < zmin
        if mask.any():
            block.z[mask] = zmin
            # zero translational velocity
            block.u[mask] = block.v[mask] = block.w[mask] = 0.0
            # wipe elastic strain & stress so they don't accumulate
            for p in ('e','r','s','as'):
                for i in range(3):
                    for j in range(3):
                        name = f'{p}{i}{j}' if i<=j else None
                        if name and name in block.properties:
                            block.get(name)[mask] = 0.0
if __name__=='__main__':
    app = GraspDeformableBlock()
    app.run()
