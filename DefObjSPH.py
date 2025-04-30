#!/usr/bin/env python3
import numpy as np

from pysph.base.utils           import get_particle_array
from pysph.solver.application   import Application
from pysph.sph.scheme           import SchemeChooser
from pysph.sph.solid_mech.basic import get_particle_array_elastic_dynamics
from pysph.sph.basic_equations  import BodyForce
from pysph.sph.rigid_body       import RigidBody3DScheme

class GraspDeformableBlock(Application):
    def initialize(self):
        # Simulation parameters
        self.dim     = 3
        self.dx      = 0.01
        self.hdx     = 1.3
        self.rho0    = 1000.0

        # Material properties (rubber-like block)
        self.E_block = 1e7
        self.nu      = 0.49
        self.c0      = 50.0

        # Geometry
        self.block_size    = (0.3, 0.2, 0.1)
        self.platform_size = (1.0, 0.6, 0.04)
        self.gripper_size  = (0.05, 0.1, 0.2)

        # Gravity flag
        self.gravity_enabled = True

    def create_box(self, center, size):
        # Uniform grid in a box
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
        # 1) Deformable block
        bx, by, bz = self.create_box(
            center=(0, 0, self.platform_size[2] + 0.5*self.block_size[2]),
            size=self.block_size
        )
        block = get_particle_array_elastic_dynamics(
            name='block',
            x=bx, y=by, z=bz,
            h=self.hdx*self.dx,
            m=self.dx**3*self.rho0,
            rho=self.rho0,
            constants={
                'E':       self.E_block,
                'nu':      self.nu,
                'rho_ref': self.rho0,
                'c0_ref':  self.c0
            }
        )

        # 2) Rigid bodies: platform & two grippers
        def make_wall(name, center, size, normal):
            x, y, z = self.create_box(center, size)
            pa = get_particle_array(
                name=name,
                x=x, y=y, z=z,
                h=self.hdx*self.dx,
                m=1e12,
                rho=self.rho0,
                is_boundary=1,
                is_rigid=1
            )
            pa.add_property('nx', type='double', default=normal[0])
            pa.add_property('ny', type='double', default=normal[1])
            pa.add_property('nz', type='double', default=normal[2])
            return pa

        plat = make_wall('platform',
                         (0, 0, self.platform_size[2]/2),
                         self.platform_size,
                         normal=(0,0,1))
        g1 = make_wall('gripper1',
                       (-0.4, 0, self.platform_size[2]+0.5*self.gripper_size[2]),
                       self.gripper_size,
                       normal=(1,0,0))
        g2 = make_wall('gripper2',
                       ( 0.4, 0, self.platform_size[2]+0.5*self.gripper_size[2]),
                       self.gripper_size,
                       normal=(-1,0,0))
        self.gr1, self.gr2 = g1, g2

        return [block, plat, g1, g2]

    def create_scheme(self):
        # RigidBody3DScheme handles both block elasticity and rigid-body collisions
        scheme = RigidBody3DScheme(
            rigid_bodies=['platform','gripper1','gripper2'],
            elastic_solids=['block'],
            dim=self.dim,
            contact_force='DEM',
            kn=1e5,      # collision stiffness
            mu=0.5,      # friction
            en=0.2       # restitution
        )
        return SchemeChooser(default='rigid', rigid=scheme)

    def configure_scheme(self):
        # Solver settings
        self.scheme.configure_solver(dt=1e-4, tf=2.0, pfreq=200)

    def create_equations(self):
        # Let the scheme assemble all required equations
        eqns = self.scheme.get_equations()

        # Add gravity on the block if desired
        if self.gravity_enabled:
            eqns.append(
                Group(equations=[BodyForce(dest='block', sources=[], fx=0, fy=0, fz=-9.81)],
                      real=False)
            )
        return eqns

    def post_step(self, solver):
        # Manually move the gripper jaws
        dt = solver.dt
        half_b = 0.5*self.block_size[0]
        half_g = 0.5*self.gripper_size[0]
        target = -half_b - half_g + 0.01
        g1, g2 = self.gr1, self.gr2
        if g1.x[0] < target:
            g1.u[:] =  0.2; g2.u[:] = -0.2
            g1.w[:] = g2.w[:] = 0.0
        else:
            g1.u[:] = g2.u[:] = 0.0
            g1.w[:] = 0.3
        for g in (g1, g2):
            g.x += g.u*dt; g.y += g.v*dt; g.z += g.w*dt

if __name__ == '__main__':
    app = GraspDeformableBlock()
    app.run()
