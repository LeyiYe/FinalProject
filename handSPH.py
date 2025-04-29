#!/usr/bin/env python3
import numpy as np

from pysph.base.utils          import get_particle_array
from pysph.solver.application  import Application
from pysph.sph.scheme          import SchemeChooser
from pysph.sph.solid_mech.basic   import (
    ElasticSolidsScheme,
    get_particle_array_elastic_dynamics
)
from pysph.sph.equation        import Group
from pysph.sph.basic_equations import BodyForce
from pysph.sph.rigid_body      import RigidBodyWallCollision

class GraspDeformableBlock(Application):
    def initialize(self):
        self.dim    = 3
        self.dx     = 0.01
        self.hdx    = 1.3
        self.rho0   = 1000.0
        self.E      = 1e7
        self.nu     = 0.49
        self.c0     = 50.0

        self.block_size    = (0.3, 0.2, 0.1)
        self.platform_size = (1.0, 0.6, 0.02)
        self.gripper_size  = (0.05, 0.1, 0.2)

    def create_block(self, center, size):
        nx = max(3, int(round(size[0]/self.dx)))
        ny = max(3, int(round(size[1]/self.dx)))
        nz = max(3, int(round(size[2]/self.dx)))
        xs = np.linspace(center[0] - size[0]/2 + self.dx/2,
                         center[0] + size[0]/2 - self.dx/2, nx)
        ys = np.linspace(center[1] - size[1]/2 + self.dx/2,
                         center[1] + size[1]/2 - self.dx/2, ny)
        zs = np.linspace(center[2] - size[2]/2 + self.dx/2,
                         center[2] + size[2]/2 - self.dx/2, nz)
        x,y,z = np.meshgrid(xs, ys, zs, indexing='ij')
        return x.ravel(), y.ravel(), z.ravel()

    def create_particles(self):
        # 1) deformable block
        bx, by, bz = self.create_block(
            center=(0, 0, self.platform_size[2] + self.block_size[2]/2),
            size=self.block_size
        )
        block = get_particle_array_elastic_dynamics(
            name='block',
            x=bx, y=by, z=bz,
            h=self.hdx*self.dx,
            m=self.dx**3 * self.rho0,
            rho=self.rho0,
            constants={
                'E'      : self.E,
                'nu'     : self.nu,
                'rho_ref': self.rho0,
                'c0_ref' : self.c0
            }
        )

        # 1a) DEM‐collision extras on the block
        block.add_property('rad_s', default=self.dx*0.5)
        for pname in ('tang_disp_x','tang_disp_y','tang_disp_z',
                      'tang_velocity_x','tang_velocity_y','tang_velocity_z'):
            block.add_property(pname, type='double', default=0.0)

        # 1b) **** add the four arrays that RigidBodyWallCollision requires ****
        block.add_property('fx',         type='double', default=0.0)
        block.add_property('fy',         type='double', default=0.0)
        block.add_property('fz',         type='double', default=0.0)
        block.add_property('total_mass', type='double', default=0.0)
        # now fill total_mass with the sum of all per-particle masses:
        block.total_mass[:] = np.sum(block.m)

        # 2) rigid “walls” (platform + two jaws)
        def make_wall(name, center, size, normal):
            x,y,z = self.create_block(center, size)
            pa = get_particle_array(
                name=name,
                x=x, y=y, z=z,
                h=self.hdx*self.dx,
                m=1e12,
                rho=self.rho0,
                is_boundary=1,
                is_rigid=1,
                arho=1.0/self.rho0,
                cs=self.c0,
                z0=z.copy()
            )
            pa.add_property('nx', type='double', default=normal[0])
            pa.add_property('ny', type='double', default=normal[1])
            pa.add_property('nz', type='double', default=normal[2])
            return pa

        platform = make_wall(
            'platform',
            (0, 0, self.platform_size[2]/2),
            self.platform_size,
            normal=(0,0,1)
        )
        gripper1 = make_wall(
            'gripper1',
            (-0.4, 0, self.platform_size[2] + self.gripper_size[2]/2),
            self.gripper_size,
            normal=( 1,0,0)
        )
        gripper2 = make_wall(
            'gripper2',
            ( 0.4, 0, self.platform_size[2] + self.gripper_size[2]/2),
            self.gripper_size,
            normal=(-1,0,0)
        )

        return [block, platform, gripper1, gripper2]

    def create_scheme(self):
        elastic = ElasticSolidsScheme(
            elastic_solids=['block'],
            solids=[],
            dim=self.dim,
            artificial_stress_eps=0.5,
            xsph_eps=0.5
        )
        return SchemeChooser(default='elastic', elastic=elastic)

    def configure_scheme(self):
        self.scheme.configure_solver(dt=5e-5, tf=2.0, pfreq=50)

    def create_equations(self):
        eqns = self.scheme.get_equations()
        # gravity on the block
        eqns.append(Group(equations=[
            BodyForce(dest='block', sources=None, fx=0, fy=0, fz=-9.81)
        ], real=False))
        # DEM‐wall collisions
        eqns.append(Group(equations=[
            RigidBodyWallCollision('block', ['platform'], kn=1e4, mu=0.0, en=0.8),
            RigidBodyWallCollision('block', ['gripper1'], kn=1e4, mu=0.0, en=0.8),
            RigidBodyWallCollision('block', ['gripper2'], kn=1e4, mu=0.0, en=0.8),
        ], real=False))
        return eqns

    def post_step(self, solver):
        g1, g2 = self.particles[2], self.particles[3]
        dt = solver.dt
        half_block = 0.5*self.block_size[0]
        half_grip  = 0.5*self.gripper_size[0]
        target = -half_block - half_grip + 0.02

        if g1.x[0] < target:
            g1.u[:] =  0.2; g2.u[:] = -0.2
            g1.w[:] = g2.w[:] = 0.0
        else:
            g1.u[:] = g2.u[:] = 0.0
            g1.w[:] = g2.w[:] = 0.3

        for gr in (g1, g2):
            gr.x  += gr.u * dt
            gr.y  += gr.v * dt
            gr.z  += gr.w * dt
            gr.z0 += gr.w * dt

if __name__ == '__main__':
    app = GraspDeformableBlock()
    app.run()
